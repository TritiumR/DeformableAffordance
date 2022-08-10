#!/usr/bin/env python
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

from models import ResNet43_8s, ResNet61_8s, ResNet53_8s, ResNet36_4s, UNet43_8s
import random

import ipdb


class Critic_MLP:
    """Transporter for the placing module.

    By default, don't use per-pixel loss, meaning that pixels collectively
    define the set of possible classes, and we just pick one of them. Per
    pixel means each pixel has positive and negative (and maybe a third
    neutral) label. Also, usually rotations=24 and crop_size=64.

    Also by default, we crop and then pass the input to the query network. We
    could also pass the full image to the query network, and then crop it.

    Note the *two* FCNs here (in0, out0, in1, out1) because this Transport
    model has two streams. Then `self.model` gets both in0 and in1 as input.
    Shapes below assume we get crops and then pass them to the query net.

    Image-to-image   (phi) = {in,out}0
    Kernel-to-kernel (psi) = {in,out}1
    input_shape: (384, 224, 6)
    kernel_shape: (64, 64, 6)
    in0: Tensor("input_2:0", shape=(None, 384, 224, 6), dtype=float32)
    in1: Tensor("input_3:0", shape=(None, 64, 64, 6), dtype=float32)
    out0: Tensor("add_31/Identity:0", shape=(None, 384, 224, 3), dtype=float32)
    out1: Tensor("add_47/Identity:0", shape=(None, 64, 64, 3), dtype=float32)

    The batch size for the forward pass is the number of rotations, by
    default 24 (they changed to 36 later).
    """

    def __init__(self, input_shape, preprocess, use_goal_image=False, out_logits=1, unet=1,
                 learning_rate=1e-4, without_global=False):
        self.preprocess = preprocess
        self.use_goal_image = use_goal_image

        self.gamma = 0.8
        self.weight = 100

        self.out_logits = out_logits
        self.without_global = without_global

        self.unet = unet

        input_shape = np.array(input_shape)
        input_shape = tuple(input_shape)

        # 2 fully convolutional ResNets [43 layers and stride 8]
        if self.unet:
            in0, out0, global_feat = UNet43_8s(input_shape, 256, prefix='critic_s0_d1_')
            self.model = tf.keras.Model(inputs=[in0], outputs=[out0, global_feat])
            if self.without_global:
                self.conv_seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                           input_shape=(320, 320, 256 + 256), name='critc_1'),  # last dimension -512
                    tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(320, 320, 256),
                                           name='critc_2'),
                ])
            else:
                self.conv_seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', input_shape=(320, 320, 256 + 256 + 512), name='critc_1'),
                    tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(320, 320, 256), name='critc_2'),
                ])
        else:
            in0, out0 = ResNet43_8s(input_shape, 256, prefix='s0_d1_')
            self.model = tf.keras.Model(inputs=[in0], outputs=[out0])
            self.conv_seq = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu', input_shape=(320, 320, 256 + 256)),
                tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(320, 320, 256)),
            ])

        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metric = tf.keras.metrics.Mean(name='transport_loss')
        self.validate_metric = tf.keras.metrics.Mean(name='validate_loss')

    def forward(self, in_img, p0):
        """Forward pass.

        dissecting this a bit more since it's the key technical
        contribution. Relevant shapes and info:

            in_img: (320,160,6)
            p: integer pixels on in_img, e.g., [158, 30]
            self.padding: [[32,32],[32,32],0,0]], with shape (3,2)

        How does the cross-convolution work? We get the output from BOTH
        streams of the network. Then, the output for one stream is the set of
        convolutional kernels for the other. Call tf.nn.convolution. That's
        it, and the entire operation is differentiable, so that gradients
        will apply to both of the two Transport FCN streams.
        """
        # ipdb.set_trace()
        input_data = self.preprocess(in_img)                        # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)

        # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
        if self.unet:
            feat, global_feat = self.model([in_tensor])
            global_feat = tf.tile(global_feat, [1, 320, 320, 1])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, 320, 320, 1])
            # ipdb.set_trace()
            # print(f'input img {in_img.shape} feat {in_img.shape} select_feat {select_feat.shape} p0 {p0}')
            if self.without_global:
                all_feat = tf.concat([feat, select_feat], axis=-1)
            else:
                all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        else:
            feat = self.model([in_tensor])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, 320, 160, 1])
            all_feat = tf.concat([feat, select_feat], axis=-1)
        output = self.conv_seq(all_feat)

        return output

    def forward_batch(self, in_img_batch, p0_batch):
        """Forward pass with batch.

            in_img: (320,160,6)
            p: integer pixels on in_img, e.g., [158, 30]
            self.padding: [[32,32],[32,32],0,0]], with shape (3,2)

        How does the cross-convolution work? We get the output from BOTH
        streams of the network. Then, the output for one stream is the set of
        convolutional kernels for the other. Call tf.nn.convolution. That's
        it, and the entire operation is differentiable, so that gradients
        will apply to both of the two Transport FCN streams.
        """
        # ipdb.set_trace()
        batch_len = len(in_img_batch)
        in_tensor_batch = []
        for i in range(batch_len):
            input_data = self.preprocess(in_img_batch[i])                     # (384,224,6)
            input_shape = (1,) + input_data.shape
            input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
            in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)
            in_tensor_batch.append(in_tensor)
        in_tensor = tf.concat(in_tensor_batch, axis=0)                      # (batch,384,224,6)

        # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
        if self.unet:
            feat, global_feat = self.model([in_tensor])
            global_feat = tf.tile(global_feat, [1, 320, 320, 1])
            select_feat_batch = []
            for i in range(batch_len):
                select_feat = feat[i:i+1, p0_batch[i][0]:p0_batch[i][0] + 1, p0_batch[i][1]:p0_batch[i][1] + 1, :]
                select_feat = tf.tile(select_feat, [1, 320, 320, 1])
                select_feat_batch.append(select_feat)
                # ipdb.set_trace()
            select_feat = tf.concat(select_feat_batch, axis=0)
            if self.without_global:
                all_feat = tf.concat([feat, select_feat], axis=-1)
            else:
                all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        else:
            feat = self.model([in_tensor])
            select_feat_batch = []
            for i in range(batch_len):
                select_feat = feat[i:i+1, p0_batch[i][0]:p0_batch[i][0] + 1, p0_batch[i][1]:p0_batch[i][1] + 1, :]
                select_feat = tf.tile(select_feat, [1, 320, 320, 1])
                select_feat_batch.append(select_feat)
                # ipdb.set_trace()
            select_feat = tf.concat(select_feat_batch, axis=0)
            all_feat = tf.concat([feat, select_feat], axis=-1)
        output = self.conv_seq(all_feat)

        return output

    def forward_aff_gt(self, in_img, p0):
        """Forward pass.

            in_img: (320,160,6)
            p: integer pixels on in_img, e.g., [158, 30]
            self.padding: [[32,32],[32,32],0,0]], with shape (3,2)

        How does the cross-convolution work? We get the output from BOTH
        streams of the network. Then, the output for one stream is the set of
        convolutional kernels for the other. Call tf.nn.convolution. That's
        it, and the entire operation is differentiable, so that gradients
        will apply to both of the two Transport FCN streams.
        """
        # ipdb.set_trace()
        input_data = self.preprocess(in_img)                        # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)

        # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
        if self.unet:
            feat, global_feat = self.model([in_tensor])
            global_feat = tf.tile(global_feat, [1, 320, 320, 1])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, 320, 320, 1])
            # ipdb.set_trace()
            # print(f'input img {in_img.shape} feat {in_img.shape} select_feat {select_feat.shape} p0 {p0}')
            all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        else:
            feat = self.model([in_tensor])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, 320, 320, 1])
            all_feat = tf.concat([feat, select_feat], axis=-1)
        output = self.conv_seq(all_feat)

        max_dis, avg_dis, convex = output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2]
        critic_score = convex * self.cvx_weight - max_dis * self.max_weight - avg_dis * self.avg_weight

        return critic_score

    def forward_aff_batch_gt(self, in_img, plist):
        # ipdb.set_trace()
        input_data = self.preprocess(in_img)                        # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)
        feat, global_feat = self.model([in_tensor])

        len_p = len(plist)
        global_feat = tf.tile(global_feat, [len_p, 320, 320, 1])
        select_feat_list = []
        for idx_p0 in range(len_p):
            p0 = plist[idx_p0]
            select_feat = tf.tile(feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :], [1, 320, 320, 1])
            select_feat_list.append(select_feat)
        select_feat = tf.concat(select_feat_list, axis=0)
        feat = tf.tile(feat, [len_p, 1, 1, 1])
        all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        output = self.conv_seq(all_feat)

        max_dis, avg_dis, convex = output[:, :, :, 0], output[:, :, :, 1], output[:, :, :, 2]
        critic_score = convex * self.cvx_weight - max_dis * self.max_weight - avg_dis * self.avg_weight

        return critic_score

    def forward_with_feat(self, in_img):
        # ipdb.set_trace()
        input_data = self.preprocess(in_img)                        # (384,224,6)
        input_shape = (1,) + input_data.shape
        input_data = input_data.reshape(input_shape)                    # (1,384,224,6)
        in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,384,224,6)

        # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
        if self.unet:
            feat, global_feat = self.model([in_tensor])
            return feat, global_feat
        else:
            feat = self.model([in_tensor])
            return feat

    def forward_with_plist(self, in_img, plist):
        len_p = len(plist)
        feat, global_feat = self.forward_with_feat(in_img)
        global_feat = tf.tile(global_feat, [len_p, 320, 320, 1])
        select_feat_list = []
        for idx_p0 in range(len_p):
            p0 = plist[idx_p0]
            select_feat = tf.tile(feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :], [1, 320, 320, 1])
            select_feat_list.append(select_feat)
        select_feat = tf.concat(select_feat_list, axis=0)
        feat = tf.tile(feat, [len_p, 1, 1, 1])
        if self.without_global:
            all_feat = tf.concat([feat, select_feat], axis=-1)
        else:
            all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)

        output = self.conv_seq(all_feat)

        return output

    def train_batch(self, in_img_batch, p0_batch, p1_list_batch, reward_batch, step_batch):
        self.metric.reset_states()
        with tf.GradientTape() as tape:
            QQ_cur = self.forward_batch(in_img_batch, p0_batch)
            loss = None
            batch_len = len(in_img_batch)

            for i in range(batch_len):
                reward = reward_batch[i].copy() * self.weight
                p1_list = p1_list_batch[i].copy()
                len_q = len(p1_list)
                for idx in range(len_q):
                    # ipdb.set_trace()
                    if loss is None:
                        loss = tf.keras.losses.MAE(reward[idx], QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1], :])
                    else:
                        loss = loss + tf.keras.losses.MAE(reward[idx],
                                                          QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1], :])
                    print(QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1]])
                    print("reward:", reward[idx])
            loss = loss / len_q
            loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, self.model.trainable_variables)
            self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        self.metric(loss)
        return np.float32(loss)

    def train_phy_batch(self, in_img_batch, p0_batch, p1_list_batch, distances_batch, nxt_distances_batch, cvx_batch, nxt_cvx_batch, ep_batch, ep_len_batch, is_validate=False, task='cable-ring'):
        if is_validate:
            self.validate_metric.reset_states()
        else:
            self.metric.reset_states()

        if task == 'cable-ring' or 'cable-ring-notarget' or task == 'bag-alone-open' or task == 'cloth-flat':
            nxt_max_dis_batch = []
            nxt_avg_dis_batch = []
            idx_list_batch = []

            batch_len = len(in_img_batch)
            for i in range(batch_len):
                nxt_cvx = np.array(nxt_cvx_batch[i].copy())
                if task != 'cloth-flat':
                    nxt_cvx = nxt_cvx * 1000
                else:
                    nxt_cvx = nxt_cvx * 20
                nxt_cvx_batch[i] = nxt_cvx

                len_q = len(p1_list_batch[i])
                if self.balanced:
                    idx_list = [0, random.randint(1, len_q - 1)]
                else:
                    idx_list = [idx for idx in range(len_q)]
                idx_list_batch.append(idx_list.copy())

                if ep_batch[i] == ep_len_batch[i]:
                    print("?" * 30)
                else:
                    nxt_max_dis = nxt_distances_batch[i].max(axis=1) * self.alpha_max_dis
                    nxt_avg_dis = nxt_distances_batch[i].mean(axis=1) * self.alpha_avg_dis
                    nxt_max_dis_batch.append(nxt_max_dis.copy())
                    nxt_avg_dis_batch.append(nxt_avg_dis.copy())

            with tf.GradientTape() as tape:
                loss_max_dis = None
                loss_avg_dis = None
                loss_cvx = None
                tot_loss = 0
                QQ_cur = self.forward_batch(in_img_batch, p0_batch)
                for i in range(batch_len):
                    for idx in idx_list_batch[i]:
                        if loss_max_dis is None:
                            loss_max_dis = tf.keras.losses.MAE(nxt_max_dis_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], :1])
                        else:
                            loss_max_dis = loss_max_dis + tf.keras.losses.MAE(nxt_max_dis_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], :1])
                        if loss_avg_dis is None:
                            loss_avg_dis = tf.keras.losses.MAE(nxt_avg_dis_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 1:2])
                        else:
                            loss_avg_dis = loss_avg_dis + tf.keras.losses.MAE(nxt_avg_dis_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 1:2])
                        if loss_cvx is None:
                            loss_cvx = tf.keras.losses.MAE(nxt_cvx_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 2:3])
                        else:
                            loss_cvx = loss_cvx + tf.keras.losses.MAE(nxt_cvx_batch[i][idx], QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 2:3])
                        print("cvx:", idx, nxt_cvx_batch[i][idx], float(QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 2:3]))
                        print("max:", idx, nxt_max_dis_batch[i][idx], float(QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], :1]))
                        print("avg:", idx, nxt_avg_dis_batch[i][idx], float(QQ_cur[i:i+1, p1_list_batch[i][idx][0], p1_list_batch[i][idx][1], 1:2]))
                    one_batch_loss = loss_max_dis + loss_avg_dis + loss_cvx
                    tot_loss += one_batch_loss / len(idx_list_batch[i])

                tot_loss /= (batch_len)
                tot_loss = tf.reduce_mean(tot_loss)

                grad = tape.gradient(tot_loss, self.model.trainable_variables)
                self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        elif task == 'cable-shape':
            len_q = len(p1_list)
            if self.balanced:
                idx_list = [0, random.randint(1, len_q - 1)]
            else:
                idx_list = [idx for idx in range(len_q)]
            len_idx_list = len(idx_list)
            if ep == ep_len:
                print("?" * 30)
            else:
                nxt_max_dis = nxt_distances.max(axis=1) * self.alpha_max_dis
                nxt_avg_dis = nxt_distances.mean(axis=1) * self.alpha_avg_dis
                # ipdb.set_trace()
                if is_validate:
                    QQ_cur = self.forward(in_img, p0)
                    # ipdb.set_trace()
                    loss_max_dis = None
                    loss_avg_dis = None
                    loss_cvx = None
                    for idx in idx_list:
                        if loss_max_dis is None:
                            loss_max_dis = tf.keras.losses.MAE(nxt_max_dis[idx],
                                                               QQ_cur[:, p1_list[idx][0], p1_list[idx][1], :1])
                        else:
                            loss_max_dis = loss_max_dis + tf.keras.losses.MAE(nxt_max_dis[idx],
                                                                              QQ_cur[:, p1_list[idx][0],
                                                                              p1_list[idx][1],
                                                                              :1])
                        if loss_avg_dis is None:
                            loss_avg_dis = tf.keras.losses.MAE(nxt_avg_dis[idx],
                                                               QQ_cur[:, p1_list[idx][0], p1_list[idx][1], 1:2])
                        else:
                            loss_avg_dis = loss_avg_dis + tf.keras.losses.MAE(nxt_avg_dis[idx],
                                                                              QQ_cur[:, p1_list[idx][0],
                                                                              p1_list[idx][1],
                                                                              1:2])
                        print("max:", idx, nxt_max_dis[idx], float(QQ_cur[:, p1_list[idx][0], p1_list[idx][1], :1]))
                        print("avg:", idx, nxt_avg_dis[idx], float(QQ_cur[:, p1_list[idx][0], p1_list[idx][1], 1:2]))
                    tot_loss = loss_max_dis + loss_avg_dis
                    tot_loss = tot_loss / len_idx_list
                    tot_loss = tf.reduce_mean(tot_loss)
                else:
                    with tf.GradientTape() as tape:
                        QQ_cur = self.forward(in_img, p0)
                        # ipdb.set_trace()
                        loss_max_dis = None
                        loss_avg_dis = None
                        loss_cvx = None
                        for idx in idx_list:
                            if loss_max_dis is None:
                                loss_max_dis = tf.keras.losses.MAE(nxt_max_dis[idx],
                                                                   QQ_cur[:, p1_list[idx][0], p1_list[idx][1], :1])
                            else:
                                loss_max_dis = loss_max_dis + tf.keras.losses.MAE(nxt_max_dis[idx],
                                                                                  QQ_cur[:, p1_list[idx][0],
                                                                                  p1_list[idx][1], :1])
                            if loss_avg_dis is None:
                                loss_avg_dis = tf.keras.losses.MAE(nxt_avg_dis[idx],
                                                                   QQ_cur[:, p1_list[idx][0], p1_list[idx][1], 1:2])
                            else:
                                loss_avg_dis = loss_avg_dis + tf.keras.losses.MAE(nxt_avg_dis[idx],
                                                                                  QQ_cur[:, p1_list[idx][0],
                                                                                  p1_list[idx][1], 1:2])
                            print("max:", idx, nxt_max_dis[idx], float(QQ_cur[:, p1_list[idx][0], p1_list[idx][1], :1]))
                            print("avg:", idx, nxt_avg_dis[idx], float(QQ_cur[:, p1_list[idx][0], p1_list[idx][1], 1:2]))
                        tot_loss = loss_max_dis + loss_avg_dis
                        tot_loss = tot_loss / len_idx_list
                        tot_loss = tf.reduce_mean(tot_loss)

                        grad = tape.gradient(tot_loss, self.model.trainable_variables)
                        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        if is_validate:
            self.validate_metric(tot_loss)
        else:
            self.metric(tot_loss)

        return np.float32(tot_loss), float(np.float32(loss_max_dis)), float(np.float32(loss_avg_dis)), float(np.float32(loss_cvx))


    def train_raw_critic(self, in_img, p, q, nxt_obs, nxt_p, nxt_q, distance, nxt_distance, ep, ep_len):
        """Transport pixel p to pixel q.

          Args:
            input:
            depth_image:
            p: pixel (y, x)
            q: pixel (y, x)
          Returns:
            A `Tensor`. Has the same type as `input`.

        the `in_img` will include the color and depth. Much is
        similar to the attention model if we're not using the per-pixel loss:
        (a) forward pass, (b) get angle discretizations [though we set only 1
        rotation for the picking model], (c) make the label consider
        rotations in the last axis, but only provide the label to one single
        (pixel,rotation) combination, (d) follow same exact steps for the
        non-per pixel loss otherwise. The output reshaping to (1, ...) is
        done in the attention model forward pass, but not in the transport
        forward pass. Note the `1` meaning a batch size of 1.
        """
        self.metric.reset_states()
        max_dis = distance.max()
        avg_dis = distance.mean()
        if nxt_obs is None:
            reward = 300 - max_dis * 1000 - avg_dis * 1000
            with tf.GradientTape() as tape:
                QQ_cur = self.forward(in_img, p, q)
                QQ_tar = reward
                loss = tf.keras.losses.MAE(QQ_tar, QQ_cur)
                loss = tf.reduce_mean(loss)
                print("QQ_cur:", QQ_cur[0][0].numpy())
                print("QQ_tar:", QQ_tar)
        else:
            nxt_max_dis = nxt_distance.max()
            nxt_avg_dis = nxt_distance.mean()
            reward = 1000 * (avg_dis + max_dis - nxt_avg_dis - nxt_max_dis)
            with tf.GradientTape() as tape:
                QQ_cur = self.forward(in_img, p, q)
                QQ_nxt = self.forward(nxt_obs, nxt_p, nxt_q)
                if self.fix_target:
                    QQ_nxt = QQ_nxt[0][0].numpy()
                # QQ_tar = self.gamma * QQ_nxt + reward
                QQ_tar = QQ_nxt - 50 + reward

                print("QQ_cur:", QQ_cur[0][0].numpy())
                if self.fix_target:
                    print("QQ_nxt:", QQ_nxt)
                    print("QQ_tar:", QQ_tar)
                else:
                    print("QQ_nxt:", QQ_nxt[0][0].numpy())
                    print("QQ_tar:", QQ_tar[0][0].numpy())

                # itheta = theta / (2 * np.pi / self.num_rotations)
                # itheta = np.int32(np.round(itheta)) % self.num_rotations

                # label_size = in_img.shape[:2] + (self.num_rotations,)
                # label = np.zeros(label_size)
                # label[q[0], q[1], itheta] = 1
                #
                # label = label.reshape(1, np.prod(label.shape))
                # label = tf.convert_to_tensor(label, dtype=tf.float32)
                # output = tf.reshape(output, (1, np.prod(output.shape)))
                loss = tf.keras.losses.MAE(QQ_tar, QQ_cur)
                loss = tf.reduce_mean(loss)

        grad = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grad, self.model.trainable_variables))

        self.metric(loss)

        print("reward:", reward)

        return np.float32(loss)

    # def get_se2(self, num_rotations, pivot):
    #     '''
    #     Get SE2 rotations discretized into num_rotations angles counter-clockwise.
    #     '''
    #     rvecs = []
    #     for i in range(num_rotations):
    #         theta = i * 2 * np.pi / num_rotations
    #         rmat = utils.get_image_transform(theta, (0, 0), pivot)
    #         rvec = rmat.reshape(-1)[:-1]
    #         rvecs.append(rvec)
    #     return np.array(rvecs, dtype=np.float32)

    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model.load_weights(fname)

    #-------------------------------------------------------------------------
    # Visualization.
    #-------------------------------------------------------------------------

    def visualize_images(self, p, in_img, input_data, crop):
        def get_itheta(theta):
            itheta = theta / (2 * np.pi / self.num_rotations)
            return np.int32(np.round(itheta)) % self.num_rotations

        plt.subplot(1, 3, 1)
        plt.title(f'Perturbed', fontsize=15)
        plt.imshow(np.array(in_img[:, :, :3]).astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.title(f'Process/Pad', fontsize=15)
        plt.imshow(input_data[0, :, :, :3])
        plt.subplot(1, 3, 3)
        # Let's stack two crops together.
        theta1 = 0.0
        theta2 = 90.0
        itheta1 = get_itheta(theta1)
        itheta2 = get_itheta(theta2)
        crop1 = crop[itheta1, :, :, :3]
        crop2 = crop[itheta2, :, :, :3]
        barrier = np.ones_like(crop1)
        barrier = barrier[:4, :, :] # white barrier of 4 pixels
        stacked = np.concatenate((crop1, barrier, crop2), axis=0)
        plt.imshow(stacked)
        plt.title(f'{theta1}, {theta2}', fontsize=15)
        plt.suptitle(f'pick: {p}', fontsize=15)
        plt.tight_layout()
        plt.show()
        #plt.savefig('viz.png')

    def visualize_transport(self, p, in_img, input_data, crop, kernel):
        """Like the attention map, let's visualize the transport data from a
        trained model.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode,
        red=hottest (highest attention values), green=medium, blue=lowest.

        See also:
        https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.subplot.html

        crop.shape: (24,64,64,6)
        kernel.shape = (65,65,3,24)
        """
        def colorize(img):
            # I don't think we have to convert to BGR here...
            img = img - np.min(img)
            img = 255 * img / np.max(img)
            img = cv2.applyColorMap(np.uint8(img), cv2.COLORMAP_RAINBOW)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        kernel = (tf.transpose(kernel, [3, 0, 1, 2])).numpy()

        # Top two rows: crops from processed RGBD. Bottom two: output from FCN.
        nrows = 4
        ncols = 12
        assert self.num_rotations == nrows * (ncols / 2)
        idx = 0
        fig, ax = plt.subplots(nrows, ncols, figsize=(12, 6))
        for _ in range(nrows):
            for _ in range(ncols):
                plt.subplot(nrows, ncols, idx+1)
                plt.axis('off')  # Ah, you need to put this here ...
                if idx < self.num_rotations:
                    plt.imshow(crop[idx, :, :, :3])
                else:
                    # Offset because idx goes from 0 to (rotations * 2) - 1.
                    _idx = idx - self.num_rotations
                    processed = colorize(img=kernel[_idx, :, :, :])
                    plt.imshow(processed)
                idx += 1
        plt.tight_layout()
        plt.show()

    def visualize_logits(self, logits):
        """Given logits (BEFORE tf.nn.convolution), get heatmap.

        Here we apply a softmax to make it more human-readable. However, the
        tf.nn.convolution with the learned kernels happens without a softmax
        on the logits.
        """
        original_shape = logits.shape
        logits = tf.reshape(logits, (1, np.prod(original_shape)))
        logits = tf.nn.softmax(logits)
        vis_transport = np.float32(logits).reshape(original_shape)
        vis_transport = vis_transport[0]
        vis_transport = vis_transport - np.min(vis_transport)
        vis_transport = 255 * vis_transport / np.max(vis_transport)
        vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
        # Only if we're saving with cv2.imwrite
        #vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
        plt.subplot(1, 1, 1)
        plt.title(f'Logits', fontsize=15)
        plt.imshow(vis_transport)
        plt.tight_layout()
        plt.show()

    def get_transport_heatmap(self, transport):
        """Given transport output, get a human-readable heatmap.

        https://docs.opencv.org/master/d3/d50/group__imgproc__colormap.html
        In my normal usage, the attention is already softmax-ed but just be
        aware in case it's not. Also be aware of RGB vs BGR mode. We should
        ensure we're in BGR mode before saving. Also with RAINBOW mode, red =
        hottest (highest attention values), green=medium, blue=lowest.
        """
        # Options: cv2.COLORMAP_PLASMA, cv2.COLORMAP_JET, etc.
        #transport = tf.reshape(transport, (1, np.prod(transport.shape)))
        #transport = tf.nn.softmax(transport)
        assert transport.shape == (320, 160, self.num_rotations), transport.shape
        vis_images = []
        for idx in range(self.num_rotations):
            t_img = transport[:, :, idx]
            vis_transport = np.float32(t_img)
            vis_transport = vis_transport - np.min(vis_transport)
            vis_transport = 255 * vis_transport / np.max(vis_transport)
            vis_transport = cv2.applyColorMap(np.uint8(vis_transport), cv2.COLORMAP_RAINBOW)
            vis_transport = cv2.cvtColor(vis_transport, cv2.COLOR_RGB2BGR)
            vis_images.append(vis_transport)
        return vis_images