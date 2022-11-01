#!/usr/bin/env python
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

from models import UNet61_8s, UNet43_8s, UNet47_8s, ResNet43_8s
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
                 learning_rate=1e-4, without_global=False, strategy=None):
        self.preprocess = preprocess
        self.use_goal_image = use_goal_image

        self.gamma = 0.8
        self.weight = 1

        self.out_logits = out_logits
        self.without_global = without_global

        self.unet = unet
        self.strategy = strategy

        input_shape = np.array(input_shape)
        input_shape = tuple(input_shape)

        if self.unet:
            in0, out0, global_feat = UNet43_8s(input_shape, 256, prefix='critic_s0_d1_')
            self.model = tf.keras.Model(inputs=[in0], outputs=[out0, global_feat])
            if self.without_global:
                self.conv_seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                           input_shape=(input_shape[0], input_shape[1], 256 + 256), name='critic_1'),
                    tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(input_shape[0], input_shape[1], 256),
                                           name='critic_2'),
                ])
            else:
                self.conv_seq = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                           input_shape=(input_shape[0], input_shape[1], 256 + 256 + 512), name='critic_1'),
                    tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(input_shape[0], input_shape[1], 256),
                                           name='critic_2'),
                ])
        else:
            in0, out0 = ResNet43_8s(input_shape, 256, prefix='s0_d1_')
            self.model = tf.keras.Model(inputs=[in0], outputs=[out0])
            self.conv_seq = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=256, kernel_size=1, activation='relu',
                                       input_shape=(input_shape[0], input_shape[1], 256 + 256)),
                tf.keras.layers.Conv2D(filters=self.out_logits, kernel_size=1, input_shape=(input_shape[0], input_shape[1], 256)),
            ])

        self.optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metric = tf.keras.metrics.Mean(name='transport_loss')

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
            global_feat = tf.tile(global_feat, [1, feat.shape[1], feat.shape[2], 1])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, feat.shape[1], feat.shape[2], 1])
            # ipdb.set_trace()
            # print(f'input img {in_img.shape} feat {in_img.shape} select_feat {select_feat.shape} p0 {p0}')
            if self.without_global:
                all_feat = tf.concat([feat, select_feat], axis=-1)
            else:
                all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        else:
            feat = self.model([in_tensor])
            select_feat = feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :]
            select_feat = tf.tile(select_feat, [1, feat.shape[1], feat.shape[2], 1])
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
            input_data = self.preprocess(in_img_batch[i])                     # (320,320,4)
            input_shape = (1,) + input_data.shape
            input_data = input_data.reshape(input_shape)                    # (1,320,320,4)
            in_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)  # (1,320,320,4)
            in_tensor_batch.append(in_tensor)
        in_tensor = tf.concat(in_tensor_batch, axis=0)                      # (batch,320,320,4)

        # Pass `in_tensor` twice, get crop from `kernel_before_crop` (not `input_data`).
        if self.unet:
            feat, global_feat = self.model([in_tensor])
            global_feat = tf.tile(global_feat, [1, feat.shape[1], feat.shape[2], 1])
            select_feat_batch = []
            for i in range(batch_len):
                select_feat = feat[i:i+1, p0_batch[i][0]:p0_batch[i][0] + 1, p0_batch[i][1]:p0_batch[i][1] + 1, :]
                select_feat = tf.tile(select_feat, [1, feat.shape[1], feat.shape[2], 1])
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
                select_feat = tf.tile(select_feat, [1, feat.shape[1], feat.shape[2], 1])
                select_feat_batch.append(select_feat)
                # ipdb.set_trace()
            select_feat = tf.concat(select_feat_batch, axis=0)
            all_feat = tf.concat([feat, select_feat], axis=-1)
        output = self.conv_seq(all_feat)

        return output

    def forward_with_plist(self, in_img, plist):
        len_p = len(plist)
        if self.unet:
            feat, global_feat = self.forward_with_feat(in_img)
            global_feat = tf.tile(global_feat, [len_p, feat.shape[1], feat.shape[2], 1])
            select_feat_list = []
            for idx_p0 in range(len_p):
                p0 = plist[idx_p0]
                select_feat = tf.tile(feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :], [1, feat.shape[1], feat.shape[2], 1])
                select_feat_list.append(select_feat)
            select_feat = tf.concat(select_feat_list, axis=0)
            feat = tf.tile(feat, [len_p, 1, 1, 1])
            if self.without_global:
                all_feat = tf.concat([feat, select_feat], axis=-1)
            else:
                all_feat = tf.concat([feat, select_feat, global_feat], axis=-1)
        else:
            feat = self.forward_with_feat(in_img)
            select_feat_list = []
            for idx_p0 in range(len_p):
                p0 = plist[idx_p0]
                select_feat = tf.tile(feat[:, p0[0]:p0[0] + 1, p0[1]:p0[1] + 1, :], [1, feat.shape[1], feat.shape[2], 1])
                select_feat_list.append(select_feat)
            select_feat = tf.concat(select_feat_list, axis=0)
            feat = tf.tile(feat, [len_p, 1, 1, 1])
            all_feat = tf.concat([feat, select_feat], axis=-1)

        output = self.conv_seq(all_feat)

        return output

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

    def train_batch(self, in_img_batch, p0_batch, p1_list_batch, reward_batch, step_batch, validate=False):
        if validate:
            self.validate_metric.reset_states()
        else:
            self.metric.reset_states()
        if not validate:
            with tf.GradientTape() as tape:
                QQ_cur = self.forward_batch(in_img_batch, p0_batch)
                loss = None
                batch_len = len(in_img_batch)

                for i in range(batch_len):
                    reward = reward_batch[i].copy()
                    # img = in_img_batch[i][:, :, :-1]
                    #
                    # for u in range(max(0, p0_batch[i][0] - 4), min(160, p0_batch[i][0] + 4)):
                    #     for v in range(max(0, p0_batch[i][1] - 4), min(160, p0_batch[i][1] + 4)):
                    #         img[u][v] = (255, 0, 0)
                    #
                    # vis_critic = np.array(QQ_cur[i])
                    # argmax = np.argmax(vis_critic)
                    # argmax = np.unravel_index(argmax, shape=vis_critic.shape)
                    # p1_pixel = argmax[0:2]
                    # vis_critic = vis_critic - np.min(vis_critic)
                    # vis_critic = 255 * vis_critic / np.max(vis_critic)
                    # vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)
                    #
                    # for u in range(max(0, p1_pixel[0] - 4), min(160, p1_pixel[0] + 4)):
                    #     for v in range(max(0, p1_pixel[1] - 4), min(160, p1_pixel[1] + 4)):
                    #         img[u][v] = (255, 255, 255)
                    #
                    # vis_img = np.concatenate((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), vis_critic), axis=1)
                    #
                    # cv2.imwrite(f'./visual/debug-test-aff_critic-{reward}.jpg', vis_img)
                    # print("save to" + f'./visual/debug-test-aff_critic-{reward}.jpg')
                    # print("reward", reward)
                    p1_list = p1_list_batch[i].copy()
                    len_q = len(p1_list)
                    for idx in range(len_q):
                        # ipdb.set_trace()
                        if loss is None:
                            loss = tf.keras.losses.MAE(reward[idx], QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1], :]) / len_q
                        else:
                            loss = loss + tf.keras.losses.MAE(reward[idx],
                                                              QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1], :]) / len_q
                        # print(QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1]])
                        # print("reward:", reward[idx])
                # print(loss)
                loss = tf.reduce_mean(loss)
                loss /= batch_len
                # print(loss)

                grad = tape.gradient(loss, self.model.trainable_variables)
                self.optim.apply_gradients(zip(grad, self.model.trainable_variables))
        else:
            QQ_cur = self.forward_batch(in_img_batch, p0_batch)
            loss = None
            batch_len = len(in_img_batch)

            for i in range(batch_len):
                reward = reward_batch[i].copy()
                # print("reward", reward)
                p1_list = p1_list_batch[i].copy()
                len_q = len(p1_list)
                for idx in range(len_q):
                    # ipdb.set_trace()
                    if loss is None:
                        loss = tf.keras.losses.MAE(reward[idx],
                                                   QQ_cur[i:i + 1, p1_list[idx][0], p1_list[idx][1], :]) / len_q
                    else:
                        loss = loss + tf.keras.losses.MAE(reward[idx],
                                                          QQ_cur[i:i + 1, p1_list[idx][0], p1_list[idx][1], :]) / len_q
                    # print(QQ_cur[i:i+1, p1_list[idx][0], p1_list[idx][1]])
                    # print("reward:", reward[idx])
            loss = tf.reduce_mean(loss)
        if validate:
            self.validate_metric(loss)
        else:
            self.metric(loss)
        return np.float32(loss)

    def save(self, fname):
        self.model.save(fname)

    def load(self, fname):
        self.model.load_weights(fname)
