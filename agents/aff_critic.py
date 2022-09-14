#!/usr/bin/env python
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from models import Affordance, Critic_MLP
from subprocess import call
from agents import agent_utils
from softgym import dataset
import copy
import random
import math

from scipy.spatial import ConvexHull

import tensorflow as tf

import ipdb


class AffCritic:
    def __init__(self, name, task, critic_pick=False, random_pick=False, expert_pick=False, use_goal_image=False,
                 out_logits=1, step=1, strategy=None):
        """Creates Transporter agent with attention and transport modules."""
        self.name = name
        self.task = task
        self.step = step
        self.total_iter = 0
        if use_goal_image:
            self.input_shape = (320, 320, 8)
        else:
            self.input_shape = (320, 320, 4)
        self.models_dir = os.path.join('checkpoints', self.name + f'-step-{self.step}')

        self.expert_pick = expert_pick
        self.critic_pick = critic_pick
        self.critic_num = 33
        self.random_pick = random_pick

        self.out_logits = out_logits
        self.use_goal_image = use_goal_image
        self.strategy = strategy

    def compute_reward(self, task, metric, step, curr_obs=None):
        m_len = len(metric)
        reward = []

        if curr_obs is None:
            for i in range(0, m_len):
                curr_percent = metric[i][1] * 50
                reward.append(curr_percent)
        else:
            for i in range(0, m_len):
                attention = self.next_model.forward(curr_obs[i].copy())

                # img_obs = obs[:, :, :3]
                # vis_aff = attention[0] - np.min(attention[0])
                # vis_aff = 255 * vis_aff / np.max(vis_aff)
                # vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
                # vis_img = np.concatenate((cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB), vis_aff), axis=1)

                gt_state = np.max(attention)

                # cv2.imwrite(f'./test-{gt_state}.jpg', vis_img)

                # print('gt_state: ', gt_state)
                curr_percent = metric[i][1] * 50
                # print("curr: ", curr_percent)
                reward_i = (curr_percent + gt_state) / 2
                reward.append(reward_i)
        return reward

    def train_aff(self, dataset, num_iter, writer):
        for i in range(num_iter):
            obs, act, _, _, not_on_cloth = dataset.sample_index(need_next=False)

            a_len = len(act)

            if self.use_goal_image:
                input_image = np.concatenate((obs, goal), axis=2)
            else:
                input_image = obs.copy()

            p0 = [int((act[0][1] + 1.) * 0.5 * self.input_shape[0]), int((act[0][0] + 1.) * 0.5 * self.input_shape[0])]
            # Do data augmentation (perturb rotation and translation).
            input_image_perturb, p0_list = agent_utils.perturb(input_image.copy(), [p0])
            if input_image_perturb is not None:
                input_image = input_image_perturb
                p0 = p0_list[0]
            else:
                print('no perturb')

            p_list = [p0]
            for i in range(a_len):
                # sample_x = max(min(np.random.normal(loc=p0[0], scale=0.12), self.input_shape[0] - 1), 0)
                # sample_y = max(min(np.random.normal(loc=p0[1], scale=0.12), self.input_shape[0] - 1), 0)
                u = random.randint(0, 319)
                v = random.randint(0, 319)
                p_list.append((u, v))

            critic_map = self.critic_model.forward_with_plist(input_image.copy(), p_list)
            critic_map = critic_map.numpy()

            p_len = len(p_list)
            aff_score = critic_map.max(axis=1).max(axis=1)
            # print("aff_score: ", aff_score)

            with tf.GradientTape() as tape:
                loss = None
                aff_pred = self.attention_model.forward(input_image.copy(), apply_softmax=False)
                for idx_p0 in range(p_len):
                    p0 = p_list[idx_p0]
                    output = aff_pred[:, p0[0], p0[1], :]
                    gt = aff_score[idx_p0]
                    if loss is None:
                        loss = tf.keras.losses.MAE(gt, output)
                    else:
                        loss = loss + tf.keras.losses.MAE(gt, output)
                loss = tf.reduce_mean(loss)
            grad = tape.gradient(loss, self.attention_model.model.trainable_variables)
            self.attention_model.optim.apply_gradients(zip(grad, self.attention_model.model.trainable_variables))
            self.attention_model.metric(loss)
            with writer.as_default():
                tf.summary.scalar('attention_loss', self.attention_model.metric.result(), step=self.total_iter + i)

            print(f'Train Iter: {self.total_iter + i} aff Loss: {loss:.4f}')

        self.total_iter += num_iter
        self.save_aff()

    def train_critic(self, dataset, num_iter, writer, batch=1, extra_dataset=None):
        for i in range(num_iter):
            # for batch training
            input_batch = []
            p0_batch = []
            p1_list_batch = []
            step_batch = []
            reward_batch = []

            flag = 0

            for bh in range(batch):
                curr_obs = None
                if self.step > 1:
                    if bh % 2 == 0:
                        obs, curr_obs, act, metric, step, not_on_cloth = dataset.sample_index(need_next=True)
                    else:
                        obs, act, metric, step, not_on_cloth = dataset.sample_index(need_next=False)
                else:
                    obs, act, metric, step, not_on_cloth = dataset.sample_index(need_next=False)

                step_batch.append(step)

                if self.use_goal_image:
                    input_image = np.concatenate((obs, goal), axis=2)
                else:
                    input_image = obs.copy()

                p0 = [int((act[0][1] + 1.) * 0.5 * self.input_shape[0]), int((act[0][0] + 1.) * 0.5 * self.input_shape[0])]
                p1_list = []
                for point in act:
                    p1 = [int((point[3] + 1.) * 0.5 * self.input_shape[0]), int((point[2] + 1.) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                # pick_area = obs[max(0, p0[0] - 4): min(self.input_shape[0], p0[0] + 4),
                #                 max(0, p0[1] - 4): min(self.input_shape[0], p0[1] + 4),
                #                 :3]
                # if np.sum(pick_area) == 0:
                #     print('not on cloth')
                #     m_len = len(metric)
                #     reward = np.zeros(m_len)
                # else:
                #     reward = self.compute_reward(self.task, metric, self.step)
                # reward_batch.append(reward.copy())

                if not_on_cloth[0] == 1:
                    print('not on cloth')
                    m_len = len(metric)
                    reward = np.zeros(m_len)
                else:
                    reward = self.compute_reward(self.task, metric, self.step, curr_obs)
                reward_batch.append(reward.copy())

                # Do data augmentation (perturb rotation and translation).
                pixels_list = [p0]
                pixels_list.extend(p1_list)
                input_image_perturb, pixels = agent_utils.perturb(input_image.copy(), pixels_list)
                if input_image_perturb is None:
                    flag = 1
                    input_batch.append(input_image.copy())
                else:
                    p0 = pixels[0]
                    p1_list = pixels[1:]
                    input_batch.append(input_image_perturb.copy())

                p0_batch.append(p0)
                p1_list_batch.append(p1_list.copy())

            if flag == 1:
                print("no perturb")

            # Compute Transport training loss.
            if self.out_logits == 1:
                loss1 = self.critic_model.train_batch(input_batch, p0_batch, p1_list_batch, reward_batch, step_batch)
                with writer.as_default():
                    tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
                print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss1:.4f}')
            else:
                if batch == 1:
                    loss1, loss_max_dis, loss_avg_dis, loss_cvx = self.critic_model.train_phy(
                        input_image_batch[0], p0_batch[0], p1_list_batch[0], distance_batch[0], nxt_distances_batch[0],
                        cvx_batch[0], nxt_cvx_batch[0], ep_batch[0], ep_len_batch[0])
                else:
                    loss1, loss_max_dis, loss_avg_dis, loss_cvx = self.critic_model.train_phy_batch(input_batch, p0_batch, p1_list_batch, distance_batch, nxt_distances_batch, cvx_batch, nxt_cvx_batch, ep_batch, ep_len_batch, False, self.task)
                with writer.as_default():
                    tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
                    tf.summary.scalar('max_dis_loss', float(loss_max_dis), step=self.total_iter+i)
                    tf.summary.scalar('avg_dis_loss', float(loss_avg_dis), step=self.total_iter+i)
                    tf.summary.scalar('cvx_loss', float(loss_cvx), step=self.total_iter+i)

                print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss1:.4f} Max_Dis Loss: {loss_max_dis:.4f} Avg_Dis Loss: {loss_avg_dis:.4f} CVX Loss: {loss_cvx:.4f}')

        self.total_iter += num_iter
        self.save_critic()

    def train_critic_multi_gpu(self, dataset, num_iter, writer, batch=1):
        for i in range(num_iter):
            # for batch training
            input_batch = []
            p0_batch = []
            p1_list_batch = []
            step_batch = []
            reward_batch = []

            flag = 0
            curr_obs = None

            for bh in range(batch):
                if self.step > 1:
                    obs, curr_obs, act, metric, step, not_on_cloth = dataset.sample_index(need_next=True)
                else:
                    obs, act, metric, step, not_on_cloth = dataset.sample_index(need_next=False)

                step_batch.append(step)

                if self.use_goal_image:
                    input_image = np.concatenate((obs, goal), axis=2)
                else:
                    input_image = obs.copy()

                p0 = [int((act[0][1] + 1.) * 0.5 * self.input_shape[0]), int((act[0][0] + 1.) * 0.5 * self.input_shape[0])]
                p1_list = []
                for point in act:
                    p1 = [int((point[3] + 1.) * 0.5 * self.input_shape[0]), int((point[2] + 1.) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                # pick_area = obs[max(0, p0[0] - 4): min(self.input_shape[0], p0[0] + 4),
                #                 max(0, p0[1] - 4): min(self.input_shape[0], p0[1] + 4),
                #                 :3]
                # if np.sum(pick_area) == 0:
                #     print('not on cloth')
                #     m_len = len(metric)
                #     reward = np.zeros(m_len)
                # else:
                #     reward = self.compute_reward(self.task, metric, self.step)
                # reward_batch.append(reward.copy())

                if not_on_cloth[0] == 1:
                    print('not on cloth')
                    m_len = len(metric)
                    reward = np.zeros(m_len)
                else:
                    reward = self.compute_reward(self.task, metric, self.step, curr_obs)
                reward_batch.append(reward.copy())

                # Do data augmentation (perturb rotation and translation).
                pixels_list = [p0]
                pixels_list.extend(p1_list)
                input_image_perturb, pixels = agent_utils.perturb(input_image.copy(), pixels_list)
                if input_image_perturb is None:
                    flag = 1
                    input_image = self.preprocess(input_image)
                    input_batch.append(input_image.copy())
                else:
                    p0 = pixels[0]
                    p1_list = pixels[1:]
                    input_image_perturb = self.preprocess(input_image_perturb)
                    input_batch.append(input_image_perturb.copy())

                p0_batch.append(p0)
                p1_list_batch.append(p1_list.copy())

            if flag == 1:
                print("no perturb")

            with self.strategy.scope():
                BUFFER_SIZE = len(input_batch)
                # input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
                # p0_batch = tf.convert_to_tensor(p0_batch, dtype=tf.float32)
                # reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
                # p1_list_batch = tf.convert_to_tensor(p1_list_batch, dtype=tf.float32)
                # step_batch = tf.convert_to_tensor(step_batch, dtype=tf.float32)

                train_dataset = tf.data.Dataset.from_tensor_slices((input_batch, p0_batch,
                                                                    p1_list_batch, reward_batch,
                                                                    step_batch)).shuffle(BUFFER_SIZE).batch(batch)

                train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

                # Compute Transport training loss.
                if self.out_logits == 1:
                    for dist_input in train_dist_dataset:
                        loss1 = self.strategy.run(self.critic_model.train_batch_multi_gpu, args=(dist_input, batch))
                    with writer.as_default():
                        tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
                    print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss1}')
                # else:
                #     if batch == 1:
                #         loss1, loss_max_dis, loss_avg_dis, loss_cvx = self.critic_model.train_phy(
                #             input_image_batch[0], p0_batch[0], p1_list_batch[0], distance_batch[0], nxt_distances_batch[0],
                #             cvx_batch[0], nxt_cvx_batch[0], ep_batch[0], ep_len_batch[0])
                #     else:
                #         loss1, loss_max_dis, loss_avg_dis, loss_cvx = self.critic_model.train_phy_batch(input_batch, p0_batch, p1_list_batch, distance_batch, nxt_distances_batch, cvx_batch, nxt_cvx_batch, ep_batch, ep_len_batch, False, self.task)
                #     with writer.as_default():
                #         tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
                #         tf.summary.scalar('max_dis_loss', float(loss_max_dis), step=self.total_iter+i)
                #         tf.summary.scalar('avg_dis_loss', float(loss_avg_dis), step=self.total_iter+i)
                #         tf.summary.scalar('cvx_loss', float(loss_cvx), step=self.total_iter+i)

                    # print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss1:.4f} Max_Dis Loss: {loss_max_dis:.4f} Avg_Dis Loss: {loss_avg_dis:.4f} CVX Loss: {loss_cvx:.4f}')

        self.total_iter += num_iter
        self.save_critic()

    def act(self, obs, goal=None, p0=None):
        """Run inference and return best action given visual observations.

        If goal-conditioning, provide `goal`. Both `obs` and `goal` have
        'color' and 'depth' keys, but `obs['color']` and `goal['color']` are
        of type list and np.array, respectively. This is different from
        training, above, where both `obs` and `goal` are sampled from the
        dataset class, which will load both as np.arrays. Here, the `goal` is
        still from dataset, but `obs` is from the environment stepping, which
        returns things in a list. Wrap an np.array(...) to get shapes:

        np.array(obs['color']) and goal['color']: (3, 480, 640, 3)
        np.array(obs['depth']) and goal['depth']: (3, 480, 640)
        """
        if self.use_goal_image:
            input_image = np.concatenate((obs, goal), axis=2)

        if self.random_pick:
            indexs = np.transpose(np.nonzero(obs[:, :, 3]))
            index = random.choice(indexs)
            u1 = index[0]
            v1 = index[1]
            print((u1, v1))
            p0_pixel = (u1, v1)

        elif self.expert_pick:
            p0_pixel = p0

        elif self.critic_pick:
            p0_list = []
            # indexs = np.transpose(np.nonzero(obs[:, :, 3]))
            interval = int(self.input_shape[0] / (self.critic_num - 1))
            for i in range(self.critic_num):
                for j in range(self.critic_num):
                    p0_list.append((min(self.input_shape[0] - 1, i * interval), min(self.input_shape[1] - 1, j * interval)))
            # p0_list.append(p0)

            img_pick = obs.copy()
            max_score = -float('inf')
            max_i = 0
            for i in range(0, self.critic_num ** 2):
                # index = random.choice(indexs)
                # u1 = index[0]
                # v1 = index[1]
                u1 = p0_list[i][0]
                v1 = p0_list[i][1]
                p0_pixel = (u1, v1)
                critic_score = np.max(self.critic_model.forward(img_pick.copy(), p0_pixel)[:, :, :, 0])
                if critic_score > max_score:
                    max_score = critic_score
                    max_i = i

            #
            # for i in range(0, self.critic_num):
            #     cable_score = np.max(critic_list[i, :, :, 0])
            #     if cable_score > max_score:
            #         max_i = i
            #         max_score = cable_score

            # if max_i == 0:
            #     print('reverse')

            p0_pixel = (p0_list[max_i][0], p0_list[max_i][1])

        else:
            # Attention model forward pass.
            if self.use_goal_image:
                input_image = obs.copy()
                maxdim = int(input_image.shape[2] / 2)
                input_only = input_image[:, :, :maxdim].copy()
                attention = self.attention_model.forward(input_only)
            else:
                img_aff = obs.copy()
                attention = self.attention_model.forward(img_aff)

            # depth = obs[:, :, -1:]
            # mask = np.where(depth == 0, 0, 1)
            #
            # attention = attention - np.min(attention)
            # attention = attention * mask

            argmax = np.argmax(attention)
            argmax = np.unravel_index(argmax, shape=attention.shape)

            p0_pixel = argmax[1:3]

        img_critic = obs.copy()
        output = self.critic_model.forward(img_critic, p0_pixel)
        critic_score = output[:, :, :, 0]
        argmax = np.argmax(critic_score)
        argmax = np.unravel_index(argmax, shape=critic_score.shape)
        p1_pixel = argmax[1:3]

        # self.visualize_critic(obs.copy(), p0_pixel, p1_pixel, critic_score)

        u1 = (p0_pixel[1]) * 2.0 / self.input_shape[0] - 1
        v1 = (p0_pixel[0]) * 2.0 / self.input_shape[0] - 1
        u2 = (p1_pixel[1]) * 2.0 / self.input_shape[0] - 1
        v2 = (p1_pixel[0]) * 2.0 / self.input_shape[0] - 1
        act = np.array([u1, v1, u2, v2])
        return act


    #-------------------------------------------------------------------------
    # Helper Functions
    #-------------------------------------------------------------------------

    def preprocess(self, image_in):
        image = copy.deepcopy(image_in)
        """Pre-process images (subtract mean, divide by std)."""
        color_mean = 0.18877631
        depth_mean = 0.00509261
        color_std = 0.07276466
        depth_std = 0.00903967
        image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
        image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
        return image

    def load(self, num_iter):
        """Load pre-trained models."""
        attention_fname = 'attention-ckpt-%d.h5' % num_iter
        critic_fname = 'transport-ckpt-%d.h5' % num_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        critic_fname = os.path.join(self.models_dir, critic_fname)
        self.attention_model.load(attention_fname)
        self.critic_model.load(critic_fname)
        self.total_iter = num_iter

    def load_critic_with_epoch(self, total_iter):
        """Load pre-trained models."""
        critic_fname = 'critic-ckpt-%d.h5' % total_iter
        critic_fname = os.path.join(self.models_dir, critic_fname)
        self.critic_model.load(critic_fname)

    def save(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        attention_fname = 'attention-ckpt-%d.h5' % self.total_iter
        critic_fname = 'critic-ckpt-%d.h5' % self.total_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        critic_fname = os.path.join(self.models_dir, critic_fname)
        self.attention_model.save(attention_fname)
        self.critic_model.save(critic_fname)

    def save_critic(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        critic_fname = 'critic-ckpt-%d.h5' % self.total_iter
        critic_fname = os.path.join(self.models_dir, critic_fname)
        self.critic_model.save(critic_fname)

    # def save_critic_with_epoch(self, remove=True):
    #     """Save models."""
    #     if not os.path.exists(self.models_dir):
    #         os.makedirs(self.models_dir)
    #     critic_fname = 'critic-ckpt-%d.h5' % self.total_iter
    #     critic_fname = os.path.join(self.models_dir, critic_fname)
    #     if remove and self.prev_critic_fname is not None:
    #         cmd = "rm %s" % self.prev_critic_fname
    #         call(cmd, shell=True)
    #     self.prev_critic_fname = critic_fname
    #     self.critic_model.save(critic_fname)
    #
    # def save_aff_with_epoch(self, remove=True):
    #     """Save models."""
    #     if not os.path.exists(self.models_dir):
    #         os.makedirs(self.models_dir)
    #     aff_fname = 'attention-ckpt-%d.h5' % self.total_iter
    #     aff_fname = os.path.join(self.models_dir, aff_fname)
    #     if remove and self.prev_aff_fname is not None:
    #         cmd = "rm %s" % self.prev_aff_fname
    #         call(cmd, shell=True)
    #     self.prev_aff_fname = aff_fname
    #     self.attention_model.save(aff_fname)

    def save_aff(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        aff_fname = 'attention-ckpt-%d.h5' % self.total_iter
        aff_fname = os.path.join(self.models_dir, aff_fname)
        self.attention_model.save(aff_fname)


class OriginalTransporterAffCriticAgent(AffCritic):
    """
    The official Transporter agent tested in the paper. Added num_rotations and
    crop_bef_q as arguments. Default arguments are 24 (though this was later
    turned to 36 for Andy's paper) and to crop to get kernels _before_ the query.
    """

    def __init__(self, name, task, use_goal_image=0, load_critic_dir='xxx', load_aff_dir='xxx', load_next_dir='xxx',
                 out_logits=1, without_global=False, critic_pick=False, random_pick=False, expert_pick=False, step=1,
                 learning_rate=1e-4, strategy=None):
        super().__init__(name, task, use_goal_image=use_goal_image, out_logits=out_logits,
                         critic_pick=critic_pick, random_pick=random_pick, expert_pick=expert_pick, step=step,
                         strategy=strategy)

        self.attention_model = Affordance(input_shape=self.input_shape,
                                          preprocess=self.preprocess,
                                          learning_rate=learning_rate,
                                          strategy=strategy
                                          )

        self.critic_model = Critic_MLP(input_shape=self.input_shape,
                                       preprocess=self.preprocess,
                                       out_logits=self.out_logits,
                                       learning_rate=learning_rate,
                                       without_global=without_global,
                                       strategy=strategy
                                       )

        if load_next_dir != 'xxx':
            self.next_model = Affordance(input_shape=self.input_shape,
                                         preprocess=self.preprocess,
                                         learning_rate=0,
                                         )

            print('*' * 50)
            print('*' * 20 + 'load next model' + '*' * 20)
            print('*' * 3 + f'load_next_dir {load_next_dir}' + '*' * 3)
            self.next_model.load(load_next_dir)
            print('*' * 50)

        if load_aff_dir != 'xxx':
            print('*' * 50)
            print('*' * 20 + 'load aff model' + '*' * 20)
            print('*' * 3 + f'load_aff_dir {load_aff_dir}' + '*' * 3)
            self.attention_model.load(load_aff_dir)
            print('*' * 50)

        if load_critic_dir != 'xxx':
            print('*' * 50)
            print('*' * 20 + 'load critic model' + '*' * 20)
            print('*' * 3 + f'load_critic_dir {load_critic_dir}' + '*' * 3)
            self.critic_model.load(load_critic_dir)
            print('*' * 50)
            print(f'*' * 20 + f'critic_pick {self.critic_pick}'+ '*' * 20)


class GoalTransporterAgent(AffCritic):
    """
    Goal-conditioned Transporter agent where we pass the goal image through another FCN,
    and then combine the resulting features with the pick and placing networks for better
    goal-conditioning. This uses our new `TransportGoal` architecture. We don't stack the
    input and target images, so we can directly use `self.input_shape` for both modules.
    """

    def __init__(self, name, task, num_rotations=24):
        # (Oct 26) set attn_no_targ=False, and that should be all we need along w/shape ...
        super().__init__(name, task, use_goal_image=True, attn_no_targ=False)

        # (Oct 26) Stack the goal image for the Attention module -- model cannot pick properly otherwise.
        a_shape = (self.input_shape[0], self.input_shape[1], int(self.input_shape[2] * 2))

        self.attention_model = Attention(input_shape=a_shape,
                                         num_rotations=1,
                                         preprocess=self.preprocess)
        # self.transport_model = TransportGoal(input_shape=self.input_shape,
        #                                      num_rotations=self.num_rotations,
        #                                      crop_size=self.crop_size,
        #                                      preprocess=self.preprocess)


