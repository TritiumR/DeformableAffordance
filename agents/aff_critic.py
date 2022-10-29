#!/usr/bin/env python
import os
import pickle

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
    def __init__(self, name, task, image_size=320, critic_pick=False, random_pick=False, expert_pick=False, use_goal_image=False,
                 out_logits=1, step=1, only_depth=False, strategy=None):
        """Creates Transporter agent with attention and transport modules."""
        self.name = name
        self.task = task
        self.image_size = image_size
        self.step = step
        self.total_iter = 0
        if use_goal_image:
            self.input_shape = (image_size, image_size, 8)
        else:
            self.input_shape = (image_size, image_size, 4)
        self.models_dir = os.path.join('checkpoints', self.name + f'-step-{self.step}')

        self.expert_pick = expert_pick
        self.critic_pick = critic_pick
        self.critic_num = 33
        self.random_pick = random_pick

        self.out_logits = out_logits
        self.use_goal_image = use_goal_image
        self.only_depth = only_depth
        self.strategy = strategy

        self.reward_cache = dict()

    def compute_reward(self, metric, not_on_cloth, curr_obs=None, only_state=False, only_gt=False, iepisode=None, obs=None):
        m_len = len(metric)
        reward = []

        if not_on_cloth == 1:
            print('not on cloth')
            m_len = len(metric)
            if self.task == 'cloth-flatten':
                reward = np.zeros(m_len)
            elif self.task == 'rope-configuration':
                reward = np.ones(m_len) * 50
            return reward

        if curr_obs is None:
            if self.task == 'cloth-flatten':
                for i in range(0, m_len):
                    curr_percent = metric[i][1] * 50
                    reward.append(curr_percent)
            elif self.task == 'rope-configuration':
                for i in range(0, m_len):
                    curr_distance = metric[i][1] * 100
                    reward.append(curr_distance)
        else:
            # l_img = np.concatenate((
            #     cv2.cvtColor(obs, cv2.COLOR_BGR2RGB),
            #     cv2.cvtColor(curr_obs[0], cv2.COLOR_BGR2RGB),
            #     cv2.cvtColor(curr_obs[1], cv2.COLOR_BGR2RGB)),
            #     axis=1)
            # r_img = np.concatenate((
            #     cv2.cvtColor(curr_obs[2], cv2.COLOR_BGR2RGB),
            #     cv2.cvtColor(curr_obs[3], cv2.COLOR_BGR2RGB),
            #     cv2.cvtColor(curr_obs[4], cv2.COLOR_BGR2RGB)),
            #     axis=1)
            # img = np.concatenate((l_img, r_img), axis=0)
            #
            # score_list = ''
            if only_gt:
                for i in range(0, m_len):
                    curr_percent = metric[i][1] * 50
                    reward.append(curr_percent)
                # print(reward)
            else:
                if iepisode in self.reward_cache:
                    return self.reward_cache[iepisode]
                if self.only_depth:
                    attention = self.attention_model.forward_batch(np.array(curr_obs)[:, :, :, -1:].copy())
                else:
                    attention = self.attention_model.forward_batch(curr_obs.copy())
                for i in range(0, m_len):
                    if self.task == 'cloth-flatten':
                        gt_state = np.max(attention[i])
                        curr_percent = metric[i][1] * 50
                    elif self.task == 'rope-configuration':
                        gt_state = np.min(attention[i])
                        curr_percent = metric[i][1] * 100
                    # if (i < 5):
                    #     score_list += f'-{int(gt_state * 2)}'

                    # img_obs = curr_obs[i][:, :, :3]
                    # vis_aff = attention[i] - np.min(attention[i])
                    # if self.task == 'rope-configuration':
                    #     vis_aff = -vis_aff
                    # vis_aff = np.exp(vis_aff) / np.sum(np.exp(vis_aff))
                    # vis_aff = 255 * vis_aff / np.max(vis_aff)
                    # vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
                    # vis_img = np.concatenate((cv2.cvtColor(curr_obs[i], cv2.COLOR_BGR2RGB), vis_aff), axis=1)
                    # cv2.imwrite(f'./visual/rope-no-debug-{iepisode}-{i}-{int(gt_state * 100)}-vis.jpg', vis_img)
                    # print(f'saved ./visual/rope-no-debug-{iepisode}-{i}-{int(gt_state * 100)}-vis.jpg')

                    # print('gt_state: ', gt_state)
                    # print("curr: ", curr_percent)
                    if only_state:
                        reward_i = gt_state
                    else:
                        reward_i = (curr_percent * 2 + gt_state) / 3
                    reward.append(reward_i)

                self.reward_cache[iepisode] = reward
                # cv2.imwrite(f'./visual/state-{score_list}.jpg', img)
                # print("save img")
        return reward

    def train_aff(self, dataset, num_iter, writer, batch, no_perturb=False):
        for i in range(num_iter):
            obs, act, _, _, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

            if self.use_goal_image:
                input_image = np.concatenate((obs, goal), axis=2)
            else:
                input_image = obs.copy()

            p0 = [min(self.input_shape[0] - 1, int((act[0][1] + 1.) * 0.5 * self.input_shape[0])),
                  min(self.input_shape[0] - 1, int((act[0][0] + 1.) * 0.5 * self.input_shape[0]))]
            # Do data augmentation (perturb rotation and translation).
            if not no_perturb:
                input_image_perturb, p0_list = agent_utils.perturb(input_image.copy(), [p0])
                if input_image_perturb is not None:
                    input_image = input_image_perturb
                    p0 = p0_list[0]
                else:
                    print('no perturb')

            p_list = [p0]
            for p_i in range(batch):
                # sample_x = max(min(np.random.normal(loc=p0[0], scale=0.12), self.input_shape[0] - 1), 0)
                # sample_y = max(min(np.random.normal(loc=p0[1], scale=0.12), self.input_shape[0] - 1), 0)
                u = random.randint(0, self.image_size - 1)
                v = random.randint(0, self.image_size - 1)
                p_list.append((u, v))

            critic_map = self.critic_model.forward_with_plist(input_image.copy(), p_list)
            critic_map = critic_map.numpy()

            p_len = len(p_list)
            if self.task == 'cloth-flatten':
                aff_score = critic_map.max(axis=1).max(axis=1)
            elif self.task == 'rope-configuration':
                aff_score = critic_map.min(axis=1).min(axis=1)
            # print("aff_score: ", aff_score)

            with tf.GradientTape() as tape:
                loss = None
                if self.only_depth:
                    aff_pred = self.attention_model.forward(input_image[:, :, -1:].copy(), apply_softmax=False)
                else:
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
                loss = loss / p_len
            grad = tape.gradient(loss, self.attention_model.model.trainable_variables)
            self.attention_model.optim.apply_gradients(zip(grad, self.attention_model.model.trainable_variables))
            self.attention_model.metric(loss)
            with writer.as_default():
                tf.summary.scalar('attention_loss', self.attention_model.metric.result(), step=self.total_iter + i)

            print(f'Train Iter: {self.total_iter + i} aff Loss: {loss:.4f}')

        self.total_iter += num_iter
        self.save_aff()

    def train_critic(self, dataset, num_iter, writer, batch=1, extra_dataset=None, no_perturb=False, only_state=False, only_gt=False):
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
                    if extra_dataset is not None:
                        if bh % 2 == 0:
                            obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                        else:
                            obs, act, metric, step, not_on_cloth, iepisode = extra_dataset.sample_index(need_next=False)
                    else:
                        obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                else:
                    obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

                # self.visualize_aff_critic(obs.copy(), id=iepisode, curr_obs=np.array(curr_obs).copy(), action=act)

                step_batch.append(step)

                if self.use_goal_image:
                    input_image = np.concatenate((obs, goal), axis=2)
                else:
                    input_image = obs.copy()

                p0 = [min(self.input_shape[0] - 1, int((act[0][1] + 1.) * 0.5 * self.input_shape[0])),
                      min(self.input_shape[0] - 1, int((act[0][0] + 1.) * 0.5 * self.input_shape[0]))]

                p1_list = []
                for point in act:
                    p1 = [int((point[3] + 1.) * 0.5 * self.input_shape[0]), int((point[2] + 1.) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                reward = self.compute_reward(metric, not_on_cloth[0], curr_obs, only_state, only_gt, iepisode, obs=obs.copy())
                # print('reward: ', reward)
                reward_batch.append(reward.copy())

                if no_perturb:
                    input_batch.append(input_image.copy())
                    p0_batch.append(p0)
                    p1_list_batch.append(p1_list.copy())
                else:
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

            if flag == 1 and not no_perturb:
                print("no perturb")

            # Compute Transport training loss.
            loss = self.critic_model.train_batch(input_batch, p0_batch, p1_list_batch, reward_batch, step_batch, validate=False)
            with writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
            print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss:.4f}')

        self.total_iter += num_iter
        self.save_critic()

    def train_critic_multi_gpu(self, dataset, num_iter, writer, batch=1):
        # for epoch training
        input_batch = []
        p0_batch = []
        p1_list_batch = []
        step_batch = []
        reward_batch = []

        flag = 0

        for bh in range(200):
            curr_obs = None
            if self.step > 1:
                if extra_dataset is not None:
                    if bh % 2 == 0:
                        obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                    else:
                        obs, act, metric, step, not_on_cloth, iepisode = extra_dataset.sample_index(need_next=False)
                else:
                    obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
            else:
                obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

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
            print("BUFFER_SIZE", BUFFER_SIZE)
            print("batch", batch)
            # input_batch = tf.convert_to_tensor(input_batch, dtype=tf.float32)
            # p0_batch = tf.convert_to_tensor(p0_batch, dtype=tf.float32)
            # reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
            # p1_list_batch = tf.convert_to_tensor(p1_list_batch, dtype=tf.float32)
            # step_batch = tf.convert_to_tensor(step_batch, dtype=tf.float32)

            train_dataset = tf.data.Dataset.from_tensor_slices((input_batch, p0_batch,
                                                                p1_list_batch, reward_batch,
                                                                step_batch)).shuffle(BUFFER_SIZE).batch(batch)

            train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

            for i in range(num_iter):
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

    def validate(self, dataset, extra_dataset, writer, visualize, iter=0):
        for i in range(dataset.validate):
            for bh in range(1):
                curr_obs = None
                if self.step > 1:
                    if extra_dataset is not None:
                        if bh % 2 == 0:
                            obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                        else:
                            obs, act, metric, step, not_on_cloth, iepisode = extra_dataset.sample_index(need_next=False)
                    else:
                        obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                else:
                    obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

                step_batch.append(step)

                if self.use_goal_image:
                    input_image = np.concatenate((obs, goal), axis=2)
                else:
                    input_image = obs.copy()

                p0 = [int((act[0][1] + 1.) * 0.5 * self.input_shape[0]),
                      int((act[0][0] + 1.) * 0.5 * self.input_shape[0])]
                p1_list = []
                for point in act:
                    p1 = [int((point[3] + 1.) * 0.5 * self.input_shape[0]),
                          int((point[2] + 1.) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                reward = self.compute_reward(metric, not_on_cloth[0], curr_obs)
                # print('reward: ', reward)
                reward_batch.append(reward.copy())

                if no_perturb:
                    input_batch.append(input_image.copy())
                    p0_batch.append(p0)
                    p1_list_batch.append(p1_list.copy())
                else:
                    # Do data augmentation (perturb rotation and translation).
                    pixels_list = [p0]
                    pixels_list.extend(p1_list)
                    input_image_perturb, pixels = agent_utils.perturb(input_image.copy(), pixels_list)
                    if input_image_perturb is None:
                        input_batch.append(input_image.copy())
                    else:
                        p0 = pixels[0]
                        p1_list = pixels[1:]
                        input_batch.append(input_image_perturb.copy())

                    p0_batch.append(p0)
                    p1_list_batch.append(p1_list.copy())

            # Compute Transport training loss.
            loss1 = self.critic_model.train_batch(input_batch, p0_batch, p1_list_batch, reward_batch, step_batch, validate=True)
            with writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_model.validate_metric.result(), step=iter + i)
            print(f'Validate Iter: {iter + i} Loss: {loss1:.4f}')

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
            min_score = float('inf')
            max_i = 0

            if self.task == 'cloth-flatten':
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
            elif self.task == 'rope-configuration':
                for i in range(0, self.critic_num ** 2):
                    u1 = p0_list[i][0]
                    v1 = p0_list[i][1]
                    p0_pixel = (u1, v1)
                    critic_score = np.min(self.critic_model.forward(img_pick.copy(), p0_pixel)[:, :, :, 0])
                    if critic_score < min_score:
                        min_score = critic_score
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
                if self.only_depth:
                    img_aff = obs[:, :, -1:].copy()
                else:
                    img_aff = obs.copy()
                attention = self.attention_model.forward(img_aff)
                # print("state: ", np.max(attention))

            # depth = obs[:, :, -1:]
            # mask = np.where(depth == 0, 0, 1)
            #
            # attention = attention - np.min(attention)
            # attention = attention * mask
            if self.task == 'cloth-flatten':
                argmax = np.argmax(attention)
            elif self.task == 'rope-configuration':
                argmax = np.argmin(attention)

            argmax = np.unravel_index(argmax, shape=attention.shape)

            p0_pixel = argmax[1:3]

        img_critic = obs.copy()
        output = self.critic_model.forward(img_critic, p0_pixel)
        critic_score = output[:, :, :, 0]
        if self.task == 'cloth-flatten':
            argmax = np.argmax(critic_score)
        elif self.task == 'rope-configuration':
            argmax = np.argmin(critic_score)
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

    def get_mean_and_std(self, path, mode):
        print("computing mean and std...")
        dirs = os.listdir(path)

        mean = np.zeros(self.input_shape[2])
        std = np.zeros(self.input_shape[2])

        for fname in dirs:
            data = pickle.load(open(os.path.join(path, fname), 'rb'))
            obs = np.array(data['obs'])
            for d in range(self.input_shape[2]):
                mean[d] += obs[:, :, d].mean() / 255
                std[d] += obs[:, :, d].std() / 255

        mean /= len(dirs)
        std /= len(dirs)

        print("mean: ", mean)
        print("std: ", std)

        if mode == 'critic':
            self.critic_mean = mean
            self.critic_std = std
        elif mode == 'aff':
            self.aff_mean = mean
            self.aff_std = std

        self.save_mean_std(mode)

    def critic_preprocess(self, image_in):
        image = copy.deepcopy(image_in)
        """Pre-process images (subtract mean, divide by std)."""
        for d in range(self.input_shape[2]):
            image[:, :, d] = (image[:, :, d] / 255 - self.critic_mean[d]) / self.critic_std[d]
        return image

    def aff_preprocess(self, image_in):
        image = copy.deepcopy(image_in)
        """Pre-process images (subtract mean, divide by std)."""
        for d in range(self.input_shape[2]):
            image[:, :, d] = (image[:, :, d] / 255 - self.aff_mean[d]) / self.aff_std[d]
        return image

    def ori_preprocess(self, image_in):
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

    def save_mean_std(self, mode):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        data = {}
        if mode == 'critic':
            data['mean'] = self.critic_mean
            data['std'] = self.critic_std
        elif mode == 'aff':
            data['mean'] = self.aff_mean
            data['std'] = self.aff_std
        fname = mode + 'mean_std.pkl'
        pickle.dump(data, open(os.path.join(self.models_dir, fname), 'wb'))

    def save_critic(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        critic_fname = 'critic-ckpt-%d.h5' % self.total_iter
        critic_fname = os.path.join(self.models_dir, critic_fname)
        self.critic_model.save(critic_fname)

    def save_critic_with_epoch(self, iter):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        critic_fname = 'critic-online-ckpt-%d.h5' % iter
        critic_fname = os.path.join(self.models_dir, critic_fname)
        print(f'save to {critic_fname}')
        self.critic_model.save(critic_fname)

    def save_aff_with_epoch(self, iter):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        aff_fname = 'attention-online-ckpt-%d.h5' % iter
        aff_fname = os.path.join(self.models_dir, aff_fname)
        print(f'save to {aff_fname}')
        self.attention_model.save(aff_fname)

    def save_aff(self):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        aff_fname = 'attention-ckpt-%d.h5' % self.total_iter
        aff_fname = os.path.join(self.models_dir, aff_fname)
        self.attention_model.save(aff_fname)

    def visualize_aff_critic(self, obs, id, curr_obs, action):
        img_aff = obs.copy()
        attention = self.attention_model.forward(img_aff)

        if self.task == 'cloth-flatten':
            state_score = int(np.max(attention) * 2)

        elif self.task == 'rope-configuration':
            state_score = int(np.min(attention) * 10)
            attention = -attention

        attention = attention - np.min(attention)
        argmax = np.argmax(attention)
        argmax = np.unravel_index(argmax, shape=attention.shape)

        p0_pixel = argmax[1:3]

        img_critic = obs.copy()
        critic = self.critic_model.forward(img_critic, p0_pixel)

        if self.task == 'cloth-flatten':
            argmax = np.argmax(critic)
            argmax = np.unravel_index(argmax, shape=critic.shape)

        elif self.task == 'rope-configuration':
            argmax = np.argmin(critic)
            argmax = np.unravel_index(argmax, shape=critic.shape)

        p1_pixel_max = argmax[1:3]

        vis_aff = np.array(attention[0])
        vis_critic = np.array(critic[0])

        if self.task == 'rope-configuration':
            vis_critic = -vis_critic
            vis_aff = np.exp(vis_aff) / np.sum(np.exp(vis_aff))
            vis_critic = np.exp(vis_critic) / np.sum(np.exp(vis_critic))

        vis_aff = vis_aff - np.min(vis_aff)
        vis_aff = 255 * vis_aff / np.max(vis_aff)
        vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

        vis_critic = vis_critic - np.min(vis_critic)
        vis_critic = 255 * vis_critic / np.max(vis_critic)
        vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

        curr_obs = curr_obs[:, :, :, :-1]
        for i in range(9):
            p1 = (action[i][2], action[i][3])
            p1_pixel = (int((p1[1] + 1.) / 2. * self.input_shape[0]), int((p1[0] + 1.) / 2. * self.input_shape[0]))
            for u in range(max(0, p1_pixel[0] - 2), min(self.input_shape[0], p1_pixel[0] + 2)):
                for v in range(max(0, p1_pixel[1] - 2), min(self.input_shape[0], p1_pixel[1] + 2)):
                    vis_critic[u][v] = (255, 255, 255)
                    curr_obs[i][u][v] = (255, 255, 255)

        img_obs = obs[:, :, :3]

        for u in range(max(0, p0_pixel[0] - 2), min(self.input_shape[0], p0_pixel[0] + 2)):
            for v in range(max(0, p0_pixel[1] - 2), min(self.input_shape[0], p0_pixel[1] + 2)):
                img_obs[u][v] = (255, 0, 0)

        for u in range(max(0, p1_pixel_max[0] - 2), min(self.input_shape[0], p1_pixel_max[0] + 2)):
            for v in range(max(0, p1_pixel_max[1] - 2), min(self.input_shape[0], p1_pixel_max[1] + 2)):
                img_obs[u][v] = (0, 255, 0)
                vis_critic[u][v] = (0, 255, 0)

        vis_img = np.concatenate((cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB), vis_aff, vis_critic), axis=1)

        l_img = np.concatenate((
            cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[0], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[1], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[2], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[3], cv2.COLOR_BGR2RGB)),
            axis=1)
        r_img = np.concatenate((
            cv2.cvtColor(curr_obs[4], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[5], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[6], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[7], cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs[8], cv2.COLOR_BGR2RGB)),
            axis=1)
        img = np.concatenate((l_img, r_img), axis=0)

        cv2.imwrite(f'./visual/{self.name}-compare-obs-aff_critic-{id}-{state_score}.jpg', vis_img)
        cv2.imwrite(f'./visual/{self.name}-compare-curr-aff_critic-{id}.jpg', img)
        print("save to" + f'./visual/{self.name}-compare-aff_critic-{id}.jpg')


class OriginalTransporterAffCriticAgent(AffCritic):
    """
    The official Transporter agent tested in the paper. Added num_rotations and
    crop_bef_q as arguments. Default arguments are 24 (though this was later
    turned to 36 for Andy's paper) and to crop to get kernels _before_ the query.
    """

    def __init__(self, name, task, image_size=320, use_goal_image=0, load_critic_dir='xxx', load_aff_dir='xxx', load_next_dir='xxx',
                 out_logits=1, without_global=False, critic_pick=False, random_pick=False, expert_pick=False, step=1,
                 learning_rate=1e-4, critic_depth=1, batch_normalize=False, layer_normalize=False, only_depth=False, strategy=None):
        super().__init__(name, task, image_size=image_size, use_goal_image=use_goal_image, out_logits=out_logits,
                         critic_pick=critic_pick, random_pick=random_pick, expert_pick=expert_pick, step=step,
                         only_depth=only_depth, strategy=strategy)

        fname = 'criticmean_std.pkl'
        load_critic_dir_list = load_critic_dir.split('/')[:-1]
        load_critic_mean_std_dir = load_critic_dir_list[0]
        for critic_dir_part in load_critic_dir_list[1:]:
            load_critic_mean_std_dir += '/' + critic_dir_part
        print(load_critic_mean_std_dir)
        mean_std = pickle.load(open(os.path.join(load_critic_mean_std_dir, fname), 'rb'))
        self.critic_mean = mean_std['mean']
        self.critic_std = mean_std['std']
        print("critic mean: ", self.critic_mean)
        print('critic std: ', self.critic_std)

        fname = 'affmean_std.pkl'
        load_aff_dir_list = load_aff_dir.split('/')[:-1]
        load_aff_mean_std_dir = load_aff_dir_list[0]
        for aff_dir_part in load_aff_dir_list[1:]:
            load_aff_mean_std_dir += '/' + aff_dir_part
        print(load_aff_mean_std_dir)
        mean_std = pickle.load(open(os.path.join(load_aff_mean_std_dir, fname), 'rb'))
        self.aff_mean = mean_std['mean']
        self.aff_std = mean_std['std']
        print("aff mean: ", self.aff_mean)
        print('aff std: ', self.aff_std)

        if only_depth:
            self.attention_model = Affordance(input_shape=(image_size, image_size, 1),
                                              preprocess=self.aff_preprocess,
                                              learning_rate=learning_rate,
                                              strategy=strategy
                                              )
        else:
            self.attention_model = Affordance(input_shape=self.input_shape,
                                              preprocess=self.aff_preprocess,
                                              learning_rate=learning_rate,
                                              strategy=strategy
                                              )

        self.critic_model = Critic_MLP(input_shape=self.input_shape,
                                       preprocess=self.critic_preprocess,
                                       out_logits=self.out_logits,
                                       learning_rate=learning_rate,
                                       without_global=without_global,
                                       depth=critic_depth,
                                       batch_normalize=batch_normalize,
                                       layer_normalize=layer_normalize,
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
            print(f'*' * 20 + f'critic_pick {self.critic_pick}' + '*' * 20)


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


