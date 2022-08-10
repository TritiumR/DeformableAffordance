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

from scipy.spatial import ConvexHull

import tensorflow as tf

import ipdb


class AffCritic:
    def __init__(self, name, task, critic_pick=False, random_pick=False, use_goal_image=False,
                 out_logits=1):
        """Creates Transporter agent with attention and transport modules."""
        self.name = name
        self.task = task
        self.total_iter = 0
        if use_goal_image:
            self.input_shape = (320, 320, 8)
        else:
            self.input_shape = (320, 320, 4)
        self.models_dir = os.path.join('checkpoints', self.name)

        self.critic_pick = critic_pick
        self.random_pick = random_pick

        self.out_logits = out_logits
        self.use_goal_image = use_goal_image

    def compute_reward(self, task, metric, step):
        m_len = len(metric)
        reward = []

        if task == 'cloth-flatten':
            if step == 1:
                for i in range(0, m_len):
                    full_area, prev_area, curr_area = metric[i]
                    reward.append((curr_area - prev_area) / full_area)
            else:
                for i in range(0, m_len):
                    full_area, prev_area, curr_area = metric[i]
                    reward.append((curr_area - prev_area) / full_area)
            return reward

    def train_OS(self, train_data, writer):
        if self.use_goal_image:
            obs, goal, act, metric, step = train_data
        else:
            obs, act, metric, step = train_data

        p0 = act[0][:2]
        p1_list = act[:, 2:]
        # Do data augmentation (perturb rotation and translation).
        pixels_list = [p0]
        pixels_list.extend(p1_list)
        input_image, pixels = agent_utils.perturb(obs, pixels_list)
        p0 = pixels[0]
        p1_list = pixels[1:]

        reward = self.compute_reward(self.task, metric, step)

        loss1 = self.critic_model.train(obs, p0, p1_list, reward, False, self.task)
        with writer.as_default():
            tf.summary.scalar('critic_loss', self.critic_model.metric.result(),
                step=self.total_iter)
            tf.summary.scalar('success_rate', float(self.success_rate),
                              step=self.total_iter)
        print(f'Train Iter: {self.total_iter} Critic Loss: {loss1:.4f}')

        self.total_iter += 1

    def train_aff_OS(self, train_data, writer):
        if self.use_goal_image:
            obs, goal, act, metric, step = train_data
        else:
            obs, act, metric, step = train_data

        p0 = act[0][:2]

        # Do data augmentation (perturb rotation and translation).
        pixels_list = [p0]
        input_image, pixels = utils.perturb(obs, pixels_list)
        p_list = pixels
        len_p = len(p_list)

        img_critic = input_image.copy()
        critic_map = self.critic_model.forward_aff_batch_gt(img_critic, p_list)
        critic_map = critic_map.numpy()
        circle_range = self.crop_circle
        for i in range(critic_map.shape[0]):
            place_mask = np.zeros((320, 160))
            for u in range(max(0, p_list[i][0] - circle_range), min(320, p_list[i][0] + circle_range)):
                v_range = int(pow((circle_range ** 2 - (p_list[i][0] - u) ** 2), 0.5))
                for v in range(max(0, p_list[i][1] - v_range), min(160, p_list[i][1] + v_range)):
                    place_mask[u][v] = 1
            critic_map[i] = critic_map[i] - np.min(critic_map[i])
            critic_map[i] = critic_map[i] * place_mask
        critic_max = critic_map.max(axis=1).max(axis=1)

        with tf.GradientTape() as tape:
            loss = None
            aff_pred = self.attention_model.forward(input_image.copy(), apply_softmax=False)
            for idx_p0 in range(len_p):
                p0 = p_list[idx_p0]
                output = aff_pred[:, p0[0], p0[1], :]
                gt = critic_max[idx_p0]
                if loss is None:
                    loss = tf.keras.losses.MAE(gt, output)
                else:
                    loss = loss + tf.keras.losses.MAE(gt, output)
                # loss = tf.nn.softmax_cross_entropy_with_logits(label, output)
            loss = tf.reduce_mean(loss)
        grad = tape.gradient(loss, self.attention_model.model.trainable_variables)
        self.attention_model.optim.apply_gradients(
            zip(grad, self.attention_model.model.trainable_variables))
        self.attention_model.metric(loss)
        with writer.as_default():
            tf.summary.scalar('attention_loss', self.attention_model.metric.result(), step=self.total_iter)

        print(f'Train Iter: {self.total_iter} aff Loss: {loss:.4f}')
        self.total_iter += 1

    def train(self, dataset, num_iter, writer, batch=1):
        for i in range(num_iter):
            # for batch training
            input_batch = []
            p0_batch = []
            p1_list_batch = []
            step_batch = []
            reward_batch = []

            flag = 0

            for bh in range(batch):
                obs, act, metric, step = dataset.sample_index(last_index=100, task=self.task)
                # print(obs.shape)

                reward = self.compute_reward(self.task, metric, step)
                reward_batch.append(reward.copy())
                step_batch.append(step)

                if self.use_goal_image:
                    input_image = np.concatenate((obs, goal), axis=2)
                else:
                    input_image = obs.copy()

                p0 = [int((act[0][0] + 1) * 0.5 * self.input_shape[0]), int((act[0][1] + 1) * 0.5 * self.input_shape[0])]
                p1_list = []
                for point in act:
                    p1 = [int((point[0] + 1) * 0.5 * self.input_shape[0]), int((point[1] + 1) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                # Do data augmentation (perturb rotation and translation).
                pixels_list = [p0]
                pixels_list.extend(p1_list)
                input_image, pixels = agent_utils.perturb(input_image, pixels_list)
                if input_image is None:
                    flag = 1
                    break
                p0 = pixels[0]
                p1_list = pixels[1:]

                input_batch.append(input_image.copy())

                p0_batch.append(p0)
                p1_list_batch.append(p1_list.copy())

            if flag == 1:
                print("empty iter")
                continue

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
                    tf.summary.scalar('critic_loss', self.critic_model.metric.result(),
                        step=self.total_iter+i)
                    tf.summary.scalar('max_dis_loss', float(loss_max_dis),
                        step=self.total_iter+i)
                    tf.summary.scalar('avg_dis_loss', float(loss_avg_dis),
                        step=self.total_iter+i)
                    tf.summary.scalar('cvx_loss', float(loss_cvx),
                        step=self.total_iter+i)

                print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss1:.4f} Max_Dis Loss: {loss_max_dis:.4f} Avg_Dis Loss: {loss_avg_dis:.4f} CVX Loss: {loss_cvx:.4f}')

        self.total_iter += num_iter
        self.save_critic()

    # def visualize_logt3(self, input_img, pick_pos, convex_map, avg_dis_map, max_dis_map, id=-1):
    #     pick_obs = input_img.copy()
    #     p0 = pick_pos
    #
    #     for u in range(p0[0] - 2, p0[0] + 2):
    #         for v in range(p0[1] - 2, p0[1] + 2):
    #             pick_obs[u][v] = (255, 0, 0)
    #
    #     convex_map = convex_map.numpy().reshape(320, 160, 1)
    #     avg_dis_map = avg_dis_map.numpy().reshape(320, 160, 1)
    #     max_dis_map = max_dis_map.numpy().reshape(320, 160, 1)
    #
    #     vis_convex = np.float32(convex_map)
    #     convex_index = np.argmax(vis_convex)
    #     convex_pos = (convex_index // vis_convex.shape[1], convex_index % vis_convex.shape[1])
    #     vis_convex = vis_convex - np.min(vis_convex)
    #     vis_convex = 255 * vis_convex / np.max(vis_convex)
    #     vis_convex = cv2.applyColorMap(np.uint8(vis_convex), cv2.COLORMAP_JET)
    #
    #     for u in range(convex_pos[0] - 2, convex_pos[0] + 2):
    #         for v in range(convex_pos[1] - 2, convex_pos[1] + 2):
    #             pick_obs[u][v] = (255, 255, 255)
    #             vis_convex[u][v] = (255, 255, 255)
    #
    #     vis_avg_dis = np.float32(avg_dis_map)
    #     avg_dis_index = np.argmin(vis_avg_dis)
    #     avg_dis_pos = (avg_dis_index // vis_avg_dis.shape[1], avg_dis_index % vis_avg_dis.shape[1])
    #     vis_avg_dis = vis_avg_dis - np.min(vis_avg_dis)
    #     vis_avg_dis = 255 * vis_avg_dis / np.max(vis_avg_dis)
    #     vis_avg_dis = cv2.applyColorMap(np.uint8(vis_avg_dis), cv2.COLORMAP_JET)
    #
    #     for u in range(avg_dis_pos[0] - 2, avg_dis_pos[0] + 2):
    #         for v in range(avg_dis_pos[1] - 2, avg_dis_pos[1] + 2):
    #             pick_obs[u][v] = (255, 255, 255)
    #             vis_avg_dis[u][v] = (255, 255, 255)
    #
    #     vis_max_dis = np.float32(max_dis_map)
    #     max_dis_index = np.argmin(vis_max_dis)
    #     max_dis_pos = (max_dis_index // vis_max_dis.shape[1], max_dis_index % vis_max_dis.shape[1])
    #     vis_max_dis = vis_max_dis - np.min(vis_max_dis)
    #     vis_max_dis = 255 * vis_max_dis / np.max(vis_max_dis)
    #     vis_max_dis = cv2.applyColorMap(np.uint8(vis_max_dis), cv2.COLORMAP_JET)
    #
    #     for u in range(max_dis_pos[0] - 2, max_dis_pos[0] + 2):
    #         for v in range(max_dis_pos[1] - 2, max_dis_pos[1] + 2):
    #             pick_obs[u][v] = (255, 255, 255)
    #             vis_max_dis[u][v] = (255, 255, 255)
    #
    #     visual_img = np.concatenate((
    #         vis_convex,
    #         cv2.cvtColor(pick_obs, cv2.COLOR_RGB2BGR),
    #         vis_avg_dis,
    #         vis_max_dis),
    #         axis=1)
    #
    #     if id == -1:
    #         return visual_img.copy()
    #
    #     base1 = f'{self.name}-{self.task}-{self.crop_size}'
    #     head = os.path.join('visual_at_validate', base1)
    #     if not os.path.exists(head):
    #         os.makedirs(head)
    #
    #     file_name = os.path.join(head, f'validate-{id}')
    #     print(f"saved {file_name}")
    #     cv2.imwrite(f'{file_name}.jpg', visual_img)

    # def visualize_place_critic(self, input_img, pick_pos, critic_map, id):
    #     pick_obs = input_img.copy()
    #     p0 = pick_pos
    #
    #     for u in range(max(0, p0[0] - 2), min(320, p0[0] + 2)):
    #         for v in range(max(0, p0[1] - 2), min(160, p0[1] + 2)):
    #             pick_obs[u][v] = (255, 0, 0)
    #
    #     vis_critic = np.float32(critic_map[0])
    #     place_index = np.argmax(vis_critic)
    #     vis_critic = vis_critic - np.min(vis_critic)
    #     vis_critic = 255 * vis_critic / np.max(vis_critic)
    #     # vis_critic = 255 - vis_critic
    #     vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)
    #
    #     # for u in range(place_pos[0] - 2, place_pos[0] + 2):
    #     #     for v in range(place_pos[1] - 2, place_pos[1] + 2):
    #     #         pick_obs[u][v] = (255, 255, 255)
    #     #         vis_critic[u][v] = (255, 255, 255)
    #
    #     visual_img = np.concatenate((
    #         cv2.cvtColor(pick_obs, cv2.COLOR_RGB2BGR),
    #         vis_critic),
    #         axis=1)
    #
    #     base1 = f'{self.name}-{self.task}-{self.crop_size}'
    #     head = os.path.join('draw', base1)
    #     if not os.path.exists(head):
    #         os.makedirs(head)
    #
    #     file_name = os.path.join(head, f'{self.total_iter + id}')
    #     print(f"saved {file_name}")
    #     cv2.imwrite(f'{file_name}.jpg', visual_img)

    def critic_forward(self, dataset, num_iter, visualize):
        for i in range(num_iter):
            if self.use_goal_image:
                obs, goal, act, metric, step = dataset.sample_index(last_index=100, goal_images=True)
                input_image = np.concatenate((obs, goal), axis=2)
            else:
                obs, act, metric, step = dataset.sample_index()
                input_image = obs.copy()

            p0 = act[0][:2]

            critic_map = self.critic_model.forward(input_image, p0)

            if visualize:
                self.visualize_place_critic(input_image[:, :, :3], p0, critic_map, i + self.total_iter)

        self.total_iter += num_iter

    # def visualize_pick_aff(self, input_img, aff_map, id):
    #     pick_obs = input_img[:, :, :3].copy()
    #     aff_map = aff_map[:, :, 0]
    #     vis_aff = np.float32(aff_map).copy()
    #     pick_index = np.argmax(vis_aff)
    #     argmax = np.unravel_index(pick_index, shape=vis_aff.shape)
    #     pick_pos = argmax[:2]
    #     vis_aff[vis_aff.nonzero()] = vis_aff[vis_aff.nonzero()] - np.min(vis_aff[vis_aff.nonzero()])
    #     vis_aff = 255 * vis_aff / np.max(vis_aff)
    #     # vis_aff = 255 - vis_aff
    #     vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
    #
    #     for u in range(pick_pos[0] - 2, pick_pos[0] + 2):
    #         for v in range(pick_pos[1] - 2, pick_pos[1] + 2):
    #             pick_obs[u][v] = (255, 0, 0)
    #
    #     visual_img = np.concatenate((
    #         cv2.cvtColor(pick_obs, cv2.COLOR_RGB2BGR),
    #         vis_aff),
    #         axis=1)
    #
    #     base1 = f'{self.name}-{self.task}-{self.crop_size}'
    #     head = os.path.join('visual_aff', base1)
    #     if not os.path.exists(head):
    #         os.makedirs(head)
    #
    #     file_name = os.path.join(head, f'0409-{id}')
    #     print(f"saved {file_name}")
    #     cv2.imwrite(f'{file_name}.jpg', visual_img)

    def aff_forward(self, dataset, num_iter, visualize):
        for i in range(num_iter):
            if self.use_goal_image:
                obs, goal, act, metric, step = dataset.sample_index(last_index=100, goal_images=True)
                input_image = np.concatenate((obs, goal), axis=2)
            else:
                obs, act, metric, step = dataset.sample_index()
                input_image = obs.copy()

            p0 = act[0][:2]
            p1_list = act[:, 2:]

            # Do data augmentation (perturb rotation and translation).
            pixels_list = [p0]
            pixels_list.extend(p1_list)

            if self.attn_no_targ and self.use_goal_image:
                maxdim = int(input_image.shape[2] / 2)
                input_only = input_image[:, :, :maxdim]
                aff_map = self.attention_model.forward(input_only)
            else:
                aff_map = self.attention_model.forward(input_image)

            if visualize:
                self.visualize_pick_aff(input_image, aff_map, i + self.total_iter)

        self.total_iter += num_iter

    def train_aff(self, dataset, num_iter, writer, visualize):
        for i in range(num_iter):
            if self.use_goal_image:
                obs, goal, act, metric, step = dataset.sample_index(last_index=100, goal_images=True)
                input_image = np.concatenate((obs, goal), axis=2)
            else:
                obs, act, metric, step = dataset.sample_index()
                input_image = obs.copy()

            p0 = act[0][:2]
            p1_list = act[:, 2:]

            # Do data augmentation (perturb rotation and translation).
            pixels_list = [p0]
            pixels_list.extend(p1_list)
            # original_pixels = pixels_list
            input_image, pixels = utils.perturb(input_image, pixels_list)
            p0 = pixels[0]
            p1_list = pixels[1:]

            critic_map = self.critic_model.forward_aff_gt(input_image, p0)
            critic_max = critic_map.numpy().max()

            if self.attn_no_targ and self.use_goal_image:
                maxdim = int(input_image.shape[2] / 2)
                input_only = input_image[:, :, :maxdim]
                loss0 = self.attention_model.train(input_only, p0, critic_max)
            else:
                loss0 = self.attention_model.train(input_image, p0, critic_max)
            with writer.as_default():
                tf.summary.scalar('attention_loss', self.attention_model.metric.result(), step=self.total_iter+i)

            print(f'Train Iter: {self.total_iter + i} aff Loss: {loss0:.4f}')

        self.total_iter += num_iter
        self.save_aff()

    # def visual_aff_critic(self, input_image, p0_pixel, p1_pixel, cable_list, critic_list, critic_score):
    #     obs_img = input_image[:, :, :3]
    #
    #     # goal_img = input_image[:, :, 6:9]
    #     #
    #     # base1 = f'{self.name}-{self.task}-{self.crop_size}'
    #     # head = os.path.join('visual_aff', base1)
    #     # if not os.path.exists(head):
    #     #     os.makedirs(head)
    #     #
    #     # file_name = os.path.join(head, f'0511-goal')
    #     # print(f"saved {file_name}")
    #     # cv2.imwrite(f'{file_name}.jpg', goal_img)
    #     # print('saves')
    #
    #     for u in range(max(0, p0_pixel[0] - 2), min(320, p0_pixel[0] + 2)):
    #         for v in range(max(0, p0_pixel[1] - 2), min(160, p0_pixel[1] + 2)):
    #             obs_img[u][v] = (255, 0, 0)
    #
    #     for u in range(max(0, p1_pixel[0] - 2), min(320, p1_pixel[0] + 2)):
    #         for v in range(max(0, p1_pixel[1] - 2), min(160, p1_pixel[1] + 2)):
    #             obs_img[u][v] = (255, 255, 255)
    #
    #     vis_critic = np.float32(critic_score[0])
    #     vis_critic = vis_critic - np.min(vis_critic)
    #     vis_critic = 255 * vis_critic / np.max(vis_critic)
    #     vis_critic = 255 - vis_critic
    #     vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)
    #
    #     cable_score = []
    #     vis_cable_critic_list = []
    #
    #     for i in range(0, len(cable_list)):
    #         if self.out_logits == 3:
    #             max_dis, avg_dis, convex = critic_list[i, :, :, 0], critic_list[i, :, :, 1], critic_list[i, :, :, 2]
    #             critic = convex * self.cvx_weight - max_dis * self.max_weight - avg_dis * self.avg_weight
    #         elif self.out_logits == 2:
    #             max_dis, avg_dis = critic_list[i, :, :, 0], critic_list[i, :, :, 1]
    #             critic = 0 - self.max_weight * max_dis - self.avg_weight * avg_dis
    #         else:
    #             critic = critic_list[i, :, :, 0]
    #         # if i % 4 == 0:
    #         #     vis_cable_critic = np.float32(critic)
    #         #     vis_cable_critic = vis_cable_critic - np.min(vis_cable_critic)
    #         #     vis_cable_critic = 255 * vis_cable_critic / np.max(vis_cable_critic)
    #         #     vis_cable_critic = 255 - vis_cable_critic
    #         #     vis_cable_critic = cv2.applyColorMap(np.uint8(vis_cable_critic), cv2.COLORMAP_JET)
    #         #     vis_cable_critic_list.append(vis_cable_critic)
    #         cable_score.append(np.max(critic))
    #
    #     low_score = np.min(np.array(cable_score))
    #     vis_attention = np.full((320, 160, 1), low_score)
    #
    #     for i in range(0, len(cable_list)):
    #         cable = cable_list[i]
    #         score = cable_score[i]
    #         for u in range(max(0, cable[0] - 2), min(320, cable[0] + 2)):
    #             for v in range(max(0, cable[1] - 2), min(160, cable[1] + 2)):
    #                 vis_attention[u][v] = score
    #
    #     vis_attention = vis_attention - np.min(vis_attention)
    #     vis_attention = 255 * vis_attention / np.max(vis_attention)
    #     vis_attention = 255 - vis_attention
    #     vis_attention = cv2.applyColorMap(np.uint8(vis_attention), cv2.COLORMAP_JET)
    #
    #     # if self.use_aff:
    #     #     vis_aff = np.float32(attention[0])
    #     # else:
    #     #     vis_aff = np.float32(attention)
    #     # vis_aff = vis_aff - np.min(vis_aff)
    #     # vis_aff = 255 * vis_aff / np.max(vis_aff)
    #     # vis_aff = 255 - vis_aff
    #     # vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
    #
    #     vis_img = np.concatenate((
    #         vis_attention,
    #         obs_img,
    #         vis_critic),
    #         axis=1)
    #
    #     # vis_critic_img = np.concatenate(np.array(vis_cable_critic_list), axis=1)
    #     # cv2.imwrite(f'./draw/vis_img-{p0_pixel}.jpg', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
    #     # cv2.imwrite(f'./draw/aff_critic_map-{p0_pixel}.jpg', cv2.cvtColor(vis_critic_img, cv2.COLOR_RGB2BGR))
    #     # print('saved img')
    #
    #     return vis_img

    def act(self, obs, info, times=0, debug_imgs=False, goal=None, task=None):
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
            indexs = np.transpose(np.nonzero(obs[:, :, 0]))
            index = random.choice(indexs)
            u1 = index[0]
            v1 = index[1]
            print((u1, v1))
            p0_pixel = (u1, v1)

        elif self.critic_pick:
            p0_list = []
            indexs = np.transpose(np.nonzero(obs[:, :, 0]))

            for i in range(0, self.critic_num):
                index = random.choice(indexs)
                u1 = index[0]
                v1 = index[1]
                p0_list.append((u1, v1))

            img_pick = obs.copy()
            critic_list = self.critic_model.forward_with_plist(img_pick, p0_list)

            max_score = -float('inf')
            max_i = 0

            for i in range(0, self.critic_num):
                cable_score = np.max(critic_list[i, :, :, 0])
                if cable_score > max_score:
                    max_i = i
                    max_score = cable_score

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

            mask = np.where(obs == 0, 0, 1)

            attention = attention - np.min(attention)
            attention = attention * mask

            argmax = np.argmax(attention)
            argmax = np.unravel_index(argmax, shape=attention.shape)

            p0_pixel = argmax[1:3]

            # self.visualize_pick_aff(obs.copy(), attention[0], argmax[1])

        img_critic = obs.copy()
        output = self.critic_model.forward(img_critic, p0_pixel)
        critic_score = output[:, :, :, 0]
        argmax = np.argmax(critic_score)
        argmax = np.unravel_index(argmax, shape=critic_score.shape)
        p1_pixel = argmax[1:3]

        # compute coverage
        """ TODO """

        act = np.array([p0_pixel, p1_pixel])
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

    def save_critic_with_epoch(self, remove=True):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        critic_fname = 'critic-ckpt-%d.h5' % self.total_iter
        critic_fname = os.path.join(self.models_dir, critic_fname)
        if remove and self.prev_critic_fname is not None:
            cmd = "rm %s" % self.prev_critic_fname
            call(cmd, shell=True)
        self.prev_critic_fname = critic_fname
        self.critic_model.save(critic_fname)

    def save_aff_with_epoch(self, remove=True):
        """Save models."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        aff_fname = 'attention-ckpt-%d.h5' % self.total_iter
        aff_fname = os.path.join(self.models_dir, aff_fname)
        if remove and self.prev_aff_fname is not None:
            cmd = "rm %s" % self.prev_aff_fname
            call(cmd, shell=True)
        self.prev_aff_fname = aff_fname
        self.attention_model.save(aff_fname)

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

    def __init__(self, name, task, use_goal_image=0, load_critic_dir='xxx', load_aff_dir='xxx', out_logits=1,
                 without_global=False, critic_pick=False, random_pick=False, learning_rate=1e-4):
        super().__init__(name, task, use_goal_image=use_goal_image, out_logits=out_logits,
                         critic_pick=critic_pick, random_pick=random_pick)

        self.attention_model = Affordance(input_shape=self.input_shape,
                                          preprocess=self.preprocess,
                                          learning_rate=learning_rate,
                                          )

        self.critic_model = Critic_MLP(input_shape=self.input_shape,
                                       preprocess=self.preprocess,
                                       out_logits=self.out_logits,
                                       learning_rate=learning_rate,
                                       without_global=without_global,
                                       )

        if load_aff_dir != 'xxx':
            print('*' * 50)
            print('*' * 20 + 'load aff model' + '*' * 20)
            print('*' * 3 + f'load_aff_dir {load_aff_dir}' + '*' * 3)
            self.attention_model.load(load_aff_dir)
            print('*' * 50)

        if load_critic_dir != 'xxx':
            print('*' * 50)
            print('*' * 20 + 'load critic model' + '*' * 20)
            print('*' * 6 + f'cvx_weight {self.cvx_weight} max_weight {self.max_weight} avg_weight {self.avg_weight}' + '*' * 6)
            print('*' * 20 + f'use_aff {self.use_aff}' + '*' * 20)
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


