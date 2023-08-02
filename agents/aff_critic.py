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
    def __init__(self, name, task, image_size=160, step=1):
        """Creates Transporter agent with attention and transport modules."""
        self.name = name
        self.task = task
        self.image_size = image_size
        self.step = step
        self.total_iter = 0
        self.input_shape = (image_size, image_size, 4)
        self.models_dir = os.path.join('checkpoints', self.name + f'-step-{self.step}')

        self.reward_cache = dict()

    def compute_reward(self, metric, not_on_cloth, curr_obs=None, iepisode=None):
        m_len = len(metric)
        reward = []

        if not_on_cloth == 1:
            # print('not on cloth')
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
            if iepisode in self.reward_cache:
                return self.reward_cache[iepisode]
            attention = self.attention_model.forward_batch(curr_obs.copy())
            for i in range(0, m_len):
                if self.task == 'cloth-flatten':
                    gt_state = np.max(attention[i])
                    curr_percent = metric[i][1] * 50
                elif self.task == 'rope-configuration':
                    gt_state = np.min(attention[i])
                    curr_percent = metric[i][1] * 100
                reward_i = (curr_percent * 2 + gt_state) / 3
                reward.append(reward_i)

            self.reward_cache[iepisode] = reward
        return reward

    def train_aff(self, dataset, num_iter, writer, batch, no_perturb=False):
        for i in range(num_iter):
            obs, act, _, _, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

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
                    # print('no perturb')
                    pass
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
                aff_pred = self.attention_model.forward(input_image.copy())
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

    def train_critic(self, dataset, num_iter, writer, batch=1, no_perturb=False):
        for i in range(num_iter):
            # for batch training
            input_batch = []
            p0_batch = []
            p1_list_batch = []
            reward_batch = []

            for bh in range(batch):
                curr_obs = None
                if self.step > 1:
                    obs, curr_obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=True)
                else:
                    obs, act, metric, step, not_on_cloth, iepisode = dataset.sample_index(need_next=False)

                input_image = obs.copy()

                p0 = [min(self.input_shape[0] - 1, int((act[0][1] + 1.) * 0.5 * self.input_shape[0])),
                      min(self.input_shape[0] - 1, int((act[0][0] + 1.) * 0.5 * self.input_shape[0]))]

                p1_list = []
                for point in act:
                    p1 = [int((point[3] + 1.) * 0.5 * self.input_shape[0]), int((point[2] + 1.) * 0.5 * self.input_shape[0])]
                    p1_list.append(p1)

                reward = self.compute_reward(metric, not_on_cloth[0], curr_obs, iepisode)
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
            loss = self.critic_model.train_batch(input_batch, p0_batch, p1_list_batch, reward_batch)
            with writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_model.metric.result(), step=self.total_iter+i)
            print(f'Train Iter: {self.total_iter + i} Critic Loss: {loss:.4f}')

        self.total_iter += num_iter
        self.save_critic()

    def act(self, obs, p0=None):
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
        # Attention model forward pass.
        img_aff = obs.copy()
        attention = self.attention_model.forward(img_aff)

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

    def __init__(self, name, task, image_size=160, load_critic_dir='xxx', load_aff_dir='xxx', step=1, learning_rate=1e-4):
        super().__init__(name, task, image_size=image_size, step=step)

        if load_critic_dir != 'xxx':
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

        if load_aff_dir != 'xxx':
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

        self.attention_model = Affordance(input_shape=self.input_shape,
                                          preprocess=self.aff_preprocess,
                                          learning_rate=learning_rate,
                                          )

        self.critic_model = Critic_MLP(input_shape=self.input_shape,
                                       preprocess=self.critic_preprocess,
                                       learning_rate=learning_rate,
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
            print('*' * 3 + f'load_critic_dir {load_critic_dir}' + '*' * 3)
            self.critic_model.load(load_critic_dir)
            print('*' * 50)




