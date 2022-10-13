#!/usr/bin/env python
import os
import sys
import json
import argparse
import cv2
import pickle
import numpy as np
import random


class Dataset:
    def __init__(self, path='', process=0, max_load=-1, demo_times=5, discard_first=False, validate=0):
        """A simple RGB-D image dataset."""
        self.path = path
        self.episode_id = []
        self.episode_set = []

        # for parallel run
        self.process = process
        self.num_demo = 0

        # max data in cache
        self.max_load = max_load
        self.cache_size = 0

        # data type
        self.demo_times = demo_times
        self.discard_first = discard_first
        self.validate = validate

        # Track existing dataset if it exists.
        if process == 0:
            path = self.path
            if os.path.exists(path):
                for fname in sorted(os.listdir(path)):
                    if '.pkl' in fname:
                        num_samples = int(fname[(fname.find('-') + 1):-4])
                        self.episode_id += [self.num_episodes] * num_samples

        self._cache = dict()
        # print("num_episodes", self.num_episodes)
        # print("episode_id", self.episode_id[0], self.episode_id[1], self.episode_id[2], self.episode_id[3], self.episode_id[4])

    @property
    def num_episodes(self):
        return np.max(np.int32(self.episode_id)) + 1 if self.episode_id else 0

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.episode_set = episodes

    def sample_index(self, need_next=False, set_iepisode=None):
        """Randomly sample from the dataset uniformly.

        The 'cached_load' will use the load (from pickle file) to
        load the list, and then extract the time step `i` within it as the
        data point. I'm also adding a `goal_images` feature to load the last
        information. The last information isn't in a list, so we don't need
        to extract an index. That is, if loading a 1-length time step, we
        should see this:

        In [11]: data = pickle.load( open('last_color/000099-1.pkl', 'rb') )
        In [12]: data.shape
        Out[12]: (3, 480, 640, 3)

        In [13]: data = pickle.load( open('color/000099-1.pkl', 'rb') )
        In [14]: data.shape
        Out[14]: (1, 3, 480, 640, 3)

        Update: now using goal_images for gt_state, but here we should interpret
        `goal_images` as just giving the 'info' portion to the agent.
        """
        if len(self.episode_set) > 0:
            iepisode = np.random.choice(self.episode_set)
        else:
            iepisode = np.random.choice(range(self.num_episodes))

        # Get length of episode, then sample time within it.
        is_episode_sample = np.int32(self.episode_id) == iepisode
        episode_samples = np.argwhere(is_episode_sample).squeeze().reshape(-1)

        ep_len = len(episode_samples)

        i = random.randint(0, ep_len - 1)

        def load(iepisode):
            path = self.path
            fname = f'{iepisode:06d}-{len(episode_samples)}.pkl'
            return pickle.load(open(os.path.join(path, fname), 'rb'))

        def cached_load(iepisode, index=-1):
            if self.cache_size == self.max_load:
                if iepisode not in self._cache:
                    return load(iepisode)
            if iepisode not in self._cache:
                self._cache[iepisode] = load(iepisode)
                self.cache_size += 1
            return self._cache[iepisode]

        if set_iepisode is not None:
            iepisode = set_iepisode

        if self.demo_times > 1:
            data = cached_load(iepisode, i)
            obs = data['obs']

            action = data['action']
            metric = data['area']

            step = i

            not_on_cloth = data['not_on_cloth']

            if need_next:
                curr_obs = data['curr_obs']
                return obs, curr_obs, action, metric, step, not_on_cloth, iepisode

            return obs, action, metric, step, not_on_cloth, iepisode

        if self.demo_times == 1:
            action = []
            metric = []

            data = cached_load(iepisode, i)
            obs = data['obs']

            action.append(data['action'])
            metric.append(data['area'])

            step = i

            not_on_cloth = data['not_on_cloth']

            if need_next:
                curr_obs = data['curr_obs']
                return obs, curr_obs, action, metric, step, not_on_cloth, iepisode

            return obs, action, metric, step, not_on_cloth, iepisode
