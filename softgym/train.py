import os.path as osp
import os
import argparse
import numpy as np
import cv2

from softgym.utils.visualization import save_numpy_as_gif
from softgym.dataset import Dataset
import agents
from models import Critic_MLP

import pyflex
from matplotlib import pyplot as plt
import tensorflow as tf

import random
import pickle


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--num_iters', type=int, default=1, help='How many iterations do you need for training')
    parser.add_argument('--learning_rate',  default=1e-4, type=float)
    parser.add_argument('--demo_times', default=10, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--max_load',       default=-1, type=int)
    parser.add_argument('--batch',          default=20, type=int)
    parser.add_argument('--model', default='critic', type=str)
    parser.add_argument('--no_perturb', action='store_true')
    args = parser.parse_args()

    dataset = Dataset(os.path.join('data', f"{args.task}-{args.suffix}"), max_load=args.max_load,
                      demo_times=args.demo_times)

    # Set up tensorboard logger.
    train_log_dir = os.path.join('logs', args.agent, args.task, args.exp_name, 'train')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set the beginning of the agent name.
    name = f'{args.task}-{args.exp_name}'

    # Initialize agent and limit random dataset sampling to fixed set.
    tf.random.set_seed(0)

    # Limit random data sampling to fixed set.
    num_demos = int(args.num_demos)

    # Given `num_demos`, only sample up to that point, and not w/replacement.
    train_episodes = np.random.choice(range(num_demos), num_demos, False)
    dataset.set(train_episodes)

    agent = agents.names[args.agent](name,
                                     args.task,
                                     load_critic_dir=args.load_critic_dir,
                                     load_aff_dir=args.load_aff_dir,
                                     learning_rate=args.learning_rate,
                                     step=args.step,
                                     )

    agent.get_mean_and_std(os.path.join('data', f"{args.task}-{args.suffix}"), args.model)

    while agent.total_iter < args.num_iters:
        if args.model == 'critic':
            # Train critic.
            tf.keras.backend.set_learning_phase(1)
            agent.train_critic(dataset, num_iter=args.num_iters // 10, writer=train_summary_writer, batch=args.batch, no_perturb=args.no_perturb)
            tf.keras.backend.set_learning_phase(0)
        if args.model == 'aff':
            # Train aff.
            tf.keras.backend.set_learning_phase(1)
            agent.train_aff(dataset, num_iter=args.num_iters // 10, writer=train_summary_writer, batch=args.batch, no_perturb=args.no_perturb)
            tf.keras.backend.set_learning_phase(0)


if __name__ == '__main__':
    main()
