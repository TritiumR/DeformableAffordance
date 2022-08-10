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

import multiprocessing
import random
import pickle


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--validate', default=0, type=int)
    parser.add_argument('--num_iters', type=int, default=1, help='How many iterations do you need for training')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--learning_rate',  default=1e-4, type=float)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--demo_times', default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--without_global', action='store_true')
    parser.add_argument('--max_load',       default=-1, type=int)
    parser.add_argument('--batch',          default=1, type=int)
    args = parser.parse_args()

    dataset = Dataset(os.path.join('data', f"{args.task}-{args.suffix}"), max_load=args.max_load,
                      demo_times=args.demo_times)

    # Set up tensorboard logger.
    train_log_dir = os.path.join('logs', args.agent, args.task, args.exp_name, 'train')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic-{args.num_demos}'

    # Initialize agent and limit random dataset sampling to fixed set.
    tf.random.set_seed(0)

    agent = agents.names[args.agent](name,
                                     args.task,
                                     use_goal_image=args.use_goal_image,
                                     load_critic_dir=args.load_critic_dir,
                                     load_aff_dir=args.load_aff_dir,
                                     out_logits=args.out_logits,
                                     learning_rate=args.learning_rate,
                                     without_global=args.without_global
                                     )

    # Limit random data sampling to fixed set.
    num_demos = int(args.num_demos)

    # Given `num_demos`, only sample up to that point, and not w/replacement.
    train_episodes = np.random.choice(range(num_demos - args.validate), num_demos, False)
    dataset.set(train_episodes)

    while agent.total_iter < args.num_iters:
        # Train agent.
        tf.keras.backend.set_learning_phase(1)
        agent.train(dataset, num_iter=args.num_iters // 20, writer=train_summary_writer, batch=args.batch)
        tf.keras.backend.set_learning_phase(0)

        # agent.train() concludes with agent.save() inside it, then exit.
        if args.save_zero:
            print('We are now exiting due to args.save_zero...')
            agent.total_iter = num_train_iters
            continue

    # num_train_iters = num_train_iters * 2
    # performance = []
    # while agent.total_iter < num_train_aff_iters + num_train_critic_iters:
    #     # Train agent.
    #     tf.keras.backend.set_learning_phase(1)
    #     agent.train_aff(dataset, num_iter=num_train_aff_iters // 10, writer=train_summary_writer, visualize=args.visualize)
    #     tf.keras.backend.set_learning_phase(0)
    #
    #     # agent.train() concludes with agent.save() inside it, then exit.
    #     if args.save_zero:
    #         print('We are now exiting due to args.save_zero...')
    #         agent.total_iter = num_train_iters
    #         continue

if __name__ == '__main__':
    main()