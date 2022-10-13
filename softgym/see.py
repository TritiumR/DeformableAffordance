import os
import argparse
import numpy as np
import cv2

import agents
from models import Critic_MLP

from matplotlib import pyplot as plt
import tensorflow as tf

import random
import pickle


def load(path, iepisode, step):
    field_path = path
    fname = f'{iepisode:06d}-{step}.pkl'
    return pickle.load(open(os.path.join(field_path, fname), 'rb'))


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--image_size', default=160, type=int)
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--critic_depth', default=1, type=int)
    parser.add_argument('--demo_times', default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--load_critic_mean_std_dir', default='xxx')
    parser.add_argument('--load_aff_mean_std_dir', default='xxx')
    parser.add_argument('--without_global', action='store_true')
    parser.add_argument('--model', default='critic', type=str)
    parser.add_argument('--only_depth', action='store_true')
    args = parser.parse_args()

    path = os.path.join('data', f"{args.task}-{args.suffix}")

    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic-{args.num_demos}-{args.exp_name}'

    agent = agents.names[args.agent](name,
                                     args.task,
                                     image_size=args.image_size,
                                     load_critic_dir=args.load_critic_dir,
                                     load_aff_dir=args.load_aff_dir,
                                     load_critic_mean_std_dir=args.load_critic_mean_std_dir,
                                     load_aff_mean_std_dir=args.load_aff_mean_std_dir,
                                     out_logits=args.out_logits,
                                     learning_rate=0,
                                     without_global=args.without_global,
                                     critic_depth=args.critic_depth,
                                     only_depth=args.only_depth,
                                     )

    for i in range(args.num_demos):
        data = load(path, i, args.step)
        action = data['action']
        crump_obs = data['obs']

        W, H = np.array(crump_obs).shape[:2]
        # print("W H = ", W, H)
        p0 = (action[0][0], action[0][1])
        p0_pixel = (int((p0[1] + 1.) / 2. * H), int((p0[0] + 1.) / 2. * W))

        curr_obs = data['curr_obs']
        for j in range(len(curr_obs)):
            curr_img = curr_obs[j]
            attention = agent.attention_model.forward(curr_img)
            vis_aff = np.array(attention[0])
            vis_aff = vis_aff - np.min(vis_aff)
            vis_aff = 255 * vis_aff / np.max(vis_aff)
            vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

            img = np.concatenate((
                cv2.cvtColor(curr_img[:, :, :-1], cv2.COLOR_BGR2RGB),
                vis_aff),
                axis=1)

            cv2.imwrite(f'./see/{args.exp_name}-{i}-{j}.jpg', img)
            print(f"saved ./see/{args.exp_name}-{i}-{j}.jpg")


if __name__ == '__main__':
    main()
