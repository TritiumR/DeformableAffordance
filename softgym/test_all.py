import os
import argparse
import numpy as np
import cv2

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif

import agents
from models import Critic_MLP

import pyflex
from matplotlib import pyplot as plt
import tensorflow as tf

import multiprocessing
import random
import pickle


def run_jobs(args, env, agent):
    full_covered_area = 0.0625
    if args.set_flat:
        if args.env_name == 'ClothFlatten':
            # from flat configuration
            full_covered_area = env._set_to_flatten()
        elif args.env_name == 'RopeConfiguration':
            # from goal configuration
            env.set_state(env.goal_state[0])
            full_distance = env.compute_reward()
        pyflex.step()

        step_i = 0

        while step_i < args.step:
            # print("step_i: ", step_i)
            if args.env_name == 'ClothFlatten':
                prev_obs, prev_depth = pyflex.render_cloth()
            elif args.env_name == 'RopeConfiguration':
                env.action_tool.hide()
                prev_obs, prev_depth = pyflex.render()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]
            prev_depth = prev_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            mask = np.where(prev_depth[:, :, 0] < 0.348, 255, 0)

            # crumple the cloth by grabbing corner
            if args.env_name == 'ClothFlatten':
                # if step_i == 0:
                mask = prev_obs[:, :, 0]
                indexs = np.transpose(np.where(mask != 0))
                corner_id = random.randint(0, 3)
                top, left = indexs.min(axis=0)
                bottom, right = indexs.max(axis=0)

                corners = [[top, left],
                           [top, right],
                           [bottom, right],
                           [bottom, left]]
                u1 = (corners[corner_id][1]) * 2.0 / 720 - 1
                v1 = (corners[corner_id][0]) * 2.0 / 720 - 1

                u2 = random.uniform(-1., 1.)
                v2 = random.uniform(-1., 1.)

                action = np.array([u1, v1, u2, v2])

            elif args.env_name == 'RopeConfiguration':
                indexs = np.transpose(np.nonzero(mask))
                index = random.choice(indexs)
                u1 = (index[1]) * 2.0 / 720 - 1
                v1 = (index[0]) * 2.0 / 720 - 1

                u2 = max(min(np.random.normal(u1, scale=0.4), 0.999), -1.)
                v2 = max(min(np.random.normal(v1, scale=0.4), 0.999), -1.)

                action = np.array([u1, v1, u2, v2])

            _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)

            if env.action_tool.not_on_cloth:
                print(f'{step_i} not on cloth')
                if args.env_name == 'ClothFlatten':
                    # from flat configuration
                    full_covered_area = env._set_to_flatten()
                elif args.env_name == 'RopeConfiguration':
                    # from goal configuration
                    if args.shape == 'S':
                        env.set_state(env.goal_state[0])
                    elif args.shape == 'O':
                        env.set_state(env.goal_state[1])
                    elif args.shape == 'M':
                        env.set_state(env.goal_state[2])
                    elif args.shape == 'C':
                        env.set_state(env.goal_state[3])
                    elif args.shape == 'U':
                        env.set_state(env.goal_state[4])
                    full_distance = -env.compute_reward()
                pyflex.step()

                step_i = 0
                continue
            if args.env_name == 'RopeConfiguration':
                env.action_tool.hide()
            step_i += 1
            if args.env_name == 'ClothFlatten' and step_i == args.step:
                now_area = env._get_current_covered_area(pyflex.get_positions())
                now_percent = now_area / full_covered_area
                if now_percent >= (0.65 - args.step * 0.05):
                    step_i = 0
                    continue
            elif args.env_name == 'RopeConfiguration':
                now_distance = env.compute_reward()
                if now_distance >= -0.06:
                    step_i = 0
                    continue

    env.start_record()

    if args.env_name == 'ClothFlatten':
        crump_area = env._get_current_covered_area(pyflex.get_positions())
        crump_percent = crump_area / full_covered_area
        print("crump percent: ", crump_percent)
    elif args.env_name == 'RopeConfiguration':
        crump_distance = env.compute_reward()
        print("crump distance: ", crump_distance)

    max_percent = -float("inf")

    in_step = 0
    for i in range(args.test_step):
        in_step += 1
        if args.env_name == 'ClothFlatten':
            crump_obs, crump_depth = pyflex.render_cloth()
        elif args.env_name == 'RopeConfiguration':
            env.action_tool.hide()
            crump_obs, crump_depth = pyflex.render()

        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        crump_obs = cv2.resize(crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        if args.expert_pick:
            reverse_p0_pixel = (int((action[3] + 1.) / 2 * args.image_size), int((action[2] + 1.) / 2 * args.image_size))
            action = agent.act(crump_obs.copy(), p0=reverse_p0_pixel)
        else:
            action = agent.act(crump_obs.copy())

        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)

        if args.env_name == 'ClothFlatten':
            curr_area = env._get_current_covered_area(pyflex.get_positions())
            curr_percent = curr_area / full_covered_area
            print("curr percent: ", i, curr_percent)
            if (curr_percent >= 0.85):
                break
        elif args.env_name == 'RopeConfiguration':
            curr_distance = env.compute_reward()
            if curr_distance > max_percent:
                max_percent = curr_distance
            if curr_distance >= -0.055:
                break
            print("curr distance: ", i, curr_distance)

    if args.env_name == 'ClothFlatten':
        normalize_score = (curr_percent - crump_percent) / (1 - crump_percent)
        if curr_percent >= 0.75:
            result = 'success'
        else:
            result = 'fail'

    elif args.env_name == 'RopeConfiguration':
        normalize_score = (curr_distance - crump_distance) / (0 - crump_distance)
        if max_percent >= -0.055:
            result = 'success'
        else:
            result = 'fail'

    if args.env_name == 'RopeConfiguration':
        env.action_tool.hide()
    print(normalize_score)
    cv2.imwrite(f'./visual/1101-09/{args.exp_name}-{args.test_id}-{normalize_score}.jpg', crump_obs)
    if args.save_video_dir is not None:
        path_name = os.path.join(args.save_video_dir, agent.name + args.exp_name)
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        save_name = os.path.join(path_name, f'{args.test_id}-{in_step}-{result}-{normalize_score}.gif')
        save_numpy_as_gif(np.array(env.video_frames), save_name)
        print('Video generated and save to {}'.format(save_name))

    env.end_record()
    return 1


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--image_size', type=int, default=320, help='Size of the input')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--shape', type=str, default='U')
    parser.add_argument('--set_flat', type=int, default=1)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--test_step', default=20, type=int)
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--test_id', type=int, default=1, help='which test')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--unet', default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--without_global', action='store_true')
    parser.add_argument('--expert_pick',    action='store_true')
    parser.add_argument('--critic_pick',    action='store_true')
    parser.add_argument('--random_pick',    action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic-{args.num_demos}'

    # Initialize agent and limit random dataset sampling to fixed set.
    tf.random.set_seed(0)

    agent = agents.names[args.agent](name,
                                     args.task,
                                     image_size=args.image_size,
                                     use_goal_image=args.use_goal_image,
                                     load_critic_dir=args.load_critic_dir,
                                     load_aff_dir=args.load_aff_dir,
                                     out_logits=args.out_logits,
                                     without_global=args.without_global,
                                     expert_pick=args.expert_pick,
                                     critic_pick=args.critic_pick,
                                     random_pick=args.random_pick,
                                     unet=args.unet,
                                     use_mask=args.use_mask
                                     )

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    run_jobs(args, env, agent)

if __name__ == '__main__':
    main()
