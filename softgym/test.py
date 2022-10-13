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


def visualize_aff_critic(obs, agent, args):
    if args.only_depth:
        img_aff = obs[:, :, -1:].copy()
    else:
        img_aff = obs.copy()
    attention = agent.attention_model.forward(img_aff)

    # depth = obs[:, :, -1:].copy()
    # mask = np.where(depth == 0, 0, 1)
    # state_map = np.where(depth == 0, 0, attention)
    if args.env_name == 'ClothFlatten':
        state_score = int(np.max(attention) * 2)

    elif args.env_name == 'RopeConfiguration':
        state_score = int(np.min(attention) * 10)
        attention = -attention

    attention = attention - np.min(attention)
    argmax = np.argmax(attention)
    argmax = np.unravel_index(argmax, shape=attention.shape)

    p0_pixel = argmax[1:3]
    print(p0_pixel)

    img_critic = obs.copy()
    critic = agent.critic_model.forward(img_critic, p0_pixel)

    if args.env_name == 'ClothFlatten':
        argmax = np.argmax(critic)
        argmax = np.unravel_index(argmax, shape=critic.shape)

    elif args.env_name == 'RopeConfiguration':
        argmax = np.argmin(critic)
        argmax = np.unravel_index(argmax, shape=critic.shape)

    p1_pixel = argmax[1:3]
    print(p1_pixel)

    vis_aff = np.array(attention[0])
    vis_critic = np.array(critic[0])

    if args.env_name == 'RopeConfiguration':
        vis_aff = np.exp(vis_aff) / np.sum(np.exp(vis_aff))
        vis_critic = -vis_critic
        vis_critic = np.exp(vis_critic) / np.sum(np.exp(vis_critic))

    vis_aff = vis_aff - np.min(vis_aff)
    vis_aff = 255 * vis_aff / np.max(vis_aff)
    vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

    vis_critic = vis_critic - np.min(vis_critic)
    vis_critic = 255 * vis_critic / np.max(vis_critic)
    vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

    img_obs = obs[:, :, :3]

    for u in range(max(0, p0_pixel[0] - 4), min(args.image_size, p0_pixel[0] + 4)):
        for v in range(max(0, p0_pixel[1] - 4), min(args.image_size, p0_pixel[1] + 4)):
            img_obs[u][v] = (255, 0, 0)

    for u in range(max(0, p1_pixel[0] - 4), min(args.image_size, p1_pixel[0] + 4)):
        for v in range(max(0, p1_pixel[1] - 4), min(args.image_size, p1_pixel[1] + 4)):
            img_obs[u][v] = (255, 255, 255)

    vis_img = np.concatenate((cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB), vis_aff, vis_critic), axis=1)

    cv2.imwrite(f'./visual/{args.exp_name}-test-aff_critic-{p0_pixel[0]}-{p0_pixel[1]}-{state_score}.jpg', vis_img)
    print("save to" + f'./visual/{args.exp_name}-test-aff_critic-{p0_pixel[0]}-{p0_pixel[1]}-{state_score}.jpg')


def visualize_aff_state(obs, env, agent, full_covered_area, args, state_crump):
    vis_aff = np.zeros((16, 16))
    gt_aff = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            env.set_state(state_crump)
            p0 = (i * 10, j * 10)
            action = agent.act(obs.copy(), p0=p0)
            _, _, _, _ = env.step(action, record_continuous_video=False, img_size=args.img_size)
            output = agent.critic_model.forward(obs.copy(), p0)
            critic_score = output[:, :, :, 0]
            if args.env_name == 'ClothFlatten':
                gt_area = env._get_current_covered_area(pyflex.get_positions())
                gt_percent = gt_area / full_covered_area
                gt_aff[i][j] = gt_percent
                vis_aff[i][j] = np.max(critic_score)
            elif args.env_name == 'RopeConfiguration':
                gt_distance = env.compute_reward()
                gt_aff[i][j] = gt_distance
                vis_aff[i][j] = np.min(critic_score)

    vis_aff = cv2.resize(vis_aff, (args.image_size, args.image_size))
    gt_aff = cv2.resize(gt_aff, (args.image_size, args.image_size))

    if args.env_name == 'ClothFlatten':
        score = int(np.max(vis_aff) * 2)
        gt_score = int(np.max(gt_aff) * 100)

        if args.exp:
            vis_aff = np.exp(vis_aff) / np.sum(np.exp(vis_aff))
        vis_aff = vis_aff - np.min(vis_aff)
        vis_aff = 255 * vis_aff / np.max(vis_aff)

        if args.exp:
            gt_aff = np.exp(gt_aff) / np.sum(np.exp(gt_aff))
        gt_aff = gt_aff - np.min(gt_aff)
        gt_aff = 255 * gt_aff / np.max(gt_aff)
    elif args.env_name == 'RopeConfiguration':
        score = int(np.min(vis_aff))
        gt_score = -int(np.max(gt_aff) * 100)

        vis_aff = -vis_aff
        if args.exp:
            vis_aff = np.exp(vis_aff) / np.sum(np.exp(vis_aff))
        vis_aff = vis_aff - np.min(vis_aff)
        vis_aff = 255 * vis_aff / np.max(vis_aff)

        gt_aff = np.exp(gt_aff) / np.sum(np.exp(gt_aff))
        gt_aff = gt_aff - np.min(gt_aff)
        gt_aff = 255 * gt_aff / np.max(gt_aff)


    vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
    gt_aff = cv2.applyColorMap(np.uint8(gt_aff), cv2.COLORMAP_JET)

    vis_img = np.concatenate((cv2.cvtColor(obs, cv2.COLOR_BGR2RGB), gt_aff, vis_aff), axis=1)

    cv2.imwrite(f'./visual/{args.exp_name}-aff_max-{gt_score}-{score}.jpg', vis_img)
    print("save to" + f'./visual/{args.exp_name}-aff_max-{gt_score}-{score}.jpg')


def visualize_critic_gt(obs, env, agent, p0, full_covered_area, args, state_crump, aff=False):
    obs_img = obs[:, :, :3].copy()
    p0_pixel = (int((p0[1] + 1.) / 2 * args.image_size), int((p0[0] + 1.) / 2 * args.image_size))

    for u in range(max(0, p0_pixel[0] - 2), min(args.image_size, p0_pixel[0] + 2)):
        for v in range(max(0, p0_pixel[1] - 2), min(args.image_size, p0_pixel[1] + 2)):
            obs_img[u][v] = (255, 0, 0)

    gt_img = np.zeros((16, 16))
    potential_img = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            env.set_state(state_crump)
            p1 = ((i - 8) / 8, (j - 8) / 8)
            action = np.array([p0[0], p0[1], p1[1], p1[0]])
            _, _, _, _ = env.step(action, record_continuous_video=False, img_size=args.img_size)
            if args.env_name == 'ClothFlatten':
                gt_area = env._get_current_covered_area(pyflex.get_positions())
                gt_percent = gt_area / full_covered_area

                if aff:
                    curr_obs, curr_depth = pyflex.render_cloth()
                    curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
                    curr_depth[curr_depth > 5] = 0
                    curr_depth = curr_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
                    curr_obs = np.concatenate([curr_obs, curr_depth], 2)
                    curr_obs = cv2.resize(curr_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                    attention = agent.attention_model.forward(curr_obs)

                    potential = np.max(attention)

                    # vis_aff = np.array(attention[0])
                    # vis_aff = vis_aff - np.min(vis_aff)
                    # vis_aff = 255 * vis_aff / np.max(vis_aff)
                    # vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
                    #
                    # test_img = np.concatenate((cv2.cvtColor(curr_obs[:, :, :-1], cv2.COLOR_BGR2RGB), vis_aff), axis=1)
                    # cv2.imwrite(f'./visual/{args.exp_name}-curr_obs-potential-{i}-{j}-{potential}.jpg', test_img)

                    potential_img[i][j] = potential
                gt_img[i][j] = gt_percent
            elif args.env_name == 'RopeConfiguration':
                gt_distance = env.compute_reward()
                gt_img[i][j] = gt_distance * 100

    gt_img = cv2.resize(gt_img, (args.image_size, args.image_size))

    output = agent.critic_model.forward(obs.copy(), p0_pixel)
    critic_score = output[:, :, :, 0]
    argmax = np.argmax(critic_score)
    argmax = np.unravel_index(argmax, shape=critic_score.shape)
    p1_pixel = argmax[1:3]

    vis_critic = np.float32(critic_score[0])

    if args.env_name == 'ClothFlatten':
        if args.exp:
            vis_critic = np.exp(vis_critic) / np.sum(np.exp(vis_critic))
        vis_critic = vis_critic - np.min(vis_critic)
        vis_critic = 255 * vis_critic / np.max(vis_critic)
        vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

        if args.exp:
            gt_img = np.exp(gt_img) / np.sum(np.exp(gt_img))
        vis_gt = gt_img - np.min(gt_img)
        vis_gt = 255 * vis_gt / np.max(vis_gt)
        vis_gt = cv2.applyColorMap(np.uint8(vis_gt), cv2.COLORMAP_JET)

        if aff:
            potential_img = cv2.resize(potential_img, (args.image_size, args.image_size))
            potential_img = potential_img - np.min(potential_img)
            potential_img = 255 * potential_img / np.max(potential_img)
            potential_img = cv2.applyColorMap(np.uint8(potential_img), cv2.COLORMAP_JET)
    elif args.env_name == 'RopeConfiguration':
        vis_critic = (-vis_critic)
        vis_critic = np.exp(vis_critic) / np.sum(np.exp(vis_critic))
        vis_critic = vis_critic - np.min(vis_critic)
        vis_critic = 255 * vis_critic / np.max(vis_critic)
        vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

        vis_gt = np.exp(gt_img) / np.sum(np.exp(gt_img))
        vis_gt = vis_gt - np.min(vis_gt)
        vis_gt = 255 * vis_gt / np.max(vis_gt)
        vis_gt = cv2.applyColorMap(np.uint8(vis_gt), cv2.COLORMAP_JET)

    for u in range(max(0, p1_pixel[0] - 2), min(args.image_size, p1_pixel[0] + 2)):
        for v in range(max(0, p1_pixel[1] - 2), min(args.image_size, p1_pixel[1] + 2)):
            obs_img[u][v] = (255, 255, 255)

    vis_img = np.concatenate((cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB), vis_gt, vis_critic), axis=1)
    if aff:
        vis_img = np.concatenate((cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB), vis_gt, vis_critic, potential_img), axis=1)

    cv2.imwrite(f'./visual/{args.exp_name}-gt_critic-{p0_pixel[0]}-{p0_pixel[1]}.jpg', vis_img)
    print("save to" + f'./visual/{args.exp_name}-gt_critic-{p0_pixel[0]}-{p0_pixel[1]}.jpg')


def run_jobs(process_id, args, env_kwargs):
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
                                     load_critic_mean_std_dir=args.load_critic_mean_std_dir,
                                     load_aff_mean_std_dir=args.load_aff_mean_std_dir,
                                     out_logits=args.out_logits,
                                     without_global=args.without_global,
                                     expert_pick=args.expert_pick,
                                     critic_pick=args.critic_pick,
                                     random_pick=args.random_pick,
                                     critic_depth=args.critic_depth,
                                     only_depth=args.only_depth
                                     )

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    test_id = 0
    full_covered_area = None
    while (test_id < args.num_test):
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
            full_distance = env.compute_reward()
        pyflex.step()

        step_i = 0

        while step_i < args.step:
            # print("step_i: ", step_i)
            if args.env_name == 'ClothFlatten':
                prev_obs, prev_depth = pyflex.render_cloth()
            elif args.env_name == 'RopeConfiguration':
                prev_obs, prev_depth = pyflex.render()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]
            prev_depth = prev_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            # print(np.min(prev_depth), np.max(prev_depth))
            mask = np.where(prev_depth[:, :, 0] < 0.295, 255, 0)
            # print(mask.shape)
            # cv2.imwrite(f'./visual/test-mask-{step_i}-depth.jpg', mask)

            # crumple the cloth by grabbing corner
            if args.env_name == 'ClothFlatten':
                # if step_i == 0:
                mask = prev_obs[:, :, 0]
                # cv2.imwrite(f'./visual/test-mask-{step_i}-cloth.jpg', mask)
                indexs = np.transpose(np.where(mask != 0))
                corner_id = random.randint(0, 3)
                # print(corner_id)
                top, left = indexs.min(axis=0)
                bottom, right = indexs.max(axis=0)

                corners = [[top, left],
                           [top, right],
                           [bottom, right],
                           [bottom, left]]
                u1 = (corners[corner_id][1]) * 2.0 / 720 - 1
                v1 = (corners[corner_id][0]) * 2.0 / 720 - 1
                # else:
                #     indexs = np.transpose(np.nonzero(prev_obs[:, :, 0]))
                #     index = random.choice(indexs)
                #     u1 = (index[1]) * 2.0 / 720 - 1
                #     v1 = (index[0]) * 2.0 / 720 - 1

                u2 = random.uniform(-1., 1.)
                v2 = random.uniform(-1., 1.)

                action = np.array([u1, v1, u2, v2])

            elif args.env_name == 'RopeConfiguration':
                indexs = np.transpose(np.nonzero(mask))
                index = random.choice(indexs)
                u1 = (index[1]) * 2.0 / 720 - 1
                v1 = (index[0]) * 2.0 / 720 - 1

                # bound = (step_i + 1) * 0.2
                # u2 = random.uniform(-bound, bound)
                # v2 = random.uniform(-bound, bound)
                u2 = max(min(np.random.normal(u1, scale=0.4), 0.999), -1.)
                v2 = max(min(np.random.normal(v1, scale=0.4), 0.999), -1.)
                # u2 = random.uniform(-1., 1.)
                # v2 = random.uniform(-1., 1.)

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

        env.start_record()

        if args.env_name == 'ClothFlatten':
            crump_area = env._get_current_covered_area(pyflex.get_positions())
            crump_percent = crump_area / full_covered_area
            if crump_percent >= 0.8:
                continue
            print("crump percent: ", crump_percent)
        elif args.env_name == 'RopeConfiguration':
            crump_distance = env.compute_reward()
            if crump_distance >= -0.06:
                continue
            print("crump distance: ", crump_distance)

        if args.env_name == 'ClothFlatten':
            crump_obs, crump_depth = pyflex.render_cloth()
        elif args.env_name == 'RopeConfiguration':
            env.action_tool.hide()
            crump_obs, crump_depth = pyflex.render()

        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        # show_obs(crump_obs)
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        crump_obs = cv2.resize(crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        state_crump = env.get_state()

        if args.expert_pick or args.critic_pick:
            reverse_p0_pixel = (int((action[3] + 1.) / 2 * args.image_size), int((action[2] + 1.) / 2 * args.image_size))
            action = agent.act(crump_obs.copy(), p0=reverse_p0_pixel)
        else:
            action = agent.act(crump_obs.copy())

        reverse_p0 = (action[0], action[1])

        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)

        if args.env_name == 'ClothFlatten':
            curr_area = env._get_current_covered_area(pyflex.get_positions())
            curr_percent = curr_area / full_covered_area
            print("curr percent: ", curr_percent)
        elif args.env_name == 'RopeConfiguration':
            curr_distance = env.compute_reward()
            print("curr distance: ", curr_distance)

        if args.step == 1:
            if args.env_name == 'ClothFlatten':
                if curr_percent >= 0.75:
                    if curr_percent - crump_percent >= 0.05:
                        result = 'success'
                    else:
                        result = 'mid'
                else:
                    result = 'fail'
            elif args.env_name == 'RopeConfiguration':
                if curr_distance >= -0.06:
                    if curr_distance - crump_distance >= 0.01:
                        result = 'success'
                    else:
                        result = 'mid'
                else:
                    result = 'fail'

        else:
            if args.env_name == 'ClothFlatten':
                if curr_percent >= 0.50:
                    if curr_percent - crump_percent >= 0.05:
                        result = 'success'
                    else:
                        result = 'mid'
                else:
                    result = 'fail'
            elif args.env_name == 'RopeConfiguration':
                if curr_distance >= -0.06:
                    if curr_distance - crump_distance >= 0.01:
                        result = 'success'
                    else:
                        result = 'mid'
                else:
                    result = 'fail'
        if args.env_name == 'RopeConfiguration':
            env.action_tool.hide()
        if args.save_video_dir is not None:
            path_name = os.path.join(args.save_video_dir, name + args.exp_name)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            save_name = os.path.join(path_name, f'{process_id}-{test_id}-{result}.gif')
            save_numpy_as_gif(np.array(env.video_frames), save_name)
            print('Video generated and save to {}'.format(save_name))

        env.end_record()
        test_id += 1

        visualize_critic_gt(crump_obs.copy(), env, agent, reverse_p0, full_covered_area, args, state_crump)
        visualize_aff_state(crump_obs.copy(), env, agent, full_covered_area, args, state_crump)
        # visualize_aff_critic(crump_obs.copy(), agent, args)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--image_size', type=int, default=320, help='Size of input observation')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--shape', type=str, default='S')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--critic_depth', default=1, type=int)
    parser.add_argument('--num_test', type=int, default=1, help='How many test do you need for inferring')
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--load_critic_mean_std_dir', default='xxx')
    parser.add_argument('--load_aff_mean_std_dir', default='xxx')
    parser.add_argument('--without_global', action='store_true')
    parser.add_argument('--expert_pick',    action='store_true')
    parser.add_argument('--critic_pick',    action='store_true')
    parser.add_argument('--random_pick',    action='store_true')
    parser.add_argument('--only_depth', action='store_true')
    parser.add_argument('--exp', action='store_true')
    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['goal_shape'] = args.shape

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')

    process_list = []
    for process_id in range(0, args.process_num):
        p = multiprocessing.Process(target=run_jobs, args=(process_id, args, env_kwargs))
        p.start()
        print(f'process {process_id} begin')
        process_list.append(p)

    for p in process_list:
        p.join()
        print(f'process end')


if __name__ == '__main__':
    main()
