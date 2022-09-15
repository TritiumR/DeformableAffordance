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


def visualize_aff_critic(obs, agent):
    img_aff = obs.copy()
    attention = agent.attention_model.forward(img_aff)

    # depth = obs[:, :, -1:].copy()
    # mask = np.where(depth == 0, 0, 1)
    # state_map = np.where(depth == 0, 0, attention)
    state_score = int(np.max(attention) * 2)
    attention = attention - np.min(attention)
    # attention = attention * mask

    # state_score = int(np.max(state_map) * 2)
    argmax = np.argmax(attention)
    argmax = np.unravel_index(argmax, shape=attention.shape)

    p0_pixel = argmax[1:3]
    print(p0_pixel)

    img_critic = obs.copy()
    critic = agent.critic_model.forward(img_critic, p0_pixel)

    argmax = np.argmax(critic)
    argmax = np.unravel_index(argmax, shape=critic.shape)

    p1_pixel = argmax[1:3]
    print(p1_pixel)

    vis_aff = np.array(attention[0])
    vis_critic = np.array(critic[0])

    vis_aff = vis_aff - np.min(vis_aff)
    # vis_aff = vis_aff * mask
    vis_aff = 255 * vis_aff / np.max(vis_aff)
    vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

    vis_critic = vis_critic - np.min(vis_critic)
    vis_critic = 255 * vis_critic / np.max(vis_critic)
    vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

    img_obs = obs[:, :, :3]

    for u in range(max(0, p0_pixel[0] - 4), min(320, p0_pixel[0] + 4)):
        for v in range(max(0, p0_pixel[1] - 4), min(320, p0_pixel[1] + 4)):
            img_obs[u][v] = (255, 0, 0)

    for u in range(max(0, p1_pixel[0] - 4), min(320, p1_pixel[0] + 4)):
        for v in range(max(0, p1_pixel[1] - 4), min(320, p1_pixel[1] + 4)):
            img_obs[u][v] = (255, 255, 255)

    vis_img = np.concatenate((cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB), vis_aff, vis_critic), axis=1)

    cv2.imwrite(f'./visual/fifth-aff_critic-{p0_pixel[0]}-{p0_pixel[1]}-{state_score}.jpg', vis_img)
    print("save to" + f'./visual/fifth-aff_critic-{p0_pixel[0]}-{p0_pixel[1]}-{state_score}.jpg')


def visualize_aff_state(obs, env, agent, full_covered_area, args, state_crump):
    vis_aff = np.zeros((16, 16))
    gt_aff = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            env.set_state(state_crump)
            p0 = (i * 20, j * 20)
            action = agent.act(obs.copy(), p0=p0)
            _, _, _, _ = env.step(action, record_continuous_video=False, img_size=args.img_size)
            gt_area = env._get_current_covered_area(pyflex.get_positions())
            gt_percent = gt_area / full_covered_area
            gt_aff[i][j] = gt_percent
            output = agent.critic_model.forward(obs.copy(), p0)
            critic_score = output[:, :, :, 0]
            vis_aff[i][j] = np.max(critic_score)

    score = int(np.max(vis_aff) * 2)
    gt_score = int(np.max(gt_aff) * 100)

    vis_aff = cv2.resize(vis_aff, (320, 320))
    gt_aff = cv2.resize(gt_aff, (320, 320))

    vis_aff = vis_aff - np.min(vis_aff)
    vis_aff = 255 * vis_aff / np.max(vis_aff)
    vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

    gt_aff = gt_aff - np.min(gt_aff)
    gt_aff = 255 * gt_aff / np.max(gt_aff)
    gt_aff = cv2.applyColorMap(np.uint8(gt_aff), cv2.COLORMAP_JET)

    vis_img = np.concatenate((cv2.cvtColor(obs, cv2.COLOR_BGR2RGB), gt_aff, vis_aff), axis=1)

    cv2.imwrite(f'./visual/9_9-7500-aff_max-{gt_score}-{score}.jpg', vis_img)
    print("save to" + f'./visual/9_9-7500-aff_max-{gt_score}-{score}.jpg')


def visualize_critic_gt(obs, env, agent, p0, full_covered_area, args, state_crump):
    obs_img = obs[:, :, :3].copy()
    p0_pixel = (int((p0[1] + 1.) * 160), int((p0[0] + 1.) * 160))

    for u in range(max(0, p0_pixel[0] - 2), min(320, p0_pixel[0] + 2)):
        for v in range(max(0, p0_pixel[1] - 2), min(320, p0_pixel[1] + 2)):
            obs_img[u][v] = (255, 0, 0)

    output = agent.critic_model.forward(obs.copy(), p0_pixel)
    critic_score = output[:, :, :, 0]
    argmax = np.argmax(critic_score)
    argmax = np.unravel_index(argmax, shape=critic_score.shape)
    p1_pixel = argmax[1:3]

    vis_critic = np.float32(critic_score[0])
    vis_critic = vis_critic - np.min(vis_critic)
    vis_critic = 255 * vis_critic / np.max(vis_critic)
    vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)

    gt_img = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            env.set_state(state_crump)
            p1 = ((i - 8) / 8, (j - 8) / 8)
            action = np.array([p0[0], p0[1], p1[1], p1[0]])
            _, _, _, _ = env.step(action, record_continuous_video=False, img_size=args.img_size)
            gt_area = env._get_current_covered_area(pyflex.get_positions())
            gt_percent = gt_area / full_covered_area
            gt_img[i][j] = gt_percent

    gt_img = cv2.resize(gt_img, (320, 320))

    vis_gt = gt_img - np.min(gt_img)
    vis_gt = 255 * vis_gt / np.max(vis_gt)
    vis_gt = cv2.applyColorMap(np.uint8(vis_gt), cv2.COLORMAP_JET)

    for u in range(max(0, p1_pixel[0] - 2), min(320, p1_pixel[0] + 2)):
        for v in range(max(0, p1_pixel[1] - 2), min(320, p1_pixel[1] + 2)):
            obs_img[u][v] = (255, 255, 255)

    vis_img = np.concatenate((cv2.cvtColor(obs_img, cv2.COLOR_BGR2RGB), vis_gt, vis_critic), axis=1)

    cv2.imwrite(f'./visual/9_9-7500-gt_critic-{p0_pixel[0]}-{p0_pixel[1]}.jpg', vis_img)
    print("save to" + f'./visual/9_9-7500-gt_critic-{p0_pixel[0]}-{p0_pixel[1]}.jpg')


def run_jobs(process_id, args, env_kwargs):
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
                                     without_global=args.without_global,
                                     expert_pick=args.expert_pick,
                                     critic_pick=args.critic_pick,
                                     random_pick=args.random_pick,
                                     )

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    test_id = 0

    while (test_id < args.num_test):
        # from flat configuration
        full_covered_area = env._set_to_flatten()
        pyflex.step()

        prev_obs, _ = pyflex.render_cloth()
        prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]

        # crumple the cloth by grabbing corner
        mask = prev_obs[10:, :, 0]
        indexs = np.transpose(np.where(mask == 255))
        corner_id = random.randint(0, 3)
        # print(corner_id)
        top, left = indexs.min(axis=0)
        bottom, right = indexs.max(axis=0)

        corners = [[top + 30, left + 20],
                   [top + 30, right - 20],
                   [bottom + (-10), right - 20],
                   [bottom + (-10), left + 20]]
        u1 = (corners[corner_id][1]) * 2.0 / env.camera_height - 1
        v1 = (corners[corner_id][0]) * 2.0 / env.camera_height - 1
        action = env.action_space.sample()
        action[0] = u1
        action[1] = v1

        _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)
        crump_area = env._get_current_covered_area(pyflex.get_positions())
        crump_percent = crump_area / full_covered_area
        if crump_percent >= 0.8:
            continue
        env.start_record()
        print("crump percent: ", crump_percent)
        crump_obs, crump_depth = pyflex.render_cloth()
        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        crump_obs = cv2.resize(crump_obs, (320, 320), interpolation=cv2.INTER_AREA)

        state_crump = env.get_state()

        reverse_p0 = (action[2], action[3])

        if args.expert_pick or args.critic_pick:
            reverse_p0_pixel = (int((action[3] + 1.) * 160), int((action[2] + 1.) * 160))
            action = agent.act(crump_obs.copy(), p0=reverse_p0_pixel)
        else:
            action = agent.act(crump_obs.copy())

        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        covered_area = env._get_current_covered_area(pyflex.get_positions())
        covered_percent = covered_area / full_covered_area
        print("curr percent: ", covered_percent)

        if covered_percent >= 0.75:
            if covered_percent - crump_percent >= 0.05:
                result = 'success'
            else:
                result = 'mid'
        else:
            result = 'fail'

        if args.save_video_dir is not None:
            path_name = os.path.join(args.save_video_dir, name + args.exp_name)
            if not os.path.exists(path_name):
                os.makedirs(path_name)
            save_name = os.path.join(path_name, f'{process_id}-{test_id}-{result}.gif')
            save_numpy_as_gif(np.array(env.video_frames), save_name)
            print('Video generated and save to {}'.format(save_name))

        env.end_record()
        test_id += 1

        # visualize_critic_gt(crump_obs.copy(), env, agent, reverse_p0, full_covered_area, args, state_crump)
        # visualize_aff_state(crump_obs.copy(), env, agent, full_covered_area, args, state_crump)
        visualize_aff_critic(crump_obs.copy(), agent)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--num_demos', type=int, default=1, help='How many data do you need for training')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--num_test', type=int, default=1, help='How many test do you need for inferring')
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--without_global', action='store_true')
    parser.add_argument('--expert_pick',    action='store_true')
    parser.add_argument('--critic_pick',    action='store_true')
    parser.add_argument('--random_pick',    action='store_true')
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