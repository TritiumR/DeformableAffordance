import os
import argparse
import numpy as np
import cv2

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils.pyflex_utils import center_object

import agents
from models import Critic_MLP

import pyflex
from matplotlib import pyplot as plt
import tensorflow as tf

import multiprocessing
import random
import pickle

def run_jobs(process_id, args, env_kwargs):
    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic'

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
                                     use_mask=args.use_mask
                                     )

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    test_id = 0
    full_covered_area = None
    while (test_id < args.num_test):
        # from flat configuration
        full_covered_area = env._set_to_flatten()
        pyflex.step()

        step_i = 0
        while step_i < args.step:
            prev_obs, _ = pyflex.render_cloth()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]

            # crumple the cloth by grabbing corner
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

            _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)

            center_object()

            if env.action_tool.not_on_cloth:
                print(f'{step_i} not on cloth')
                # reset to flat configuration
                full_covered_area = env._set_to_flatten()
                pyflex.step()

                step_i = 0
                continue
            step_i += 1

        env.start_record()

        crump_area = env._get_current_covered_area(pyflex.get_positions())
        crump_percent = crump_area / full_covered_area
        if crump_percent >= 0.8:
            continue
        print("crump percent: ", crump_percent)

        vis_img = []
        curr_score = ''
        for step_id in range(args.test_step):
            # rener obs
            env.action_tool.hide()
            crump_obs, crump_depth = pyflex.render_cloth()
            render_obs, _ = pyflex.render()
            crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
            crump_depth[crump_depth > 5] = 0
            crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            crump_obs = np.concatenate([crump_obs, crump_depth], 2)
            crump_obs = cv2.resize(crump_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            render_obs = render_obs.reshape((720, 720, 4))[::-1, :, :3]
            render_obs = cv2.resize(render_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

            crump_mask = crump_obs[:, :, 0]
            crump_indexs = np.transpose(np.where(crump_mask != 0))
            state_crump = env.get_state()

            # render aff
            attention = agent.attention_model.forward(crump_obs.copy())
            state_score = int(np.max(attention) * 20) / 10
            attention = attention - np.min(attention)
            if args.use_mask:
                aff_mask = np.where(crump_obs[:, :, :-1] == (0, 0, 0), 0, 1)
                attention *= aff_mask
            aff_argmax = np.argmax(attention)
            aff_argmax = np.unravel_index(aff_argmax, shape=attention.shape)
            p0_pixel = aff_argmax[1:3]

            vis_aff = np.array(attention[0])
            vis_aff = vis_aff - np.min(vis_aff)
            vis_aff = 255 * vis_aff / np.max(vis_aff)
            vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)

            vis_img.append(cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB).copy())
            vis_img.append(vis_aff.copy())
            for i in range(args.pick_num):
                if i == args.pick_num - 1:
                    pick_pixel = p0_pixel
                else:
                    crump_index = random.choice(crump_indexs)
                    pick_pixel = crump_index
                # render critic
                critic = agent.critic_model.forward(crump_obs.copy(), pick_pixel)
                critic_argmax = np.argmax(critic)
                critic_argmax = np.unravel_index(critic_argmax, shape=critic.shape)
                p1_pixel = critic_argmax[1:3]
                vis_critic = np.array(critic[0])

                for u in range(max(0, pick_pixel[0] - (i + 1)), min(args.image_size - 1, pick_pixel[0] + (i + 1))):
                    for v in range(max(0, pick_pixel[1] - (i + 1)), min(args.image_size - 1, pick_pixel[1] + (i + 1))):
                        vis_img[step_id * (2 + args.pick_num * (1 + args.place_num))][u][v] = (0, 255, 255)

                # min_critic = np.min(vis_critic)
                # print(min_critic)
                vis_critic = vis_critic - np.min(vis_critic)
                if i == args.pick_num - 1:
                    critic_max = np.max(vis_critic)
                if args.exp:
                    vis_critic = 2 * vis_critic / np.max(vis_critic)
                    vis_critic = np.exp(vis_critic) / np.sum(np.exp(vis_critic))
                # if np.max(vis_critic) > (40 - min_critic):
                #     vis_critic = 255 * vis_critic / np.max(vis_critic)
                # else:
                #     vis_critic = 255 * vis_critic / (40 - min_critic)
                vis_critic = 255 * vis_critic / np.max(vis_critic)
                vis_critic = cv2.applyColorMap(np.uint8(vis_critic), cv2.COLORMAP_JET)
                vis_img.append(vis_critic.copy())
                for j in range(args.place_num):
                    env.set_state(state_crump)
                    if j == args.place_num - 1:
                        place_pixel = p1_pixel
                    else:
                        place_pixel = (random.randint(0, 159), random.randint(0, 159))

                    u1 = (pick_pixel[1]) * 2.0 / args.image_size - 1
                    v1 = (pick_pixel[0]) * 2.0 / args.image_size - 1
                    u2 = (place_pixel[1]) * 2.0 / args.image_size - 1
                    v2 = (place_pixel[0]) * 2.0 / args.image_size - 1
                    act = np.array([u1, v1, u2, v2])

                    _, _, _, info = env.step(act, record_continuous_video=True, img_size=args.img_size)

                    center_object()

                    curr_area = env._get_current_covered_area(pyflex.get_positions())
                    curr_percent = curr_area / full_covered_area
                    print("curr percent: ", curr_percent)
                    curr_score += f'{int(1000 * curr_percent) / 10}-'
                    # render curr_obs
                    env.action_tool.hide()
                    render_obs, _ = pyflex.render()
                    render_obs = render_obs.reshape((720, 720, 4))[::-1, :, :3]
                    render_obs = cv2.resize(render_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
                    for u in range(max(0, place_pixel[0] - 4), min(args.image_size - 1, place_pixel[0] + 4)):
                        for v in range(max(0, place_pixel[1] - 4), min(args.image_size - 1, place_pixel[1] + 4)):
                            render_obs[u][v] = (255, 255, 0)
                    vis_img.append(cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB).copy())

        vis_img = np.concatenate(vis_img, axis=1)
        cv2.imwrite(f'./visual/{args.exp_name}-test-render-{curr_score}{state_score}.jpg', vis_img)
        print("save to" + f'./visual/{args.exp_name}-test-render-{curr_score}{state_score}.jpg')
        # if args.save_video_dir is not None:
        #     path_name = os.path.join(args.save_video_dir, name + args.exp_name)
        #     if not os.path.exists(path_name):
        #         os.makedirs(path_name)
        #     save_name = os.path.join(path_name, f'{process_id}-{test_id}-{result}.gif')
        #     save_numpy_as_gif(np.array(env.video_frames), save_name)
        #     print('Video generated and save to {}'.format(save_name))
        #
        # env.end_record()
        test_id += 1


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--image_size', type=int, default=320, help='Size of input observation')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--test_step', type=int, default=1)
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--shape', type=str, default='S')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--critic_depth', default=1, type=int)
    parser.add_argument('--num_test', type=int, default=1, help='How many test do you need for inferring')
    parser.add_argument('--pick_num', type=int, default=3)
    parser.add_argument('--place_num', type=int, default=3)
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--unet', default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('--without_global', action='store_true')
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
