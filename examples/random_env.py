import os.path as osp
import argparse
import numpy as np
import cv2

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt

import multiprocessing
import random

def run_jobs(process_id, args, env_kwargs):
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    frames = [env.get_image(args.img_size, args.img_size)]
    env.start_record()
    for i in range(env.horizon):
        # from flat configuration
        full_covered_area = env._set_to_flatten()
        pyflex.step()

        for step in range(args.step):
            prev_obs, prev_depth = pyflex.render_cloth()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]
            prev_depth[prev_depth > 5] = 0
            prev_depth = prev_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)

            # show_obs(prev_obs)

            # crumple the cloth randomly
            indexs = np.transpose(np.nonzero(prev_obs[:, :, 0]))
            index = random.choice(indexs)
            u1 = (index[1]) * 2.0 / env.camera_height - 1
            v1 = (index[0]) * 2.0 / env.camera_height - 1

            # # crumple the cloth by grabbing corner
            # mask = prev_obs[10:, :, 0]
            # indexs = np.transpose(np.where(mask == 255))
            # corner_id = random.randint(0, 3)
            # top, left = indexs.min(axis=0)
            # bottom, right = indexs.max(axis=0)
            #
            # print(top, left)
            # print(bottom, right)
            #
            # corners = [[top + 11, left],
            #            [top + 11, right],
            #            [bottom + 10, right],
            #            [bottom + 10, left]]
            # u1 = (corners[corner_id][1]) * 2.0 / env.camera_height - 1
            # v1 = (corners[corner_id][0]) * 2.0 / env.camera_height - 1

            action = env.action_space.sample()
            action[0] = 0
            action[1] = 0
            print("action: ", action[0], action[1])

            action[2] = 0.9
            action[3] = -0.9

            print("action: ", action[2], action[3])

            _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
            crump_area = env._get_current_covered_area(pyflex.get_positions())
            crump_percent = crump_area / full_covered_area
            # print(f"percent-{step}: ", crump_percent)

            if args.test_depth:
                # show_obs(env._get_obs())
                show_depth()

            # reverse_action = np.array([action[2], action[3], action[0], action[1]])
            # _, _, _, info = env.step(reverse_action, record_continuous_video=True, img_size=args.img_size)
            # covered_area = env._get_current_covered_area(pyflex.get_positions())
            # covered_percent = covered_area / full_covered_area
            # print("percent222: ", covered_percent)

        # curr_obs, _ = pyflex.render_cloth()
        # curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
        # cv2.imwrite(f'./visual/obs-{i}-step-{args.step}.jpg', cv2.cvtColor(curr_obs, cv2.COLOR_BGR2RGB))
        # print("save to" + f'./visual/obs-{i}-step-{args.step}.jpg')


    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + f'{process_id}.gif')
        save_numpy_as_gif(np.array(env.video_frames), save_name)
        print('Video generated and save to {}'.format(save_name))

def show_obs(obs):
    window_name = 'obs'
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 720, 720)
    cv2.imshow(window_name, obs)  # 显示窗口的名字， 所要显示的图片
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    rgb = rgb.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    obs = np.where(rgb == 0, img, rgb)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--step', type=int, default=1, help='How many step do you need to crumple')

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
