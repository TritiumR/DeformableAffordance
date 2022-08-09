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
import pickle

def dump(data, process, curr_num, data_num, step):
    path = './data/'
    fname = f'{curr_num + process * data_num:06d}-{step}.pkl'
    pickle.dump(data, open(osp.join(path, fname), 'wb'))
    print(f'process {process} saved {fname} to {path}')

def run_jobs(process_id, args, env_kwargs):
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    frames = [env.get_image(args.img_size, args.img_size)]
    # env.start_record()
    data_id = 0

    while (data_id < args.data_num):
        # from flat configuration
        full_covered_area = env._set_to_flatten()
        state_dicts = env.get_state()
        print(f'full: {full_covered_area}')
        pyflex.step()

        prev_obs, _ = pyflex.render_cloth()
        prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]

        # crumple the cloth
        indexs = np.transpose(np.nonzero(prev_obs[:, :, 0]))
        index = random.choice(indexs)
        u1 = (index[0]) * 2.0 / env.camera_height - 1
        v1 = (index[1]) * 2.0 / env.camera_height - 1
        print(u1, v1)
        # action = np.array([0, 0, 0.5, 0.5])
        action = env.action_space.sample()
        action[0] = u1
        action[1] = v1

        _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)
        crump_area = env._get_current_covered_area(pyflex.get_positions())
        crump_percent = crump_area / full_covered_area
        print("percent: ", crump_percent)
        # print(f'crumpled: {crump_area}')
        # frames.extend(info['flex_env_recorded_frames'])
        crump_obs, crump_depth = pyflex.render_cloth()
        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        # show_obs(crump_depth)

        # take reverse action
        reverse_action = np.array([action[2], action[3], action[0], action[1]])
        _, _, _, info = env.step(reverse_action, record_continuous_video=False, img_size=args.img_size)
        covered_area = env._get_current_covered_area(pyflex.get_positions())
        # print(f'covered: {covered_area}')
        covered_percent = covered_area / full_covered_area
        print("percent: ", covered_percent)

        curr_obs, curr_depth = pyflex.render_cloth()
        curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
        curr_depth[curr_depth > 5] = 0
        curr_depth = curr_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        curr_obs = np.concatenate([curr_obs, curr_depth], 2)
        # show_obs(curr_depth)

        env.set_state(state_dicts)

        if covered_percent - 0.1 > crump_percent:
            data = {}
            data['obs'] = [crump_obs, curr_obs]
            data['area'] = [full_covered_area, crump_area, covered_area]
            data['action'] = reverse_action
            dump(data, args.curr_data + process_id, data_id, args.data_num, 1)
            data_id += 1

    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + f'{process_id}.gif')
        save_numpy_as_gif(np.array(env.video_frames), save_name)
        print('Video generated and save to {}'.format(save_name))

def show_obs(obs):
    window_name = 'obs'
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 720, 720)
    cv2.imshow(window_name, obs)
    cv2.waitKey(3000)
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
    parser.add_argument('--data_num', type=int, default=1, help='How many data do you need for each process')
    parser.add_argument('--curr_data', type=int, default=0, help='How many data have existed')

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
