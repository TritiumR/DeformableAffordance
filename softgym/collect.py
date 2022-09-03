import os
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


def dump(path, data, process, curr_num, data_id, data_num, step):
    fname = f'{curr_num + data_id + process * data_num:06d}-{step}.pkl'
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(data, open(os.path.join(path, fname), 'wb'))
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
        state_flat = env.get_state()
        pyflex.step()

        for step_i in range(args.step):
            prev_obs, prev_depth = pyflex.render_cloth()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]

            # crumple the cloth by grabbing corner
            if step_i == 1:
                mask = prev_obs[10:, :, 0]
                indexs = np.transpose(np.where(mask == 255))
                corner_id = random.randint(0, 3)
                # print(corner_id)
                top, left = indexs.min(axis=0)
                bottom, right = indexs.max(axis=0)

                corners = [[top + 10, left],
                           [top + 10, right],
                           [bottom + 10, right],
                           [bottom + 10, left]]
                u1 = (corners[corner_id][1]) * 2.0 / env.camera_height - 1
                v1 = (corners[corner_id][0]) * 2.0 / env.camera_height - 1
            else:
                indexs = np.transpose(np.nonzero(prev_obs[:, :, 0]))
                index = random.choice(indexs)
                u1 = (index[1]) * 2.0 / env.camera_height - 1
                v1 = (index[0]) * 2.0 / env.camera_height - 1
            action = env.action_space.sample()
            action[0] = u1
            action[1] = v1

            _, _, _, info = env.step(action, record_continuous_video=False, img_size=args.img_size)

        crump_area = env._get_current_covered_area(pyflex.get_positions())
        crump_percent = crump_area / full_covered_area
        # print("percent: ", crump_percent)
        crump_obs, crump_depth = pyflex.render_cloth()
        crump_obs = crump_obs.reshape((720, 720, 4))[::-1, :, :3]
        crump_depth[crump_depth > 5] = 0
        crump_depth = crump_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
        crump_obs = np.concatenate([crump_obs, crump_depth], 2)
        crump_obs = cv2.resize(crump_obs, (320, 320), interpolation=cv2.INTER_AREA)

        state_crump = env.get_state()

        action_data = []
        area_data = []
        # curr_data = []
        not_on_cloth_data = []
        max_recover = 0

        another_pick = random.randint(0, 25)
        if another_pick == 0:
            another_action = env.action_space.sample()
            action[2] = another_action[0]
            action[3] = another_action[1]
            print('another_pick')

        # p0 = [int((action[3] + 1.) * 160), int((action[2] + 1.) * 160)]
        # pick_area = crump_obs[max(0, p0[0] - 4): min(320, p0[0] + 4),
        #                       max(0, p0[1] - 4): min(320, p0[1] + 4),
        #                       :3].copy()
        # if np.sum(pick_area) == 0 and another_pick != 0:
        #     print('not on cloth')
        #     img, _ = pyflex.render()
        #     img = img.reshape((720, 720, 4))[::-1, :, :3]
        #     img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        #     for u in range(max(0, p0[0] - 4), min(320, p0[0] + 4)):
        #         for v in range(max(0, p0[1] - 4), min(320, p0[1] + 4)):
        #             img[u][v] = (255, 0, 0)
        #     cv2.imwrite(f'./visual/not-on-cloth-{p0[0]}-{p0[1]}.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     print("save to" + f'./visual/not-on-cloth-{p0[0]}-{p0[1]}.jpg')
        #     continue

        for id in range(args.data_type):
            env.set_state(state_crump)
            # take reverse action
            if id == 0:
                reverse_action = np.array([action[2], action[3], action[0], action[1]])
                action_data.append(reverse_action.copy())
                _, _, _, info = env.step(reverse_action, record_continuous_video=False, img_size=args.img_size)
                covered_area = env._get_current_covered_area(pyflex.get_positions())
                covered_percent = covered_area / full_covered_area
                if env.action_tool.not_on_cloth:
                    not_on_cloth_data.append(1)
                    print('not on cloth')
                else:
                    not_on_cloth_data.append(0)
                    # p0 = [int((reverse_action[1] + 1.) * 160), int((reverse_action[0] + 1.) * 160)]
                    # img, _ = pyflex.render()
                    # img = img.reshape((720, 720, 4))[::-1, :, :3]
                    # img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
                    # for u in range(max(0, p0[0] - 4), min(320, p0[0] + 4)):
                    #     for v in range(max(0, p0[1] - 4), min(320, p0[1] + 4)):
                    #         img[u][v] = (255, 0, 0)
                    # cv2.imwrite(f'./visual/not-on-cloth-{p0[0]}-{p0[1]}.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    # print("save to" + f'./visual/not-on-cloth-{p0[0]}-{p0[1]}.jpg')
                area_data.append([crump_percent, covered_percent])

            # take random action
            else:
                random_action = env.action_space.sample()
                random_action[0] = reverse_action[0]
                random_action[1] = reverse_action[1]
                action_data.append(random_action.copy())
                _, _, _, info = env.step(random_action, record_continuous_video=False, img_size=args.img_size)
                covered_area = env._get_current_covered_area(pyflex.get_positions())
                covered_percent = covered_area / full_covered_area
                area_data.append([crump_percent, covered_percent])

            # curr_obs, curr_depth = pyflex.render_cloth()
            # curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
            # curr_depth[curr_depth > 5] = 0
            # curr_depth = curr_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            # curr_obs = np.concatenate([curr_obs, curr_depth], 2)
            # curr_obs = cv2.resize(curr_obs, (320, 320), interpolation=cv2.INTER_AREA)
            # curr_data.append(curr_obs)
            # # show_obs(curr_depth)

            # print(id, covered_percent)

            if covered_percent > max_recover:
                max_recover = covered_percent

        print("percent: ", max_recover, crump_percent)
        if args.step == 1:
            if_save = (max_recover >= 0.8 and max_recover - 0.05 >= crump_percent) or another_pick == 0
        else:
            if_save = max_recover - 0.1 >= crump_percent
        if if_save:
            data = {}
            data['obs'] = np.array(crump_obs).copy()
            # data['curr'] = np.array(curr_data).copy()
            assert data['obs'].shape == (320, 320, 4)
            data['area'] = area_data
            data['action'] = action_data
            data['not_on_cloth'] = not_on_cloth_data
            dump(args.path, data, process_id, args.curr_data, data_id, args.data_num, 1)
            data_id += 1

        env.set_state(state_flat)

    if args.save_video_dir is not None:
        save_name = os.path.join(args.save_video_dir, args.env_name + f'{process_id}.gif')
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
    parser.add_argument('--path', type=str, default='./data/')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default=None, help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--data_num', type=int, default=1, help='How many data do you need for each process')
    parser.add_argument('--curr_data', type=int, default=0, help='How many data have existed')
    parser.add_argument('--data_type', type=int, default=1, help='What kind of data')
    parser.add_argument('--step', type=int, default=1, help='How many steps from goal')

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
