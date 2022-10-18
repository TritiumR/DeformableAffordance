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


def visualize_aff_critic(obs, agent, args, iter):
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

    img_critic = obs.copy()
    critic = agent.critic_model.forward(img_critic, p0_pixel)

    if args.env_name == 'ClothFlatten':
        argmax = np.argmax(critic)
        argmax = np.unravel_index(argmax, shape=critic.shape)

    elif args.env_name == 'RopeConfiguration':
        argmax = np.argmin(critic)
        argmax = np.unravel_index(argmax, shape=critic.shape)

    p1_pixel = argmax[1:3]

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

    cv2.imwrite(f'./visual/{args.exp_name}-aff_critic-{iter}-{state_score}.jpg', vis_img)
    print("save to" + f'./visual/{args.exp_name}-aff_critic-{iter}-{state_score}.jpg')


def run_jobs(process_id, args, env_kwargs):
    # Set the beginning of the agent name.
    name = f'{args.task}-Aff_Critic-{args.num_online}-{args.exp_name}'

    # Initialize agent and limit random dataset sampling to fixed set.
    tf.random.set_seed(process_id)
    agent = agents.names[args.agent](name,
                                     args.task,
                                     image_size=args.image_size,
                                     use_goal_image=args.use_goal_image,
                                     load_critic_dir=args.load_critic_dir,
                                     load_aff_dir=args.load_aff_dir,
                                     load_critic_mean_std_dir=args.load_critic_mean_std_dir,
                                     load_aff_mean_std_dir=args.load_aff_mean_std_dir,
                                     out_logits=args.out_logits,
                                     learning_rate=args.learning_rate,
                                     without_global=args.without_global,
                                     critic_depth=args.critic_depth,
                                     only_depth=args.only_depth
                                     )

    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    if args.model == 'aff' or args.model == 'both':
        agent.save_mean_std("aff")
    if args.model == 'critic' or args.model == 'both':
        agent.save_mean_std("critic")

    online_id = 0
    full_covered_area = None
    while (online_id < args.num_online):
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

        # online crumple online crumple online crumple online crumple
        step_i = 0
        while step_i < args.step:
            if args.env_name == 'ClothFlatten':
                prev_obs, prev_depth = pyflex.render_cloth()
            elif args.env_name == 'RopeConfiguration':
                env.action_tool.hide()
                prev_obs, prev_depth = pyflex.render()
            prev_obs = prev_obs.reshape((720, 720, 4))[::-1, :, :3]
            prev_depth = prev_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            mask = np.where(prev_depth[:, :, 0] < 0.295, 255, 0)
            # cv2.imwrite(f'./visual/test-mask-{step_i}-depth.jpg', mask)

            # crumple the cloth by grabbing corner
            if args.env_name == 'ClothFlatten':
                mask = prev_obs[:, :, 0]
                # cv2.imwrite(f'./visual/test-mask-{step_i}-cloth.jpg', mask)
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

            step_i += 1

        # online manipulate online manipulate online manipulate online manipulate

        if args.env_name == 'ClothFlatten':
            crump_area = env._get_current_covered_area(pyflex.get_positions())
            crump_percent = crump_area / full_covered_area
            if crump_percent >= (0.8 - args.step * 0.05):
                continue
            # print("crump percent: ", crump_percent)
        elif args.env_name == 'RopeConfiguration':
            crump_distance = -env.compute_reward()
            if crump_distance <= 0.06:
                continue
            # print("crump distance: ", crump_distance)

        state_crump = env.get_state()

        if args.env_name == 'ClothFlatten':
            max_recover = float('-inf')
        elif args.env_name == 'RopeConfiguration':
            min_distance = float('inf')

        curr_data = []
        action_data = []
        metric_data = []

        for id in range(args.data_type):
            env.set_state(state_crump)

            # step-1 expert
            if id == 0:
                # take reverse action
                reverse_action = np.array([action[2], action[3], action[0], action[1]])
                _, _, _, info = env.step(reverse_action, record_continuous_video=False, img_size=args.img_size)
                if env.action_tool.not_on_cloth:
                    print('reverse not on cloth')

            else:
                # step-1 random action
                random_action = env.action_space.sample()
                random_action[0] = reverse_action[0]
                random_action[1] = reverse_action[1]
                _, _, _, info = env.step(random_action, record_continuous_video=False, img_size=args.img_size)

            if args.env_name == 'ClothFlatten':
                reserse_area = env._get_current_covered_area(pyflex.get_positions())
                reverse_percent = reserse_area / full_covered_area
                # print("reverse percent: ", reverse_percent)
            elif args.env_name == 'RopeConfiguration':
                reverse_distance = -env.compute_reward()
                # print("reverse distance: ", reverse_distance)

            if args.env_name == 'ClothFlatten':
                curr_obs, curr_depth = pyflex.render_cloth()
            elif args.env_name == 'RopeConfiguration':
                env.action_tool.hide()
                curr_obs, curr_depth = pyflex.render()

            curr_obs = curr_obs.reshape((720, 720, 4))[::-1, :, :3]
            curr_depth[curr_depth > 5] = 0
            curr_depth = curr_depth.reshape((720, 720))[::-1].reshape(720, 720, 1)
            curr_obs = np.concatenate([curr_obs, curr_depth], 2)
            curr_obs = cv2.resize(curr_obs, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(f'./visual/test-online-curr-{reverse_percent}.jpg', curr_obs)
            curr_data.append(curr_obs.copy())

            # step-2 manipulation
            state_curr = env.get_state()

            for critic_id in range(args.critic_type):
                env.set_state(state_curr)
                agent_action = agent.act(curr_obs.copy())

                if critic_id != 0:
                    agent_action[2] = random.uniform(-1., 1.)
                    agent_action[3] = random.uniform(-1., 1.)

                action_data.append(agent_action.copy())

                _, _, _, info = env.step(agent_action, record_continuous_video=False, img_size=args.img_size)

                if args.env_name == 'ClothFlatten':
                    final_area = env._get_current_covered_area(pyflex.get_positions())
                    final_percent = final_area / full_covered_area
                    # print(f'final percent {critic_id}: ', final_percent)
                    if final_percent > max_recover:
                        max_recover = final_percent
                    metric_data.append(final_percent)
                elif args.env_name == 'RopeConfiguration':
                    final_distance = -env.compute_reward()
                    # print(f'final distance {critic_id}: ', final_distance)
                    if final_distance < min_distance:
                        min_distance = final_distance
                    metric_data.append(final_distance)

        if args.env_name == 'ClothFlatten':
            if max_recover <= 1.0 - (args.step * 0.1):
                continue
        elif args.env_name == 'RopeConfiguration':
            if min_distance >= 0.6:
                continue

        # train aff with online data
        if args.model == 'aff' or args.model == 'both':
            with tf.GradientTape() as tape:
                loss = None
                if agent.only_depth:
                    aff_pred = agent.attention_model.forward_batch(np.array(curr_data)[:, :, :, -1:].copy(), apply_softmax=False)
                else:
                    aff_pred = agent.attention_model.forward_batch(curr_data.copy(), apply_softmax=False)
                for bh in range(args.data_type):
                    base = bh * args.critic_type
                    p0 = [min(args.image_size - 1, int((action_data[base][1] + 1.) * 0.5 * args.image_size)),
                          min(args.image_size - 1, int((action_data[base][0] + 1.) * 0.5 * args.image_size))]
                    output = aff_pred[bh, p0[0], p0[1], :]
                    gt = metric_data[base] * 50
                    print("aff output: ", output, "gt: ", gt)
                    if loss is None:
                        loss = tf.keras.losses.MAE(gt, output)
                    else:
                        loss = loss + tf.keras.losses.MAE(gt, output)
                loss = tf.reduce_mean(loss)
                loss = loss / args.data_type
            grad = tape.gradient(loss, agent.attention_model.model.trainable_variables)
            agent.attention_model.optim.apply_gradients(zip(grad, agent.attention_model.model.trainable_variables))
            agent.attention_model.metric(loss)
            print(f"aff iter: {online_id} loss: {loss}")

            if online_id % 500 == 0:
                agent.save_aff_with_epoch(online_id)

        # train critic with online data
        if args.model == 'critic' or args.model == 'both':
            with tf.GradientTape() as tape:
                loss_critic = None

                for bh in range(args.data_type):
                    base = bh * args.critic_type
                    p0 = [min(args.image_size - 1, int((action_data[base][1] + 1.) * 0.5 * args.image_size)),
                          min(args.image_size - 1, int((action_data[base][0] + 1.) * 0.5 * args.image_size))]
                    QQ_cur = agent.critic_model.forward(curr_data[bh], p0)

                    for p1_id in range(args.critic_type):
                        p1 = [min(args.image_size - 1, int((action_data[base + p1_id][3] + 1.) * 0.5 * args.image_size)),
                              min(args.image_size - 1, int((action_data[base + p1_id][2] + 1.) * 0.5 * args.image_size))]
                        output = QQ_cur[0, p1[0], p1[1], :]
                        gt = metric_data[base + p1_id] * 50
                        print("critic output: ", output, "gt: ", gt)
                        if loss_critic is None:
                            loss_critic = tf.keras.losses.MAE(gt, output) / args.critic_type
                        else:
                            loss_critic = loss_critic + tf.keras.losses.MAE(gt, output) / args.critic_type
                loss_critic = tf.reduce_mean(loss_critic)
                loss_critic /= args.data_type

            grad = tape.gradient(loss_critic, agent.critic_model.model.trainable_variables)
            agent.critic_model.optim.apply_gradients(zip(grad, agent.critic_model.model.trainable_variables))
            agent.critic_model.metric(loss_critic)
            print(f"critic iter: {online_id} loss: {loss_critic}")

            if online_id % 500 == 0:
                agent.save_critic_with_epoch(online_id)

        if online_id % 100 == 0:
            visualize_aff_critic(curr_data[0].copy(), agent, args, online_id)
        online_id += 1


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--image_size', type=int, default=320, help='Size of input observation')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--task', type=str, default='cloth-flatten')
    parser.add_argument('--shape', type=str, default='S')
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--agent', default='aff_critic')
    parser.add_argument('--critic_depth', default=1, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--num_online', type=int, default=1, help='How many test do you need for inferring')
    parser.add_argument('--data_type', type=int, default=1, help='How many children for one crumple state')
    parser.add_argument('--critic_type', type=int, default=1, help='How many children for one current state')
    parser.add_argument('--process_num', type=int, default=1, help='How many process do you need')
    parser.add_argument('--use_goal_image',       default=0, type=int)
    parser.add_argument('--out_logits',     default=1, type=int)
    parser.add_argument('--exp_name', type=str, default='0809-01')
    parser.add_argument('--load_critic_dir',       default='xxx')
    parser.add_argument('--load_aff_dir',       default='xxx')
    parser.add_argument('--load_critic_mean_std_dir', default='xxx')
    parser.add_argument('--load_aff_mean_std_dir', default='xxx')
    parser.add_argument('--model', default='aff', type=str)
    parser.add_argument('--without_global', action='store_true')
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
