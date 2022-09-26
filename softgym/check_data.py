import pickle
import argparse
import os
import random
import cv2
import numpy as np


def load(path, iepisode, step):
    field_path = path
    fname = f'{iepisode:06d}-{step}.pkl'
    return pickle.load(open(os.path.join(field_path, fname), 'rb'))

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',            default='./data/')
    parser.add_argument('--save_path',       default='./')
    parser.add_argument('--iepisode',        default=0, type=int)
    parser.add_argument('--step',            default=1, type=int)
    parser.add_argument('--render_demo',     action='store_true')

    args = parser.parse_args()

    path = args.path
    step = args.step
    i = args.iepisode
    data = load(path, i, step)
    print(data['area'])
    print(data['action'])
    action = data['action']

    if args.render_demo:
        next_obs = []
        crump_obs = data['obs'][:, :, :-1]
        W, H = np.array(crump_obs).shape[:2]
        print("W H = ", W, H)
        p0 = (action[0][0], action[0][1])
        p0_pixel = (int((p0[1] + 1.) / 2. * H), int((p0[0] + 1.) / 2. * W))

        for u in range(max(0, p0_pixel[0] - 2), min(H, p0_pixel[0] + 2)):
            for v in range(max(0, p0_pixel[1] - 2), min(W, p0_pixel[1] + 2)):
                crump_obs[u][v] = (255, 0, 0)

        curr_obs = data['curr_obs']
        if len(curr_obs):
            for i in range(5):
                curr_img = curr_obs[i]
                next_img = curr_img[:, :, :-1]
                p1 = (action[i][2], action[i][3])
                p1_pixel = (int((p1[1] + 1.) / 2. * H), int((p1[0] + 1.) / 2. * W))
                for u in range(max(0, p1_pixel[0] - 2), min(H, p1_pixel[0] + 2)):
                    for v in range(max(0, p1_pixel[1] - 2), min(W, p1_pixel[1] + 2)):
                        next_img[u][v] = (255, 255, 255)
                next_obs.append(next_img.copy())
            l_img = np.concatenate((
                cv2.cvtColor(crump_obs, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(next_obs[0], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(next_obs[1], cv2.COLOR_BGR2RGB)),
                axis=1)
            r_img = np.concatenate((
                cv2.cvtColor(next_obs[2], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(next_obs[3], cv2.COLOR_BGR2RGB),
                cv2.cvtColor(next_obs[4], cv2.COLOR_BGR2RGB)),
                axis=1)
            img = np.concatenate((l_img, r_img), axis=0)
        else:
            img = cv2.cvtColor(crump_obs, cv2.COLOR_BGR2RGB)

        print(f"saved {args.save_path}{args.iepisode}-{step}.jpg")
        cv2.imwrite(f'{args.save_path}{args.iepisode}-{step}.jpg', img)
