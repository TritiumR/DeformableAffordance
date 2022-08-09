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
    parser.add_argument('--save_path',       default='./visual/')
    parser.add_argument('--iepisode',        default=0, type=int)
    parser.add_argument('--step',             default=1, type=int)
    parser.add_argument('--render_demo',     action='store_true')

    args = parser.parse_args()

    path = args.path
    step = args.step

    data = load(path, args.iepisode, step)
    print(data['area'])
    print(data['action'])
    if args.render_demo:
        crump_obs = data['obs'][0][:, :, -1]
        curr_obs = data['obs'][1][:, :, -1]
        img = np.concatenate((
            cv2.cvtColor(crump_obs, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(curr_obs, cv2.COLOR_BGR2RGB)),
            axis=1)

        print(f"saved {args.save_path}")
        cv2.imwrite(f'./{args.save_path}-{args.iepisode}-{step}.jpg', img)
