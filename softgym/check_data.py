import pickle
import argparse
import os
import random
import cv2
import numpy as np
import copy
from models import Affordance

def load(path, iepisode, step):
    field_path = path
    fname = f'{iepisode:06d}-{step}.pkl'
    return pickle.load(open(os.path.join(field_path, fname), 'rb'))

def preprocess(image_in):
    image = copy.deepcopy(image_in)
    """Pre-process images (subtract mean, divide by std)."""
    color_mean = 0.18877631
    depth_mean = 0.00509261
    color_std = 0.07276466
    depth_std = 0.00903967
    image[:, :, :3] = (image[:, :, :3] / 255 - color_mean) / color_std
    image[:, :, 3:] = (image[:, :, 3:] - depth_mean) / depth_std
    return image

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',            default='./data/')
    parser.add_argument('--save_path',       default='./')
    parser.add_argument('--iepisode',        default=0, type=int)
    parser.add_argument('--step',            default=1, type=int)
    parser.add_argument('--render_demo',     action='store_true')
    parser.add_argument('--need_aff',        action='store_true')
    parser.add_argument('--load_aff_dir',    default='xxx')

    args = parser.parse_args()

    path = args.path
    step = args.step
    for i in range(args.iepisode):
        data = load(path, i, step)
        score_list = ''
        print(data['area'])
        print(data['action'])
        if args.render_demo:
            next_obs = []
            crump_obs = data['obs'][:, :, :-1]
            curr_obs = data['curr_obs']
            if len(curr_obs):
                if args.need_aff:
                    aff_model = Affordance(input_shape=(320, 320, 4), preprocess=preprocess, learning_rate=0)
                    aff_model.load(args.load_aff_dir)
                curr_obs = data['curr_obs']
                for i in range(5):
                    curr_img = curr_obs[i]
                    if args.need_aff:
                        attention = aff_model.forward(curr_img)
                        vis_aff = np.array(attention[0])
                        vis_aff = vis_aff - np.min(vis_aff)
                        vis_aff = 255 * vis_aff / np.max(vis_aff)
                        vis_aff = cv2.applyColorMap(np.uint8(vis_aff), cv2.COLORMAP_JET)
                        cv2.imwrite(f'{args.save_path}{args.iepisode}-{step}-{i}.jpg', vis_aff)

                        score = np.max(attention)
                        print(score)
                        score_list += f'-{int(score * 2)}'
                    next_img = curr_img[:, :, :-1]
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

            print(f"saved {args.save_path}")
            cv2.imwrite(f'{args.save_path}{args.iepisode}-{step}-{score_list}.jpg', img)
