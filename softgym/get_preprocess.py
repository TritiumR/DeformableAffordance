import os
import argparse
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data/')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--render_demo', action='store_true')
    args = parser.parse_args()

    dirs = os.listdir(args.path)

    mean = np.zeros(4)
    std = np.zeros(4)

    i = 0
    for fname in dirs:
        i += 1
        if (i == 100):
            break
        data = pickle.load(open(os.path.join(args.path, fname), 'rb'))
        obs = np.array(data['obs'])
        for d in range(4):
            mean[d] += obs[:, :, d].mean() / 255.
            std[d] += obs[:, :, d].std() / 255.

    mean /= len(dirs)
    std /= len(dirs)

    mean = np.tile(mean, (5, 5, 1))
    std = np.tile(std, (5, 5, 1))
    print(mean)
    for d in range(4):
        print(mean[:, :, d].std())
        print(std[:, :, d].std())

if __name__ == '__main__':
    main()
