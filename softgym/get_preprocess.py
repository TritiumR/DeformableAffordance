import os
import argparse
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./data/')
    args = parser.parse_args()

    dirs = os.listdir(args.path)

    mean = np.zeros(4)
    std = np.zeros(4)

    for fname in dirs:
        print(fname)
        data = pickle.load(open(os.path.join(args.path, fname), 'rb'))
        obs = np.array(data['obs'])
        for d in range(4):
            mean[d] += obs[:, :, d].mean() / 255.
            std[d] += obs[:, :, d].std() / 255.

    mean /= len(dirs)
    std /= len(dirs)

    print("mean: ", mean)
    print("std: ", std)

if __name__ == '__main__':
    main()
