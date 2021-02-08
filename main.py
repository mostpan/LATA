import dataset
import argparse
import model
import numpy as np
import os
import random

seed_value = 2020
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LATA")

    parser.add_argument('--dataset', type=str, default='FWFW')
    parser.add_argument('--fraction', type=float, default=0.5)

    parser.add_argument('--times', type=int, default=10)
    parser.add_argument('--pre_epochs', type=int, default=10000)
    parser.add_argument('--for_epochs', type=int, default=10000)
    parser.add_argument('--verbose', type=int, default=100)
    parser.add_argument('--use_batch', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--activation', type=str, default='leak_relu')

    parser.add_argument('--alpha', type=float, default=3)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--aug_fraction', type=float, default=0.3)
    parser.add_argument('--cor_fraction', type=float, default=0.3)

    parser.add_argument('--encoder_sizes', nargs='?', default='[128,16]')
    parser.add_argument('--z_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--reg', type=float, default=0.0015)

    parser.add_argument('--n_node', type=int, default='-1')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    print(args)

    times = args.times
    datas = []
    for i in range(times):
        data = dataset.load_data(args.dataset, args.fraction)
        datas.append(data)

    for i in range(0,times):
        g_o = datas[i]
        args.n_node = len(g_o.nodes())
        mode = model.Model(args)
        mode.train(g_o, args)

