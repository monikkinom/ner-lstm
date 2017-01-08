from __future__ import print_function
from random_vec import RandomVec
import numpy as np
import pickle as pkl
import argparse
import os


class GloveVec:
    def __init__(self, args):
        self.model = {}
        self.rand_model = RandomVec(args.dimension)
        if args.restore is None:
            os.chdir(args.glove_path)
            os.system('bash demo.sh ' + args.corpus + ' ' + str(args.dimension))
            os.system('rm -r build *.bin vocab.txt')
            path = 'vectors.txt'
        else:
            path = args.restore
        invalid = 0
        for line in open(path):
            line = line.split()
            word = line[0]
            vector = np.array([float(val) for val in line[1:]])
            if len(vector) != args.dimension:
                invalid += 1
                continue
            self.model[word] = vector
        print('invalid entries %d' % invalid)
        os.system('rm ' + path)
        os.chdir('..')

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.model[word]
        except KeyError:
            return self.rand_model[word]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, help='corpus location', required=True)
    parser.add_argument('--dimension', type=int, help='vector dimension', required=True)
    parser.add_argument('--restore', type=str, default=None, help='pre-trained glove vectors.txt')
    parser.add_argument('--glove_path', type=str, help='path to glove c code', required=True)
    args = parser.parse_args()
    model = GloveVec(args)
    pkl.dump(model, open('glovevec_model_' + str(args.dimension) + '.pkl', 'wb'))
