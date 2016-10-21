from random import random
import numpy as np


class RandomVec:
    def __init__(self, dim):
        self.dim = dim
        self.vocab = {}
        self.vec = []

    def __getitem__(self, word):
        ind = self.vocab.get(word, -1)
        if ind == -1:
            new_vec = np.array([random() for i in range(self.dim)])
            self.vocab[word] = len(self.vocab)
            self.vec.append(new_vec)
            return new_vec
        else:
            return self.vec[ind]
