import numpy as np
from random import random

class RandomVec():

	def __init__(self, dim):
		self.dim = dim
		self.vocab = {}
		self.vec = []

	def getVec(self, word):
		ind = self.vocab.get(word, -1)
		if ind == -1:
			nvec = [random() for i in range(self.dim)]
			self.vocab[word] = len(self.vocab)
			self.vec.append(nvec)
			return nvec
		else:
			return self.vec[ind]
