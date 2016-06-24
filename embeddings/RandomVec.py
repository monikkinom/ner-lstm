import numpy as np

class RandomVec():

	def __init__(self, dim):
		self.dim = dim
		self.vocab = {}
		self.vec = []

	def getVec(self, word):
		ind = self.vocab.get(word, -1)
		if ind == -1:
			nvec = np.random.random((self.dim))
			self.vocab[word] = len(self.vocab)
			self.vec.append(nvec)
			return nvec
		else:
			return self.vec[ind]
