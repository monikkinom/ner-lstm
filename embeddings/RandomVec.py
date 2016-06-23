import numpy as np

class RandomVec():

	def __init__(dim):
		self.dim = dim
		self.vocab = {}
		self.vec = np.zeros((0, self.dim))

	def getVec(self, word):
		ind = self.vocab.get(word, -1)
		if ind == -1:
			nvec = np.random.random((self.dim))
			self.vocab[word] = len(self.vocab)
			self.vec = np.insert(self.vec, len(self.vec), nvec, axis = 0)
			return nvec
		else:
			return self.vec[ind, :]
