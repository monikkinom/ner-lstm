import numpy as np, collections
from numpy.random import randint as rint
import math, matplotlib.pyplot as plt
import pickle as pkl
import tensorflow as tf

class TrigramIndex:

	def __init__(self):
		self.tri = {}
		self.alpha = [chr(i) for i in range(97, 123)]
		self.alpha.insert(0, '#')
		self.trigramBuilder()

	def trigramBuilder(self):
		_ = [0, 0, 0]
		index = 0
		for i in range(27):
			_[0] = self.alpha[i]
			for j in range(27):
				_[1] = self.alpha[j]
				for k in range(27):
					_[2] = self.alpha[k]
					self.tri[''.join(_)] = index
					index = index + 1

	def get(self, trigram):
		return self.tri[trigram]

	def triList(self, word, inPlace = None):
		word = '#' + word + '#'
		tl = []
		for i in range(len(word) - 2):
			trigram = word[i : i + 3]
			ti = self.tri[trigram]
			if len(inPlace) != 0:
				inPlace[ti] += 1
			else:
				tl.append(ti)
		return tl

class TensorCorpus:

	def __init__(self, fName):
		self.fName = fName
		self.worPtr = 0
		self.trigrams = TrigramIndex()
		self.vocab = {}
		self.data = open(fName, 'r', errors = 'ignore').read()
		self.data = self.data.split()
		self.vocabBuilder(50000)
		print('Corpus Initialized.')

	def vocabBuilder(self, size):
		count = [['UNK', 0]]
		count.extend(collections.Counter(self.data).most_common(size - 1))	
		for word, _ in count:
			self.vocab[word] = len(self.vocab)		

	def nextBatch(self, batchSize, window, C):
		batch = np.zeros((batchSize, len(self.trigrams.tri)))
		label = np.zeros((batchSize, 1))
		bind = 0
		assert(batchSize % C == 0)
		assert(2 * window >= C)
		while self.worPtr < len(self.data):
			s = max(0, self.worPtr - window)
			e = min(len(self.data), self.worPtr + window + 1)
			if e - s - 1 < C:
				self.worPtr = self.worPtr + 1
				continue
			visited = [self.worPtr]
			for i in range(C):
				add = rint(s, e)
				while add in visited:
					add = rint(s, e)
				self.trigrams.triList(self.data[self.worPtr], batch[i])
				label[i] = self.vocab.get(self.data[add], 0)
				visited.append(add)
				bind = bind + 1
			self.worPtr = self.worPtr + 1
			if bind == batchSize:
				return [batch, label]
		self.worPtr = 0
		return [batch, label]

class WordVec:

	def __init__(self, corpus = None):
		self.trigrams = TrigramIndex()
		self.triVec = None
		if corpus != None:
			self.corpus = corpus
		else:
			self.loadVec()

	def genVec(self, word):
		ind = self.trigrams.triList(word)
		embedding = np.sum(self.triVec[:, ind], axis = 1)
		return [embedding.reshape(-1)]

	def loadVec(self):
		self.triVec = np.load('TriEmbeddings.npy')

	def similarity(self, word1, word2):
		vec1 = self.genVec(word1)
		vec2 = self.genVec(word2)
		
		scale = np.linalg.norm(vec1) \
		* np.linalg.norm(vec2)
		
		score = np.sum(vec1 * vec2) / scale
		
		return score

	def train(self, dim, batchSize, epoch):
		graph = tf.Graph()
		triSize = len(self.trigrams.tri)
		with graph.as_default():
			trainInputs = tf.placeholder(tf.float32, shape=[batchSize, triSize])
			trainLabels = tf.placeholder(tf.float32, shape=[batchSize, 1])
			with tf.device('/cpu:0'):
				embeddings = tf.Variable(tf.random_uniform([triSize, dim], -1.0, 1.0))
				embed = tf.matmul(trainInputs, embeddings)
				nceWeights = tf.Variable(tf.truncated_normal([len(self.corpus.vocab), dim],stddev=1.0 / math.sqrt(dim)))
				nceBiases = tf.Variable(tf.zeros([len(self.corpus.vocab)]))
			loss = tf.reduce_mean(tf.nn.nce_loss(nceWeights, nceBiases, embed, trainLabels, 64, len(self.corpus.vocab)))
			optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
			norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
			normalizedEmbeddings = embeddings / norm
		with tf.Session(graph = graph) as sess:
			tf.initialize_all_variables().run()
			total = epoch * len(self.corpus.data) * 2
			for i in range(total):
				[batch, label] = self.corpus.nextBatch(batchSize, 5, 2)
				_, lossVal = sess.run([optimizer, loss], feed_dict = {trainInputs:batch, trainLabels:label})
				print(str(i) + '/' + str(total) + ' Loss:', lossVal)
			triembed = normalizedEmbeddings.eval()
		np.save('TriEmbeddings.npy', triembed)

def main():
	corpus = TensorCorpus('/home/shreenivas/Desktop/Corpus/text8')
	wvec = WordVec(corpus)
	wvec.train(100, 5000, 5)

if __name__ == '__main__':
	main()
