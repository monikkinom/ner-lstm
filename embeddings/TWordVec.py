import numpy as np, collections
from numpy.random import randint as rint
import math, pickle as pkl
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

	def triList(self, word, wl):
		word = word.lower()
		pad = wl - len(word)
		word += '#' * pad
		tl = []
		for i in range(len(word) - 2):
			trigram = word[i : i + 3]
			ti = self.tri[trigram]
			tl.append(ti)
		return np.array(tl)

class TensorCorpus:

	def __init__(self, fName, maxWordLength = 47):
		self.fName = fName
		self.worPtr = 0
		self.mwl = maxWordLength
		self.trigrams = TrigramIndex()
		self.vocab = {}
		self.mostCommon = []
		self.data = open(fName, 'r').read()
		self.data = self.data.split()
		self.vocabBuilder(100000)
		print('Corpus Initialized.')

	def vocabBuilder(self, size):
		count = [['UNK', 0]]
		count.extend(collections.Counter(self.data).most_common(size - 1))
		for word, _ in count:
			self.vocab[word] = len(self.vocab)
		self.mostCommon = [count[i][0] for i in range(150)]

	def nextBatch(self, batchSize, window, C):
		batch = np.zeros((batchSize, self.mwl - 2))
		label = np.zeros((batchSize, 1))
		bind = 0
		assert(batchSize % C == 0)
		assert(2 * window >= C)
		while True:
			s = max(0, self.worPtr - window)
			e = min(len(self.data), self.worPtr + window + 1)
			if e - s - 1 < C:
				self.worPtr = self.worPtr + 1
				continue
			visited = [self.worPtr]
			batch[bind : bind + C] = self.trigrams.triList(self.data[self.worPtr], self.mwl)
			for i in range(C):
				add = rint(s, e)
				while add in visited:
					add = rint(s, e)
				label[bind] = self.vocab.get(self.data[add], 0)
				visited.append(add)
				bind = bind + 1
			self.worPtr = self.worPtr + 1
			if self.worPtr == len(self.data):
				self.worPtr = 0
			if bind == batchSize:
				return [batch, label]

class WordVec:

	def __init__(self, corpus = None, maxWordLength = 47):
		self.trigrams = TrigramIndex()
		self.triVec = None
		self.mwl = maxWordLength
		if corpus != None:
			self.corpus = corpus
		else:
			self.loadVec()

	def genVec(self, word):
		word = list(word.lower())
		for ptr in range(len(word)):
			if not word[ptr].isalpha():
				word[ptr] = ' '
		words = ''.join(word).split()
		ind = []
		for word in words:
			ind += list(self.trigrams.triList(word, self.mwl))
		embedding = np.sum(self.triVec[ind, :], axis = 0)
		return embedding.reshape(-1)

	def loadVec(self):
		self.triVec = np.load('TrigramVectors300.npy')

	def similarity(self, word1, word2):
		vec1 = self.genVec(word1)
		vec2 = self.genVec(word2)
		n1 = np.linalg.norm(vec1)
		n2 = np.linalg.norm(vec2)
		score = np.sum(vec1 * vec2) / (n1 * n2)
		return score

	def train(self, dim, batchSize, epoch):
		graph = tf.Graph()
		triSize = len(self.trigrams.tri)
		zind = [0]
		with graph.as_default():
			trainInputs = tf.placeholder(tf.int64, shape=[batchSize, self.mwl - 2])
			trainLabels = tf.placeholder(tf.int64, shape=[batchSize, 1])
			embeddings = tf.Variable(tf.random_uniform([triSize, dim], -1.0, 1.0))
			zeros = tf.zeros([1, dim], dtype = tf.float32)
			zeroindices = tf.constant(zind, dtype = tf.int32)
			setz = tf.scatter_update(embeddings, zeroindices, zeros)
			embedlook = tf.nn.embedding_lookup(embeddings, trainInputs)
			embed = tf.reduce_sum(embedlook, 1)
			nceWeights = tf.Variable(tf.truncated_normal([len(self.corpus.vocab), dim],stddev=1.0 / math.sqrt(dim)))
			nceBiases = tf.Variable(tf.zeros([len(self.corpus.vocab)]))
			loss = tf.reduce_mean(tf.nn.nce_loss(nceWeights, nceBiases, embed, trainLabels, 256, len(self.corpus.vocab)))
			optimizer = tf.train.GradientDescentOptimizer(.003).minimize(loss)
		with tf.Session(graph = graph) as sess:
			saver = tf.train.Saver()
			tf.initialize_all_variables().run()
			sess.run(setz)
			i = 0
			if restore:
				saver.restore(sess, 'backModel.ckpt')
				i = pkl.load(open('backEPOCH.pkl', 'rb'))
				print('Restored Model state.')
			total = (epoch * len(self.corpus.data) * 2 + batchSize - 1) // batchSize
			lossAvg = 0
			while i < total:
				[batch, label] = self.corpus.nextBatch(batchSize, 5, 2)
				_, lossVal = sess.run([optimizer, loss], feed_dict = {trainInputs:batch, trainLabels:label})
				sess.run(setz)
				lossAvg = lossAvg + lossVal
				i = i + 1
				if i % 1000 == 0:
					print(i * 100 / total, '% done. Avg Batch Loss:', lossAvg / 1000)
					lossAvg = 0
				if i % 10000 == 0:
					savep = saver.save(sess, 'backModel.ckpt')
					pkl.dump(self.corpus.worPtr, open('backCorpus.pkl', 'wb'))
					pkl.dump(i, open('backEPOCH.pkl', 'wb'))
					self.triVec = embeddings.eval()
					np.save('TrigramVectors' + str(dim) + '.npy', self.triVec)	
					print('Saved Everything.')
			self.triVec = embeddings.eval()
			np.save('TrigramVectors' + str(dim) + '.npy', self.triVec)

def main():	
	#corpus = TensorCorpus('/home/shreenivas/Desktop/Corpus/Corpus.txt')
	#if restore:
	#	corpus.worPtr = pkl.load(open('backCorpus.pkl', 'rb'))
	#	print('Restored Corpus state.')
	numList = ['zero', 'one', 'two', 'three', 'four', 'six', 'seven', 'eight', 'nine']
	wvec = WordVec()
	s = 0
	no = 0
	for i in numList:
		for j in numList:
			if i != j:
				s = s + wvec.similarity(i, j)
				no = no  + 1
	print(s / no)
	#wvec.train(300, 1000, 3)

if __name__ == '__main__':
	restore = input('Restore:')
	restore = int(restore)
	main()
