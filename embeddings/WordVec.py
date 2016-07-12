import numpy as np, collections
from numpy.random import randint as rint
import math, pickle as pkl
import tensorflow as tf

class TensorCorpus:

	def __init__(self, fName):
		self.fName = fName
		self.worPtr = 0
		self.vocab = {}
		self.mostCommon = []
		self.data = open(fName, 'r').read().split(u' ')
		self.vocabBuilder(300000)
		print('Corpus Initialized.')

	def vocabBuilder(self, size):
		count = [[u'UNK', 0]]
		count.extend(collections.Counter(self.data).most_common(size - 1))
		for word, _ in count:
			self.vocab[word] = len(self.vocab)
		self.mostCommon = [count[i][0] for i in range(150)]

	def nextBatch(self, batchSize, window, C):
		batch = np.zeros((batchSize))
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
			batch[bind : bind + C] = self.vocab.get(self.data[self.worPtr], 0)
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

	def __init__(self, corpus, load):
		self.vec = None
		self.corpus = corpus
		if load:
			self.loadVec()

	def genVec(self, word):
		return self.vec[self.corpus.vocab.get(word, 0)]

	def loadVec(self):
		self.triVec = np.load('WordVectors300.npy')

	def similarity(self, word1, word2):
		vec1 = self.genVec(word1)
		vec2 = self.genVec(word2)
		n1 = np.linalg.norm(vec1)
		n2 = np.linalg.norm(vec2)
		score = np.sum(vec1 * vec2) / (n1 * n2)
		return score

	def train(self, dim, batchSize, epoch):
		graph = tf.Graph()
		vocabSize = len(self.corpus.vocab)
		with graph.as_default():
			trainInputs = tf.placeholder(tf.int64, shape=[batchSize])
			trainLabels = tf.placeholder(tf.int64, shape=[batchSize, 1])
			embeddings = tf.Variable(tf.random_uniform([vocabSize, dim], -1.0, 1.0))
			embedlook = tf.nn.embedding_lookup(embeddings, trainInputs)
			nceWeights = tf.Variable(tf.truncated_normal([vocabSize, dim],stddev=1.0 / math.sqrt(dim)))
			nceBiases = tf.Variable(tf.zeros([vocabSize]))
			loss = tf.reduce_mean(tf.nn.nce_loss(nceWeights, nceBiases, embedlook, trainLabels, 128, vocabSize))
			optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
		with tf.Session(graph = graph) as sess:
			saver = tf.train.Saver()
			tf.initialize_all_variables().run()
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
				lossAvg = lossAvg + lossVal
				i = i + 1
				if i % 1000 == 0:
					print(i * 100 / total, '% done. Avg Batch Loss:', lossAvg / 1000)
					lossAvg = 0
				if i % 10000 == 0:
					savep = saver.save(sess, 'backModel.ckpt')
					pkl.dump(self.corpus.worPtr, open('backCorpus.pkl', 'wb'))
					pkl.dump(i, open('backEPOCH.pkl', 'wb'))
					self.vec = embeddings.eval()
					np.save('WordVectors' + str(dim) + '.npy', self.vec)	
					print('Saved Everything.')
			self.vec = embeddings.eval()
			np.save('WordVectors' + str(dim) + '.npy', self.vec)

def main():	
	corpus = TensorCorpus('/home/shreenivas/Desktop/Corpus/HINDI.txt')
	if restore:
		corpus.worPtr = pkl.load(open('backCorpus.pkl', 'rb'))
		print('Restored Corpus state.')
	#numList = ['run', 'swim']
	wvec = WordVec(corpus, False)
	#s = 0
	#no = 0
	#for i in numList:
	#	for j in numList:
	#		if i != j:
	#			s = s + wvec.similarity(i, j)
	#			no = no  + 1
	#print(s / no)
	#print(wvec.triVec[0])
	wvec.train(300, 10000, 35)

if __name__ == '__main__':
	restore = input('Restore:')
	restore = int(restore)
	main()
