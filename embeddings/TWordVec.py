import numpy as np, collections
from numpy.random import randint as rint
import math, pickle as pkl
import tensorflow as tf

NGRAM = 3
MWL = 45
WINDOW = 5

class NGram:
	def __init__(self, N):
		self.N = N

	def alpha(self, ch):
		return ord(ch)-97

	def __getitem__(self, gram):
		gram = gram.lower()
		ind = 0
		for ch in gram:
			ind = ind*26+self.alpha(ch)
		app = 0
		for l in range(len(gram)):
			app += 26**l
		return ind+app

	def gramList(self, word, gl):
		word = word.lower()
		gramlist = []
		if len(word) < self.N:
			gram = word
			gramlist.append(self.__getitem__(gram))
		else:
			for i in range(0, len(word), self.N):
				if i+self.N>len(word):
					break
				gram = word[i:i+self.N]
				gramlist.append(self.__getitem__(gram))
		l = len(gramlist)
		gl -= l
		gl = [0]*gl
		gramlist.extend(gl)
		return [gramlist, l]

class TensorCorpus:
	def __init__(self, fName, N = 3, window = 5, maxWordLength = 50):
		self.fName = fName
		self.N = N
		self.window = window
		self.mwl = maxWordLength
		self.ngram = NGram(N)
		self.vocab = {}
		self.data = open(fName, 'r').read().split()
		self.trim(self.mwl)
		self.vocabBuilder(50000)
		self.createData()
		print('Corpus created.')

	def trim(self, mwl):
		tmp = []
		for word in self.data:
			if len(word) <= self.mwl:
				tmp.append(word)
		self.data = tmp

	def vocabBuilder(self, size):
		count = [['UNK', 0], ['END', 0]]
		count.extend(collections.Counter(self.data).most_common(size-2))
		for word, _ in count:
			self.vocab[word] = len(self.vocab)

	def createData(self):
		X = np.zeros((len(self.data), self.mwl-self.N+1), dtype = np.int32)
		Y = np.ones((len(self.data), 2*self.window), dtype = np.int32)
		L = np.zeros((len(self.data), 1), dtype = np.float32)
		flatinp = []
		i = 0 
		for word in self.data:
			flatinp.append(self.vocab.get(word, 0))
			X[i, :], L[i] = self.ngram.gramList(word, self.mwl-self.N+1)
			i += 1
		for i in range(len(flatinp)):
			bound = max(0, i-self.window)
			behind = flatinp[bound:i]
			Y[i, self.window:self.window+len(behind)] = behind
			ahead = flatinp[i+1:i+1+self.window]
			Y[i, 0:len(ahead)] = ahead
		pkl.dump(self.vocab, open('vocab.pkl', 'wb'))
		np.save("Y"+str(WINDOW)+str(self.mwl)+".npy", Y)
		np.save("X"+str(self.N)+str(self.mwl)+".npy", X)
		np.save("L"+str(self.N)+str(self.mwl)+".npy", L)

class GVec:
	def __init__(self, N, dim, mwl, vec = []):
		self.ngram = NGram(N)
		self.tot = 0
		for i in range(N+1):
			self.tot += 26**i
		self.N = N
		self.dim = dim
		self.mwl = mwl	
		self.vec = vec

	def __getitem__(self, word):
		word = list(word.lower())
		for ptr in range(len(word)):
			if not word[ptr].isalpha():
				word[ptr] = ' '
		words = ''.join(word).split()
		embedding = np.zeros((self.dim))
		for word in words:
			obj = self.ngram.gramList(word, self.mwl-self.N+1)
			embedding += np.sum(self.vec[obj[0], :], axis = 0)/obj[1]
		l = max(1, len(words))
		return embedding/l

	def cosineDistance(self, word1, word2):
		vec1 = self.__getitem__(word1)
		vec2 = self.__getitem__(word2)
		n1 = np.linalg.norm(vec1)
		n2 = np.linalg.norm(vec2)
		score = np.dot(vec1, vec2)/(n1*n2)
		return score

	def mostSimilar(self, vocab, word, no = 10):
		word = word.lower()
		wordtup = []
		for attr in vocab:
			if attr != 'UNK':
				r = self.cosineDistance(word, attr)
				wordtup.append([attr, r])
		return sorted(wordtup, key = lambda x:abs(x[1]), reverse = True)[0:no]

	def train(self, X, Y, L, batchSize, vocab, epoch = 1):
		graph = tf.Graph()
		zind = [0]
		with graph.as_default():
			trainInputs = tf.placeholder(tf.int32, shape=[None, X.shape[1]])
			trainLabels = tf.placeholder(tf.int32, shape=[None, Y.shape[1]])
			trainLength = tf.placeholder(tf.float32, shape=[None, 1])
			embeddings = tf.Variable(tf.random_uniform([self.tot, self.dim], -1.0, 1.0))
			zeros = tf.zeros([1, self.dim], dtype = tf.float32)
			zeroindices = tf.constant(zind, dtype = tf.int32)
			setz = tf.scatter_update(embeddings, zeroindices, zeros)
			embedlook = tf.nn.embedding_lookup(embeddings, trainInputs)
			embed = tf.reduce_sum(embedlook, 1)/trainLength
			nceWeights = tf.Variable(tf.truncated_normal([len(vocab), self.dim], stddev = 1.0/math.sqrt(self.dim)))
			nceBiases = tf.Variable(tf.zeros([len(vocab)]))
			loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(nceWeights, nceBiases, embed, trainLabels, 1000, len(vocab), Y.shape[1]))
			optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
		with tf.Session(graph = graph) as sess:
			saver = tf.train.Saver()
			tf.initialize_all_variables().run()
			sess.run(setz)
			NO = (X.shape[0]+batchSize-1)/batchSize
			try:
				saver.restore(sess, 'backModel.ckpt')
			except:
				pass
			lossAvg = 0
			for i in range(0, X.shape[0], batchSize):
				_, lossVal = sess.run([optimizer, loss], feed_dict = {trainInputs:X[i:i+batchSize], trainLength:L[i:i+batchSize], trainLabels:Y[i:i+batchSize]})
				sess.run(setz)
				lossAvg = lossAvg + lossVal
				print('\rBatch %d/%d Avg batch Loss:%f' % (i//batchSize+1, NO, lossVal), end = "...")
			print("Average epoch loss =", lossAvg/NO)
			saver.save(sess, 'backModel.ckpt')
			self.vec = embeddings.eval()
			np.save('Vectors'+str(self.N)+str(self.dim)+'.npy', self.vec)

def main():
	try:
		X = np.load("X"+str(NGRAM)+str(MWL)+".npy")
		L = np.load("L"+str(NGRAM)+str(MWL)+".npy")
		Y = np.load("Y"+str(WINDOW)+str(MWL)+".npy")
	except:
		corpus = TensorCorpus('/media/shreenivas/LinuxData/IIITH/Corpus/Tensor.txt', NGRAM, WINDOW, MWL)
		del corpus
		X = np.load("X"+str(NGRAM)+str(MWL)+".npy")
		L = np.load("L"+str(NGRAM)+str(MWL)+".npy")
		Y = np.load("Y"+str(WINDOW)+str(MWL)+".npy")
	wvec = GVec(NGRAM, 300, MWL)
	wvec.train(X, Y, L, 5000, pkl.load(open('vocab.pkl', 'rb')))

if __name__ == '__main__':
	main()
