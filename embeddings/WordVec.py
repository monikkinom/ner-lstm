import numpy as np, collections
from numpy.random import randint as rint
import math, pickle as pkl
import tensorflow as tf

WINDOW = 5

class TensorCorpus:
	def __init__(self, fName, window = 5):
		self.fName = fName
		self.window = window
		self.vocab = {}
		self.data = open(fName, 'r').read().split()
		self.vocabBuilder(100000)
		self.createData()
		print('Corpus created.')

	def vocabBuilder(self, size):
		count = [['UNK', 0]]
		count.extend(collections.Counter(self.data).most_common(size-1))
		for word, _ in count:
			self.vocab[word] = len(self.vocab)

	def createData(self):
		Y = np.zeros((len(self.data), 2*self.window), dtype = np.int32)
		flatinp = []
		i = 0 
		for word in self.data:
			flatinp.append(self.vocab.get(word, 0))
		X = np.array(flatinp, dtype = np.int32)
		for i in range(len(flatinp)):
			bound = max(0, i-self.window)
			behind = flatinp[bound:i]
			ahead = flatinp[i+1:i+1+self.window]
			total = np.resize(np.append(ahead, behind), (2*self.window))
			Y[i] = total
			print("\rCreating %f" % (i*100/len(flatinp)), end = " ")
		pkl.dump(self.vocab, open('vocab.pkl', 'wb'))
		np.save("Y"+str(WINDOW)+".npy", Y)
		np.save("X.npy", X)

class WVec:
	def __init__(self, dim, vocab, vec = []):
		self.dim = dim
		self.vec = vec
		self.vocab = vocab

	def __getitem__(self, word):
		embedding = self.vec[self.vocab.get(word, 0)]
		return embedding

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

	def train(self, X, Y, batchSize, epoch = 1):
		graph = tf.Graph()
		with graph.as_default():
			trainInputs = tf.placeholder(tf.int32, shape=[None])
			trainLabels = tf.placeholder(tf.int32, shape=[None, Y.shape[1]])
			embeddings = tf.Variable(tf.random_uniform([len(self.vocab), self.dim], -1.0, 1.0))
			embed = tf.nn.embedding_lookup(embeddings, trainInputs)
			nceWeights = tf.Variable(tf.truncated_normal([len(self.vocab), self.dim], stddev = 1.0/math.sqrt(self.dim)))
			nceBiases = tf.Variable(tf.zeros([len(self.vocab)]))
			loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(nceWeights, nceBiases, embed, trainLabels, 1000, len(self.vocab), Y.shape[1]))
			optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)
		with tf.Session(graph = graph) as sess:
			saver = tf.train.Saver()
			tf.initialize_all_variables().run()
			NO = (X.shape[0]+batchSize-1)/batchSize
			try:
				saver.restore(sess, 'backModel.ckpt')
			except:
				pass
			for itr in range(epoch):
				lossAvg = 0
				for i in range(0, X.shape[0], batchSize):
					_, lossVal = sess.run([optimizer, loss], feed_dict = {trainInputs:X[i:i+batchSize], trainLabels:Y[i:i+batchSize]})
					lossAvg = lossAvg + lossVal
					print('\rBatch %d/%d Avg batch Loss:%f' % (i//batchSize+1, NO, lossVal), end = "...")
				print("Average epoch loss =", lossAvg/NO)
				saver.save(sess, 'backModel.ckpt')
			self.vec = embeddings.eval()
			np.save('Vectors'+str(self.dim)+'.npy', self.vec)

def main():
	try:
		X = np.load("X.npy")
		Y = np.load("Y"+str(WINDOW)+".npy")
		vocab = pkl.load(open('vocab.pkl', 'rb'))
	except:
		corpus = TensorCorpus('/media/shreenivas/LinuxData/IIITH/Corpus/HINDI.txt', WINDOW)
		del corpus
		X = np.load("X.npy")
		Y = np.load("Y"+str(WINDOW)+".npy")
		vocab = pkl.load(open('vocab.pkl', 'rb'))
	wvec = WVec(300, vocab)
	wvec.train(X, Y, 20000, 3)
	pkl.dump(wvec, open('WVec.pkl', 'wb'))

if __name__ == '__main__':
	main()
