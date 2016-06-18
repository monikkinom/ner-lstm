import numpy as np
from numpy.random import randint as rint
import os, matplotlib.pyplot as plt
from NeuralNet import NN3
import pickle as pkl, time

class Corpus:

	def __init__(self, dirName):
		self.dirName = dirName

	def __iter__(self):
		for fName in os.listdir(self.dirName):
			fName = os.path.join(self.dirName, fName)

			punc = str.maketrans({\
			'.' : ' ', '?' : ' ', '!' : ' ', ',' : ' ', \
			';' : ' ', ':' : ' ', '-' : ' ', '_' : ' ', \
			'(' : ' ', ')' : ' ', ']' : ' ', '[' : ' ', \
			'{' : ' ', '}' : ' ', '\"' : ' ', '\'' : None})
			
			if not os.path.isfile(fName):
				continue

			for sentence in \
			open(fName, 'r', errors = 'ignore'):

				sentence = sentence.translate(punc)

				sentenceWords = [\
				'#' + word.lower() + '#' \
				for word in sentence.split() \
				if word.isalpha()]

				yield sentenceWords

class WordVec:

	def __init__(self, corpus = None, exist = True):
		if corpus != None:
			self.corpus = corpus
		self.alphabets = [chr(i) for i in range(97, 123)]
		self.alphabets.insert(0, '#')
		self.trigrams = {}
		self.triVec = []
		self.exist = exist
		self.trigramBuilder()
		if corpus == None:
			self.loadVec()

	def trigramBuilder(self):
		_ = [0, 0, 0]
		index = 0
		for i in range(27):
			_[0] = self.alphabets[i]
			for j in range(27):
				_[1] = self.alphabets[j]
				for k in range(27):
					_[2] = self.alphabets[k]
					
					self.trigrams[''.join(_)] \
					= index
				
					index = index + 1
	
	def triHot(self, word):
		tv = np.zeros((len(self.trigrams)))
		for i in range(len(word) - 2):
			trigram = word[i : i + 3]
			ti = self.trigrams[trigram]
			tv[ti] += 1
		return tv

	def triList(self, word):
		tl = []
		for i in range(len(word) - 2):
			trigram = word[i : i + 3]
			ti = self.trigrams[trigram]
			tl.append(ti)
		return tl

	def genVec(self, word):
		th = self.triList(word)
		if self.exist:
			th = list(set(th))
		IVec = np.sum(self.triVec[0][:, th], axis = 1)
		OVec = np.sum(self.triVec[1][th, :], axis = 0)
		return [IVec.reshape(-1), OVec.reshape(-1)]

	def loadVec(self):
		self.triVec.append(np.load('IVec.npy'))
		self.triVec.append(np.load('OVec.npy'))

	def similarity(self, word1, word2):
		vec1 = self.genVec(word1)
		vec2 = self.genVec(word2)
		
		scaleI = np.linalg.norm(vec1[0]) \
		* np.linalg.norm(vec2[0])

		scaleO = np.linalg.norm(vec1[1]) \
		* np.linalg.norm(vec2[1])
		
		scoreI = np.sum(vec1[0] * vec2[0]) / scaleI
		scoreO = np.sum(vec1[1] * vec2[1]) / scaleO
		
		return [scoreI, scoreO]

	def trainVec(self, dim, window, C, batchSize, epoch, \
	minEps = 0.0001):

		model = NN3(\
		len(self.trigrams), dim, len(self.trigrams), \
		hType = 0, oType = 1, epsilon = 0.001)
	
		decay = (model.epsilon - minEps) / (epoch - 1)

		batch = np.zeros((batchSize, len(self.trigrams)))
		label = np.zeros((batchSize, len(self.trigrams)))
		b = 0
		l = 1
		for itr in range(epoch):
			for sentence in self.corpus:
				for c in range(len(sentence)):
					s = max(0, c - window)

					e = min(len(sentence), \
					c + window + 1)

					if e - s - 1 < C:
						continue
					if self.exist:
						
						ind = \
						self.triList(\
						sentence[c])
						
						batch[b, ind] = 1
					else:
						
						batch[b] = \
						self.triHot(\
						sentence[c])
					
					visited = [c]

					for i in range(C):

						add = rint(s, e)
						while add in visited:
							
							add = \
							rint(s, e)	
						if self.exist:

							ind = \
							self.triList(\
							sentence[add])

							label[b, ind]\
							= 1
				
						else:

							label[b] += \
							self.triHot(\
							sentence[add])
						
						visited.append(add)
					b = (b + 1) % batchSize
					if b == 0:
					
						model.batch(batch, \
						label)
					
						batch = np.zeros(\
						(batchSize, \
						len(self.trigrams)))

						label = np.zeros(\
						(batchSize, \
						len(self.trigrams)))

						print('Batch:', l)
						l = l + 1
			model.epsilon = model.epsilon - decay
		if b != 0:
			model.batch(batch, label)
			print('Batch:', l)
		self.triVec = model.W

def main():
	corpus = Corpus('/home/shreenivas/Desktop/Corpus')
	wvec = WordVec(corpus, True)
	wvec.trainVec(100, 5, 2, 5000, 50)
	np.save('IVec.npy', wvec.triVec[0])
	np.save('OVec.npy', wvec.triVec[1])

if __name__ == '__main__':
	main()
