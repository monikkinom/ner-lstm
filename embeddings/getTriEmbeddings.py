from TWordVec import WordVec
import numpy as np
import random
import pickle as pkl
import sys

WORD_DIM = 300
model = WordVec()


def findMaxLenght(FILE_NAME):
	temp = 0
	max_lenght = 0

	for line in open(FILE_NAME):
		if line in ['\n', '\r\n']:
			if temp > max_lenght:
				max_lenght = temp
			temp = 0
		else:
			temp += 1

	return max_lenght 

def pos(tag):

	onehot = np.zeros(5)
	if tag == 'NN' or tag == 'NNS':
		onehot[0] = 1
	elif tag == 'FW':
		onehot[1] = 1
	elif tag == 'NNP' or tag == 'NNPS':
		onehot[2] = 1
	elif 'VB' in tag:
		onehot[3] = 1
	else:
		onehot[4] = 1

	return onehot

def chunk(tag):
	
	onehot = np.zeros(5)
	if 'NP' in tag:
		onehot[0] = 1
	elif 'VP' in tag:
		onehot[1] = 1
	elif 'PP' in tag:
		onehot[2] = 1
	elif tag == 'O':
		onehot[3] = 1
	else:
		onehot[4] = 1

	return onehot

def capital(word):
	if ord(word[0]) >= 'A' and ord(word[0]) <= 'Z':
		return np.array([1])
	else:
		return np.array([0])


def get_input(FILE_NAME, op, optag):
	word = []
	tag = []

	sentence = []
	sentence_tag = []

	#get max words in sentence
	max_sentence_length = findMaxLenght(FILE_NAME)
	sentence_length = 0

	print "size is : " + str(max_sentence_length)


	for line in open(FILE_NAME):
		if line in ['\n', '\r\n']:
			for _ in range(max_sentence_length - sentence_length):
				tag.append(np.array([0,0,0,0,0]))
				temp = [0 for _ in range(WORD_DIM+11)]
				word.append(temp)
				
			sentence.append(word)
			sentence_tag.append(np.array(tag))

			sentence_length = 0	
			word = []
			tag = []


		else:
			assert(len(line.split()) == 4)
			sentence_length += 1

			temp = model.genVec(line.split()[0])
			temp = np.append(temp,pos(line.split()[1])) # adding pos embeddings
			temp = np.append(temp,chunk(line.split()[2])) # adding chunk embeddings
			temp = np.append(temp,capital(line.split()[0])) # adding capital embedding

			word.append(temp)
			t = line.split()[3]

			# Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

			if t.endswith('O'):
				tag.append(np.array([1, 0, 0, 0, 0]))
			elif t.endswith('PER'):
				tag.append(np.array([0, 1, 0, 0, 0]))
			elif t.endswith('LOC'):
				tag.append(np.array([0, 0, 1, 0, 0]))
			elif t.endswith('ORG'):
				tag.append(np.array([0, 0, 0, 1, 0]))
			elif t.endswith('MISC'):
				tag.append(np.array([0, 0, 0, 0, 1]))
			else:
				print t
				print "error in input"
				sys.exit(0)

	assert(len(sentence) == len(sentence_tag))
	print 'pickling'
	pkl.dump(sentence,open(op,'wb'))
	pkl.dump(sentence_tag,open(optag,'wb'))

train = raw_input("enter train file")
testa = raw_input("enter testa file")
testb = raw_input("enter testb file")

get_input(train,'50_train_tvec','50_train_tag')
get_input(testa,'50_testa_tvec','50_testa_tag')
get_input(testb,'50_testb_tvec','50_testb_tag')

