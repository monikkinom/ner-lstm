#from gensim.models import word2vec
from RandomVec import RandomVec
#from WordVec import WVec
import numpy as np
import random
import sys, pickle as pkl

WORD_DIM = 300
model = pkl.load(open('glove.pkl', 'rb'))
rvec = RandomVec(WORD_DIM)

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

def get_input(FILE_NAME,op,optag):
	word = []
	tag = []

	sentence = []
	sentence_tag = []

	#get max words in sentence
	max_sentence_length = 70
	sentence_length = 0

	print("size is : " + str(max_sentence_length))


	for line in open(FILE_NAME):
		if line in ['\n', '\r\n']:
#	print "aa"	
			for _ in range(max_sentence_length - sentence_length):
				tag.append(np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
				temp = [0 for _ in range(WORD_DIM+5)]
				word.append(temp)
				
			
#			assert (len(word) == 113)
#			assert (len(tag) == 113)

			sentence.append(word)
			sentence_tag.append(np.array(tag))

			sentence_length = 0	
			word = []
			tag = []


		else:
			assert(len(line.split()) == 3)
			sentence_length += 1

			try:
				temp = model[line.split()[0].lower()]
			except:
				temp = rvec.getVec(line.split()[0].lower())
			temp = np.append(temp, pos(line.split()[1]))
			word.append(temp)
			t = line.split()[2]

			# Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc

			if t == 'PERSON':
				tag.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
			elif t == 'ORGANIZATION':
				tag.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
			elif t == 'LOCATION':
				tag.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
			elif t == 'ENTERTAINMENT':
				tag.append(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
			elif t == 'FACILITIES':
				tag.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
			elif t == 'ARTIFACT':
				tag.append(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
			elif t == 'LIVTHINGS':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
			elif t == 'LOCOMOTIVE':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
			elif t == 'PLANTS':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
			elif t == 'MATERIALS':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
			elif t == 'DISEASE':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))	
			elif t == 'O':
				tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
			else:
				print(t)
				print("error in input")
				sys.exit(0)

	assert(len(sentence) == len(sentence_tag))
	#print sentence_tag[0]
	print("pickling")
	pkl.dump(sentence,open(op,'wb'))
	pkl.dump(sentence_tag,open(optag,'wb'))


train = input("enter train file")
testa = input("enter testa file")
testb = input("enter testb file")

get_input(train,'50_train_wvec','50_train_tag')
get_input(testa,'50_testa_wvec','50_testa_tag')
get_input(testb,'50_testb_wvec','50_testb_tag')
