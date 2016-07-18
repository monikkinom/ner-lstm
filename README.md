# ner-lstm
###Named Entity Recognition using multi-layered bidirectional LSTMs and task adapted word embeddings

[Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) is a classification problem of identifying the names of people,organisations,etc (different classes) in a text corpus. 

Previous approaches to the problems have involved the usage of hand crafted language specific features, CRF and HMM based models, gazetteers, etc. Growing interest in deep learning has led to application of deep neural networks to the existing problems like that of NER. 

We have implemented a 2 layer bidirectional LSTM network using tensorflow to classify the named entities for [CoNNL 2003 NER Shared Task](http://www.cnts.ua.ac.be/conll2003/ner/). 

This project can be broadly divided into three components, the deep learning model, input and embeddings. 

####Deep Neural Network
We have used Google's Tensorflow to implement a bidirectional multilayered rnn cell (LSTM). The hyper parameters are present at the top in main.py. Tweaking the parameters can yield a variety of results which are worth noting.

```python
WORD_DIM = 311
MAX_SEQ_LEN = 50
NUM_CLASSES = 5
BATCH_SIZE = 64
NUM_HIDDEN = 256
NUM_LAYERS = 3
NUM_EPOCH = 100
```

We have used a softmax layer as the last layer of the network to produce the final classification outputs. We tried working with different optimizers and we found that AdamOptimzer produced the best results.

The function to calculate the F1 Scores, Prediction, Accuracy and Recall is also included in main.py. We have also included the ability to save/restore an existing model using tensorflow's saver functions.

####Input
The various codes in the embeddings directory creates the embedding pickles which will be used by input.py to give input to main.py while the code is running.
input.py contains the code that loads the embeddings for the words in the CoNLL test and train datasets. It has three functions and a dummy function to test if the network works. The embeddings are first generated using the files in the embedding folder and stored as pickle files. The input is then read from the pickle file and loaded during the training and prediction. The input should return a list of sequences of word embeddings.

####Embeddings
Each unique word should have certain number of features, these are called embeddings or also vectors. These are the input features to the neural architecture we are using.

We experimented with different embeddings and observed the results.
Random Vectors, WordVectors made by Google inc and our new generated trigram embeddings.

Random Vectors:Dimensions=300
WordVectors:Dimensions=300
TrigramVectors:Dimensions=300
We are using extra 11 lexical and other features like Capital adding up to 300+11=311
Final dimensions=311

TWordVec.py generates the Trigram Vectors from a supplied Corpus(Cleaned and Removed punctuations and is just a list of words separated by spaces)
We used a corpus of tensorflow + Open american national library + some random clippings total amounting to 290mb of raw text data(File contating just words separated by spaces).
We need to supply the max word length to the TWordVec to generate the trigram embeddings.

We are using these embeddings to generate the input features of CONLL-2003 NER TASK DATA(Train, Test and Development).
getRandomEmbeddings.py generates the pickled embedding file for random vectors.
getWordEmbeddings.py generates the pickled embedding file for Word vectors.
getTriEmbeddings.py generates the pickled embedding file for Trigram vectors.
After the respective pickles are generated the input.py can sends these pickles to main.py when called.

Papers on WordVectors:

https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf

Good tutorial on WordVectors:https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html

###Results(Ran on CONLL eval script):

**Word2Vec 311 dimensions**
---------------------------
*Testa*
-------
processed 49644 tokens with 8211 phrases; found: 8080 phrases; correct: 7619.

Accuracy = 98.54%

Class | Precission | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 94.29 | 92.79 | 93.54 | 8080
*LOC* | 94.86 | 94.25 | 94.56 | 2023
*MISC* | 91.99 | 85.09 | 88.40 | 1123
*ORG* | 91.98 | 89.76 | 90.86 | 2020
*PER* | 96.40 | 97.16 | 96.78 | 2914

*Testb*
-------
processed 45151 tokens with 7719 phrases; found: 7740 phrases; correct: 6911.

Accuracy = 97.49%

Class | Precission | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 89.29 | 89.53 | 89.41 | 7740
*LOC* | 89.67 | 90.87 | 90.27 | 1898
*MISC* | 75.80 | 75.80 | 75.80 | 905
*ORG* | 88.03 | 87.27 | 87.65 | 2415
*PER* | 95.04 | 95.69 | 95.37 | 2522

**TrigramVec 311 dimensions**
---------------------------
*Testa*
-------
processed 49644 tokens with 8211 phrases; found: 8080 phrases; correct: 7619.

Accuracy = 95.97%

Class | Precission | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 82.46 | 78.48 | 80.42 | 8080
*LOC* | 81.53 | 85.41 | 83.43 | 2023
*MISC* | 88.35 | 70.59 | 78.48 | 1123
*ORG* | 73.99 | 72.27 | 73.12 | 2020
*PER* | 87.43 | 81.36 | 84.29 | 2914

*Testb*
-------
processed 45151 tokens with 7719 phrases; found: 7740 phrases; correct: 6911.

Accuracy = 94.19%

Class | Precission | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 73.64 | 73.44 | 73.54 | 7740
*LOC* | 75.89 | 78.80 | 77.32 | 1898
*MISC* | 73.35 | 65.08 | 68.97 | 905
*ORG* | 68.27 | 66.50 | 67.37 | 2415
*PER* | 76.99 | 79.20 | 78.08 | 2522
