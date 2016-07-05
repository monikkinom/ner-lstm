# ner-lstm
####Named Entity Recognition using multi-layered bidirectional LSTMs and task adapted word embeddings

[Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) is a classification problem of identifying the names of people,organisations,etc (different classes) from the given text corpus. 

Previous approaches to the problems have involved the usage of hand crafted language specific features, CRF and HMM based models, gazetteers, etc. Growing interest in deep learning has led to application of deep neural networks to the existing problems like that of NER. 

We have implemented a 2 layer bidirectional LSTM network using tensorflow to classify the named entities for [CoNNL 2003 NER Shared Task](https://en.wikipedia.org/wiki/Named-entity_recognition). 

EMBEDDINGS
Each unique word should have certain number of features, these are called embeddings or also vectors. These are the input features to the neural architecture we are using.

We experimented with different embeddings and observed the results.
Random Vectors, WordVectors made by Google inc and our new generated trigram embeddings.

Papers on WordVectors:
https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf
Good tutorial on WordVectors:https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html
