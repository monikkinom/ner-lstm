# ner-lstm
Named Entity Recognition using multi-layered bidirectional LSTMs and task adapted word embeddings
---------------------------------------------------------------------------------------------------------------------------------
EMBEDDINGS
Each unique word should have certain number of features, these are called embeddings or also vectors. These are the input features to the neural architecture we are using.

We experimented with different embeddings and observed the results.
Random Vectors, WordVectors made by Google inc and our new generated trigram embeddings.

Papers on WordVectors:
https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf
Good tutorial on WordVectors:https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html
