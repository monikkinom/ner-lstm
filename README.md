This Repository contains the code which implements the approach described in the following Arxiv Preprint: https://arxiv.org/abs/1610.09756
which is published in ICON-16 conference.

# ner-lstm

### Requirements

* tensorflow
* http://github.com/ltrc/indic-wx-converter (only for hindi)
* gensim 

### Named Entity Recognition using multi-layered bidirectional LSTMs and task adapted word embeddings

[Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition) is a classification problem of identifying the names of people,organisations,etc (different classes) in a text corpus. 
Previous approaches to the problems have involved the usage of hand crafted language specific features, CRF and HMM based models, gazetteers, etc. Growing interest in deep learning has led to application of deep neural networks to the existing problems like that of NER. 
We have implemented a 2 layer bidirectional LSTM network using tensorflow to classify the named entities for [CoNNL 2003 NER Shared Task](http://www.cnts.ua.ac.be/conll2003/ner/). 
Classification on the NER Hindi dataset of [icon-2013](http://ltrc.iiit.ac.in/icon/2013/) proceedings was also done.
The process and code usage are given below. All codes use argparse for intuitive usage.

### Generating the embedding model

Sentences are used as inputs for the recurrent neural network.
Representation of words in the sentence is via the form of embeddings.
Hence the features for the recurrent neural network are sentences a.k.a sequence of words a.k.a sequence of embeddings.
Each unique word should have certain number of features, these are called embeddings or also vectors. These are the input features to the neural architecture we are using.

###### English

You need a corpus comprising of text separated by only spaces if you are training a model.
First train the model or load from an existing one from the files given in embeddings.
**wordvec_model.py** - creates a model of word2vec, 2 ways to create the model either by supplying a corpus to train or restore from word2vec gensim bin file.
**glove_model.py** - creates a model of glove, 2 ways to create the model either by supplying a corpus to train or restore from a glove vector.txt file. Copy corpus in Glove-1.2 directory and run the code from embeddings folder and give name of corpus as param.
**rnnvec_model.py** - creates a model of LSTM, only way is by supplying a corpus.

###### Hindi
Follow the same steps as english but first convert corpus to english type using **hindi_util.py**.
######Comparison
We have done a comparison between **111 dimension** embedding models by training all of them on a small 100mb corpus and evaluating on the conll ner dataset.

Model | Test_a | Test_b
--- | --- | ---
Word2Vec | 88.33 | 83.40
Glove | 89.62 | 83.10
RnnVec | 81.07 | 75.20

### Preparing the inputs

Now we have the embedding model, we have to use that to convert our sentences of words to sentences of embeddings.
First use **resize_input.py** to resize your data set to a max sentence length.
Use the trained embedding model along with **get_conll_embeddings.py** or **get_icon_embeddings.py** for conll and icon respectively to get the pickled input data ready to be fed and train the recurrent neural network.
Note that we are adding 11 extra features here to the embeddings themselves which include pos, chunk and capital features of the word.

### Deep Neural Network
We have used Google's Tensorflow to implement a bidirectional multilayered rnn cell (LSTM). The hyper parameters are present at the top in main.py. Tweaking the parameters can yield a variety of results which are worth noting.
We have used a softmax layer as the last layer of the network to produce the final classification outputs. We tried working with different optimizers and we found that AdamOptimzer produced the best results.
The function to calculate the F1 Scores, Prediction Accuracy and Recall is also included in model.py. We have also included the ability to save/restore an existing model using tensorflow's saver functions.
The path of the generated pickle file from above needs to be set in **input.py**.
Use the **model.py** to run the deep neural network which will start running and optimizing the F1 scores.

### Final Results

Dataset | Model | Embedding size | Test_a | Test_b
--- | --- | --- | --- | ---
**CONLL** | Glove | 311 | 93.99 | 90.32
**CONLL** | Word2Vec | 311 | 93.5 | 89.4
**ICON** | Glove | 311 | 78.6 | 77.48 

### CONLL samples

A sample result produced by conll eval script is presented here.

###### Word2Vec 311 dimensions

* Test_a

processed 49644 tokens with 8211 phrases; found: 8080 phrases; correct: 7619.
Accuracy = 98.54%

Class | Precision | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 94.29 | 92.79 | 93.54 | 8080
*LOC* | 94.86 | 94.25 | 94.56 | 2023
*MISC* | 91.99 | 85.09 | 88.40 | 1123
*ORG* | 91.98 | 89.76 | 90.86 | 2020
*PER* | 96.40 | 97.16 | 96.78 | 2914

* Test_b

processed 45151 tokens with 7719 phrases; found: 7740 phrases; correct: 6911.
Accuracy = 97.49%

Class | Precision | Recall | FB1 | Numbers
--- | --- | --- | --- | ---
*NER* | 89.29 | 89.53 | 89.41 | 7740
*LOC* | 89.67 | 90.87 | 90.27 | 1898
*MISC* | 75.80 | 75.80 | 75.80 | 905
*ORG* | 88.03 | 87.27 | 87.65 | 2415
*PER* | 95.04 | 95.69 | 95.37 | 2522

### Papers on WordVectors

https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
http://www-personal.umich.edu/~ronxin/pdf/w2vexp.pdf
Good tutorial on WordVectors: https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html
