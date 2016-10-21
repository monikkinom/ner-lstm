from __future__ import print_function
from gensim.models.word2vec import Word2Vec
from random_vec import RandomVec
import pickle as pkl
import argparse


class WordVec:
    def __init__(self, args):
        print('processing corpus')
        if args.restore is None:
            corpus = open(args.corpus, 'r').read().lower().split()
            sentences = []
            sentence = []
            length = 0
            for word in corpus:
                sentence.append(word)
                length += 1
                if length == args.sentence_length:
                    sentences.append(sentence)
                    sentence = []
                    length = 0
            if length != 0:
                sentences.append(sentence)
            print('training')
            self.wvec_model = Word2Vec(sentences=sentences, size=args.dimension, window=args.window,
                                       workers=args.workers,
                                       sg=args.sg,
                                       batch_words=args.batch_size, min_count=1, max_vocab_size=args.vocab_size)
        else:
            self.wvec_model = Word2Vec.load_word2vec_format(args.restore, binary=True)
        self.rand_model = RandomVec(args.dimension)

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.wvec_model[word]
        except KeyError:
            return self.rand_model[word]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, help='corpus location', required=True)
    parser.add_argument('--dimension', type=int, help='vector dimension', required=True)
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--vocab_size', type=int, help='vocabulary size', required=True)
    parser.add_argument('--workers', type=int, default=3, help='number of threads')
    parser.add_argument('--sg', type=int, default=1, help='if skipgram 1 if cbow 0')
    parser.add_argument('--batch_size', type=int, default=10000, help='batch size of training')
    parser.add_argument('--sentence_length', type=int, help='sentence length', required=True)
    parser.add_argument('--restore', type=str, default=None, help='word2vec format save')
    args = parser.parse_args()
    model = WordVec(args)
    pkl.dump(model, open('wordvec_model_' + str(args.dimension) + '.pkl', 'wb'))
