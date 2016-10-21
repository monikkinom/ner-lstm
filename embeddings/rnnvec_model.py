from __future__ import print_function
import numpy as np
import tensorflow as tf
from random_vec import RandomVec
from collections import Counter
import pickle as pkl
import argparse


class RnnVec:
    def __init__(self, args):
        self.rand_model = RandomVec(args.rnn_size)
        corpus = open(args.corpus, 'r').read().lower().split()
        word_count = [['unk', 0]]
        word_count.extend(Counter(corpus).most_common(args.vocab_size - 1))
        word_count.sort(key=lambda x: x[1], reverse=True)
        self.vocab = {}
        for word, freq in word_count:
            self.vocab[word] = len(self.vocab)
        print('vocabulary built')
        int_corpus = []
        for word in corpus:
            int_corpus.append(self.vocab.get(word, 0))
        input_data = np.zeros(len(int_corpus))
        output_data = np.zeros((len(int_corpus), 2 * args.window))
        for i in range(len(int_corpus)):
            input_data[i] = int_corpus[i]
            past = int_corpus[i - args.window:i]
            future = int_corpus[i + 1:i + 1 + args.window]
            total = past + future
            total.extend([0] * (2 * args.window - len(total)))
            output_data[i] = total
        print('data formed')
        del corpus
        del word_count
        del int_corpus
        num_sentence = (len(input_data) + args.sentence_length - 1) // args.sentence_length
        input_data = np.resize(input_data, (num_sentence, args.sentence_length))
        output_data = np.resize(output_data, (num_sentence, args.sentence_length, 2 * args.window))
        print('training')
        cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        input_tensor = tf.placeholder(tf.int32, [args.batch_size, args.sentence_length])
        output_tensor = tf.placeholder(tf.int32, [args.batch_size, args.sentence_length, 2 * args.window])
        initial_state = cell.zero_state(args.batch_size, tf.float32)
        with tf.variable_scope('rnnlm'):
            weights = tf.get_variable("softmax_w", [args.vocab_size, args.rnn_size])
            bias = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, input_tensor)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, args.rnn_size])
        loss = tf.nn.sampled_softmax_loss(weights, bias, output, tf.reshape(output_tensor, [-1, 2 * args.window]), 128,
                                          args.vocab_size, num_true=2 * args.window)
        cost = tf.reduce_mean(loss)
        lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        for e in range(3):
            state = sess.run(initial_state)
            for i in range(0, input_data.shape[0], args.batch_size):
                if i + args.batch_size > input_data.shape[0]:
                    i = input_data.shape[0] - args.batch_size
                _, error, state = sess.run([train_op, cost, last_state],
                                           feed_dict={input_tensor: input_data[i:i + args.batch_size],
                                                      output_tensor: output_data[i:i + args.batch_size],
                                                      initial_state: state,
                                                      lr: .01 * (.3 ** e)})
                print('batch percentage = %f, cost = %f, epoch %d' % (i * 100 / input_data.shape[0], error, e))
        self.embeddings = sess.run(embedding)

    def __getitem__(self, word):
        word = word.lower()
        try:
            return self.embeddings[self.vocab[word]]
        except KeyError:
            return self.rand_model[word]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, help='corpus location', required=True)
    parser.add_argument('--window', type=int, default=5, help='window size')
    parser.add_argument('--vocab_size', type=int, default=50000, help='number of threads')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of training')
    parser.add_argument('--sentence_length', type=int, help='sentence length', required=True)
    parser.add_argument('--rnn_size', type=int, default=128, help='hidden layer size of rnn')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the rnn')
    args = parser.parse_args()
    model = RnnVec(args)
    pkl.dump(model, open('rnnvec_model_' + str(args.rnn_size) + '.pkl', 'wb'))
