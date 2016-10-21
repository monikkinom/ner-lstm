from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
from input import *


class Model:
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])
        fw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * args.num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * args.num_layers, state_is_tuple=True)
        used = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)
        output, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unpack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)
        weight, bias = self.weight_and_bias(2 * args.rnn_size, args.class_size)
        output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]), [-1, 2 * args.rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(0.003)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def f1(args, prediction, target, length):
    tp = np.array([0] * (args.class_size + 1))
    fp = np.array([0] * (args.class_size + 1))
    fn = np.array([0] * (args.class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = args.class_size - 1
    for i in range(args.class_size):
        if i != unnamed_entity:
            tp[args.class_size] += tp[i]
            fp[args.class_size] += fp[i]
            fn[args.class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(args.class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[args.class_size]


def train(args):
    train_inp, train_out = get_train_data()
    test_a_inp, test_a_out = get_test_a_data()
    test_b_inp, test_b_out = get_test_b_data()
    model = Model(args)
    maximum = 0
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, 'model.ckpt')
            print("model restored")
        for e in range(args.epoch):
            for ptr in range(0, len(train_inp), args.batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                          model.output_data: train_out[ptr:ptr + args.batch_size]})
            if e % 10 == 0:
                save_path = saver.save(sess, "model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out})
            print("epoch %d:" % e)
            print('test_a score:')
            m = f1(args, pred, test_a_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, "model_max.ckpt")
                print("max model saved in file: %s" % save_path)
                pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                                                                           model.output_data: test_b_out})
                print("test_b score:")
                f1(args, pred, test_b_out, length)


parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=True)
parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
parser.add_argument('--class_size', type=int, help='number of classes', required=True)
parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--restore', type=str, default=None, help="path of saved model")
train(parser.parse_args())
