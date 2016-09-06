import tensorflow as tf
import numpy as np
import functools
import random
import argparse

from input import get_train_data,get_test_data,get_final_data

# from tensorflow.models.rnn import rnn_cell 
# from tensorflow.models.rnn import rnn
# for tensorflow version < 0.9 uncomment the above two lines

WORD_DIM = 311
MAX_SEQ_LEN = 50
NUM_CLASSES = 5
BATCH_SIZE = 128
NUM_HIDDEN = 256
NUM_LAYERS = 3
NUM_EPOCH = 2000

def lazy_property(function):
    attribute = '_' + function.__name__
    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class Model():

    def __init__(self, data, target, dropout, num_hidden, num_layers):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def prediction(self):
        rnn_cell = tf.nn.rnn_cell #comment this line if tensorflow version < 0.9
        fw_cell = rnn_cell.LSTMCell(self._num_hidden,state_is_tuple=True)
        fw_cell = rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout)
        bw_cell = rnn_cell.LSTMCell(self._num_hidden,state_is_tuple=True)
        bw_cell = rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout)

        if self._num_layers > 1:
            fw_cell = rnn_cell.MultiRNNCell([fw_cell] * self._num_layers)
            bw_cell = rnn_cell.MultiRNNCell([bw_cell] * self._num_layers)
        output, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, tf.unpack(tf.transpose(self.data, perm=[1, 0, 2])), dtype=tf.float32, sequence_length=self.length) 
        #output, _, _ = rnn.bidirectional_rnn(fw_cell, bw_cell, tf.unpack(tf.transpose(self.data, perm=[1, 0, 2])), dtype=tf.float32, sequence_length=self.length)
        #uncomment the above line and comment the line above it for tensorflow version < 0.9
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = self._weight_and_bias(2*self._num_hidden, num_classes)
        output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]), [-1, 2*self._num_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction


    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length


    @lazy_property
    def cost(self):
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)


    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(0.003)
        return optimizer.minimize(self.cost)


    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))

        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @staticmethod
    def _weight_and_bias(in_size,out_size):
        weight = tf.truncated_normal([in_size,out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)
    
    @lazy_property
    def getpredf1(self):
        return self.prediction,self.length


def f1(prediction,target,length):
    tp=np.array([0]*(NUM_CLASSES+2))
    fp=np.array([0]*(NUM_CLASSES+2))
    fn=np.array([0]*(NUM_CLASSES+2))

    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)


    for i in range(len(target)):
        for j in range(length[i]):
            if target[i][j] == prediction[i][j]:
                tp[target[i][j]] += 1
            else:
                fp[target[i][j]] += 1
                fn[prediction[i][j]] += 1

    NON_NAMED_ENTITY = 0
    for i in range(NUM_CLASSES):
        if i != NON_NAMED_ENTITY:
            tp[5] += tp[i]
            fp[5] += fp[i]
            fn[5] += fn[i]
        else:
            tp[6] += tp[i]
            fp[6] += fp[i]
            fn[6] += fn[i]

    precision = []
    recall = []
    fscore = []
    for i in range(NUM_CLASSES+2):
        precision.append(tp[i]*1.0/(tp[i]+fp[i]))
        recall.append(tp[i]*1.0/(tp[i]+ fn[i]))
        fscore.append(2.0*precision[i]*recall[i]/(precision[i]+recall[i]))

    #print "precision = " ,precision
    #print "recall = " ,recall
    #print "f1score = " ,fscore
    
    print "Entity fscore :", fscore[5]    
    return fscore[5]


def train(args):

    train_inp, train_out = get_train_data()
    print "train data loaded"
    no_of_batches = len(train_inp) / BATCH_SIZE

    test_inp, test_out = get_test_data()
    print "test data loaded"

    final_inp, final_out = get_final_data()
    print "final data loaded"

    data = tf.placeholder(tf.float32,[None, MAX_SEQ_LEN, WORD_DIM])
    target = tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, NUM_CLASSES])
    dropout = tf.placeholder(tf.float32)
    model = Model(data,target,dropout,NUM_HIDDEN,NUM_LAYERS)
    maximum = 0

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, 'model.ckpt')
            print "last model restored"

        for epoch in range(NUM_EPOCH):
            ptr=0
            for _ in range(no_of_batches):
                batch_inp, batch_out = train_inp[ptr:ptr+BATCH_SIZE], train_out[ptr:ptr+BATCH_SIZE]
                ptr += BATCH_SIZE
                sess.run(model.optimize,{data: batch_inp, target : batch_out, dropout: 0.5})
            if epoch % 10 == 0:
                save_path = saver.save(sess, "model.ckpt")
                print("Model saved in file: %s" % save_path)
                
            pred = sess.run(model.prediction, {data: test_inp, target: test_out, dropout: 1})
            pred,length = sess.run(model.getpredf1, {data: test_inp, target: test_out, dropout: 1})
            print "Epoch:" + str(epoch), "TestA score,"
            m = f1(pred,test_out,length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, "model_max.ckpt")
                print("Max Model saved in file: %s" % save_path)
                pred = sess.run(model.prediction, {data: final_inp, target: final_out, dropout: 1})
                pred,length = sess.run(model.getpredf1, {data: final_inp, target: final_out, dropout: 1})
                print "TestB score,"
                f1(pred,final_out,length)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', type=str, default=None,
                       help="hi")
    args = parser.parse_args()
    train(args)
