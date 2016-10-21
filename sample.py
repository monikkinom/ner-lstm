from model import Model
import argparse
import tensorflow as tf
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=True)
parser.add_argument('--sentence_length', type=int, help='max sentence length', rquired=True)
parser.add_argument('--class_size', type=int, help='number of classes', required=True)
parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--input_embed', type=str, help='location of input pickle embedding', required=True)
parser.add_argument('--restore', type=str, help="path of saved model", required=True)
args = parser.parse_args()
model = Model(args)
inp = pkl.load(open(args.input_embed, 'rb'))
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, args.restore)
pred = sess.run(model.prediction, {model.input_data: inp})
pkl.dump(pred, open('predictions.npy', 'wb'))
