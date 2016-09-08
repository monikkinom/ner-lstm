import numpy as np, sys
import tensorflow as tf
import collections, pickle as pkl
from time import time

def linear(x):
    return x

LOOK = 4
VSIZE = 50000
HIDDEN = 300
BATCHSIZE = 2048
NUMSAMPLED = 1000
LEARNRATE = 1.0
RNNACT = tf.nn.tanh
TRUNC = 4
PATH = '../../Corpus/tf.txt'

def rnnCell(xU, s, W, rnnbias):
    hs = xU+rnnbias+tf.matmul(s, W)
    out = RNNACT(hs)
    return [out, out]

def rnnLook(x, s, U, W, rnnbias):
    out = []
    for i in range(TRUNC):
        xU = tf.nn.embedding_lookup(U, x[:, i])
        o, s = rnnCell(xU, s, W, rnnbias)
        out.append(o)
    O = tf.pack(out, axis = 1)
    return [O, s]

def rnnIter(X, Y, epoch = 1):
    trunc = TRUNC
    x = tf.placeholder(tf.int32, shape = [None, trunc])
    y = tf.placeholder(tf.int32, shape = [None, trunc, Y.shape[2]])
    s = tf.placeholder(tf.float32, shape = [None, HIDDEN])
    if RNNACT == tf.nn.tanh:
        U = tf.Variable(tf.truncated_normal([VSIZE, HIDDEN], stddev = 1.0/np.sqrt(VSIZE)))
        W = tf.Variable(tf.truncated_normal([HIDDEN, HIDDEN], stddev = 1.0/np.sqrt(HIDDEN)))
    else:
        U = tf.Variable(tf.truncated_normal([VSIZE, HIDDEN], stddev = 0.01))
        W = tf.Variable(tf.truncated_normal([HIDDEN, HIDDEN], stddev = 0.01))
    rnnbias = tf.Variable(tf.constant(0.1, shape = [HIDDEN]), dtype = tf.float32) 
    V = tf.Variable(tf.truncated_normal([VSIZE, HIDDEN], stddev = 1.0/np.sqrt(HIDDEN)))
    bias = tf.Variable(tf.constant(0.0, shape = [VSIZE]), dtype = tf.float32)
    O, resState = rnnLook(x, s, U, W, rnnbias)
    nce_loss = tf.nn.sampled_softmax_loss(V, bias, tf.reshape(O, [-1, HIDDEN]), tf.reshape(y, [-1, Y.shape[2]]), NUMSAMPLED, VSIZE, Y.shape[2])
    cost = tf.reduce_mean(nce_loss)
    train = tf.train.GradientDescentOptimizer(LEARNRATE).minimize(cost)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()
    NO = (X.shape[0]+BATCHSIZE-1)/BATCHSIZE
    NOS = (X.shape[1]+trunc-1)/trunc
    if len(sys.argv) > 1:
        try:
            saver.restore(sess, sys.argv[1])
            print("Restored")
        except:
            pass
    for itr in range(epoch):
        totloss = 0
        for i in range(0, X.shape[0], BATCHSIZE):
            binp = X[i:i+BATCHSIZE]
            bout = Y[i:i+BATCHSIZE]
            initstate = np.zeros((binp.shape[0], HIDDEN))
            batloss = 0
            for j in range(0, X.shape[1], trunc):
                feed_dict = {x:binp[:, j:j+trunc], y:bout[:, j:j+trunc, :], s:initstate}
                _, loss, initstate = sess.run([train, cost, resState], feed_dict)
                batloss += loss
            totloss += batloss
            print("\rAverage loss in batch %d/%d = %f..." % (i//BATCHSIZE+1, NO, batloss/NOS), end = "")
        if len(sys.argv) > 1:
            saver.save(sess, sys.argv[1])
        else:
            saver.save(sess, "model%s.ckpt" % (time()))
        print("Average loss in epoch %d = %f" % (itr+1, totloss/NO))
    return sess.run(U)

def createData(SENT):
    data = open(PATH).read().split()
    vocab = {}
    count = [['UNK', 0], ['END', 0]]
    count.extend(collections.Counter(data).most_common(VSIZE-2))
    for word, _ in count:
        vocab[word] = len(vocab)
    inp = np.zeros((len(data)), dtype = np.int32)
    i = 0
    for word in data:
        inp[i] = vocab.get(word, 0)
        i += 1 
    NO = (inp.shape[0]+SENT-1)//SENT
    NO = NO*SENT-inp.shape[0]
    inp = np.append(inp, [1]*NO)
    flatinp = inp.reshape(-1)
    out = np.ones((inp.shape[0], 2*LOOK), dtype = np.int32)
    for i in range(inp.shape[0]):
        bound = max(0, i-LOOK)
        behind = flatinp[bound:i]
        out[i, LOOK:LOOK+behind.shape[0]] = behind
        ahead = flatinp[i+1:i+1+LOOK]
        out[i, 0:ahead.shape[0]] = ahead
    inp = inp.reshape((-1, SENT))
    out = out.reshape((-1, SENT, 2*LOOK))
    pkl.dump(vocab, open('vocab.pkl', 'wb'))
    np.save('inp.npy', inp)
    np.save('out.npy', out)

class WVec:
    def __init__(self, vocab, vec):
        assert len(vocab) == len(vec)
        self.vocab = vocab
        self.vec = vec

    def __getitem__(self, word):
        word = word.lower()
        return self.vec[self.vocab[word]]

    def cosineDistance(self, word1, word2):
        word1, word2 = word1.lower(), word2.lower()
        vec1, vec2 = self.vocab[word1], self.vocab[word2]
        vec1, vec2 = self.vec[vec1], self.vec[vec2]
        norm = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        return np.dot(vec1, vec2)/norm

    def mostSimilar(self, word, no = 10):
        word = word.lower()
        wordtup = []
        for attr in self.vocab:
            if attr != 'UNK':
                r = self.cosineDistance(word, attr)
                wordtup.append([attr, r])
        return sorted(wordtup, key = lambda x:abs(x[1]), reverse = True)[0:no]

def main():
    try:
        vocab = pkl.load(open('./data_rnn/vocab.pkl', 'rb'))
        X = np.load('./data_rnn/inp.npy')
        Y = np.load('./data_rnn/out.npy')
        print("Loaded data...")
    except:
        createData()
        vocab = pkl.load(open('./data_rnn/vocab.pkl', 'rb'))
        X = np.load('./data_rnn/inp.npy')
        Y = np.load('./data_rnn/out.npy')
        print("Created data...")
    global LEARNRATE
    if len(sys.argv) > 2:
        LEARNRATE = float(sys.argv[2])
    wvec = WVec(vocab, rnnIter(X, Y, 1))
    pkl.dump(wvec, open('Vec.pkl', 'wb'))

if __name__ == '__main__':
    main()
