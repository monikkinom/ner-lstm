from __future__ import print_function
import numpy as np
import pickle as pkl
import sys
import argparse
from wordvec_model import WordVec
from glove_model import GloveVec
from rnnvec_model import RnnVec


def find_max_length(file_name):
    temp_len = 0
    max_length = 0
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if temp_len > max_length:
                max_length = temp_len
            temp_len = 0
        else:
            temp_len += 1
    return max_length


def pos(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def chunk(tag):
    one_hot = np.zeros(5)
    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])


def get_input(model, word_dim, input_file, output_embed, output_tag, sentence_length=-1):
    print('processing %s' % input_file)
    word = []
    tag = []
    sentence = []
    sentence_tag = []
    if sentence_length == -1:
        max_sentence_length = find_max_length(input_file)
    else:
        max_sentence_length = sentence_length
    sentence_length = 0
    print("max sentence length is %d" % max_sentence_length)
    for line in open(input_file):
        if line in ['\n', '\r\n']:
            for _ in range(max_sentence_length - sentence_length):
                tag.append(np.array([0] * 12))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            sentence_length = 0
            word = []
            tag = []
        else:
            assert (len(line.split()) == 4)
            sentence_length += 1
            temp = model[line.split()[0]]
            assert len(temp) == word_dim
            temp = np.append(temp, pos(line.split()[1]))  # adding pos embeddings
            temp = np.append(temp, chunk(line.split()[2]))  # adding chunk embeddings
            temp = np.append(temp, capital(line.split()[0]))  # adding capital embedding
            word.append(temp)
            t = line.split()[3]
            # Five classes 0-None,1-Person,2-Location,3-Organisation,4-Misc
            if t == 'PERSON':
                tag.append(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            elif t == 'ORGANIZATION':
                tag.append(np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            elif t == 'LOCATION':
                tag.append(np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            elif t == 'ENTERTAINMENT':
                tag.append(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
            elif t == 'FACILITIES':
                tag.append(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
            elif t == 'ARTIFACT':
                tag.append(np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))
            elif t == 'LIVTHINGS':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
            elif t == 'LOCOMOTIVE':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
            elif t == 'PLANTS':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
            elif t == 'MATERIALS':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
            elif t == 'DISEASE':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]))
            elif t == 'O':
                tag.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
            else:
                print(t)
                print("error in input")
                sys.exit(0)
    assert (len(sentence) == len(sentence_tag))
    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='train file location', required=True)
    parser.add_argument('--test_a', type=str, help='test_a file location', required=True)
    parser.add_argument('--test_b', type=str, help='test_b location', required=True)
    parser.add_argument('--sentence_length', type=int, default=-1, help='max sentence length')
    parser.add_argument('--use_model', type=str, help='model location', required=True)
    parser.add_argument('--model_dim', type=int, help='model dimension of words', required=True)
    args = parser.parse_args()
    trained_model = pkl.load(open(args.use_model, 'rb'))
    get_input(trained_model, args.model_dim, args.train, 'train_embed.pkl', 'train_tag.pkl',
              sentence_length=args.sentence_length)
    get_input(trained_model, args.model_dim, args.test_a, 'test_a_embed.pkl', 'test_a_tag.pkl',
              sentence_length=args.sentence_length)
    get_input(trained_model, args.model_dim, args.test_b, 'test_b_embed.pkl', 'test_b_tag.pkl',
              sentence_length=args.sentence_length)
