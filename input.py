from __future__ import print_function
import pickle


def get_train_data():
    emb = pickle.load(open('embeddings/train_embed.pkl', 'rb'))
    tag = pickle.load(open('embeddings/train_tag.pkl', 'rb'))
    print('train data loaded')
    return emb, tag


def get_test_a_data():
    emb = pickle.load(open('embeddings/test_a_embed.pkl', 'rb'))
    tag = pickle.load(open('embeddings/test_a_tag.pkl', 'rb'))
    print('test_a data loaded')
    return emb, tag


def get_test_b_data():
    emb = pickle.load(open('embeddings/test_b_embed.pkl', 'rb'))
    tag = pickle.load(open('embeddings/test_b_tag.pkl', 'rb'))
    print('test_b data loaded')
    return emb, tag
