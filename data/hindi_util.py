from wxconv import WXC
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--format', type=str, help='format of the file text/ssf', required=True)
parser.add_argument('--input', type=str, help='input file to be converted', required=True)
parser.add_argument('--dist', type=int, nargs='+', default=[1, 0, 0], help='train:test_a:test_b')
args = parser.parse_args()
wxc = WXC(order='utf2wx')
if args.format == 'text':
    open('hin.text', 'w').write(wxc.convert(open(args.input).read()))
elif args.format == 'ssf':
    assert len(args.dist) == 3


    def tag_extract(string):
        tag_list = ['PERSON', 'ORGANIZATION', 'LOCATION', 'ENTERTAINMENT', 'FACILITIES', 'ARTIFACT', 'LIVTHINGS',
                    'LOCOMOTIVE', 'PLANTS', 'MATERIALS', 'DISEASE', 'O']
        for tag in tag_list:
            if tag in string:
                return tag


    def write_conll(sentences, output):
        f = open(output, 'w')
        for sentence in sentences:
            for word in sentence:
                f.write(word + '\n')
            f.write('\n')
        f.close()


    sentences = []
    sentence = []
    ner_tag = 'O'
    for line in open(args.input):
        if line.startswith('<Sentence'):
            sentence = []
        elif line.startswith('</Sentence'):
            sentences.append(sentence)
        elif line.startswith('<ENAMEX'):
            ner_tag = tag_extract(line)
        else:
            line = line.split()
            if len(line) == 0:
                continue
            try:
                index = float(line[0])
                if index != int(index):
                    sentence.append(wxc.convert(line[1]) + ' ' + line[2] + ' ' + '.' + ' ' + ner_tag)
            except ValueError:
                pass
            ner_tag = 'O'
    random.shuffle(sentences)
    train = args.dist[0] * len(sentences) // sum(args.dist)
    test_a = args.dist[1] * len(sentences) // sum(args.dist)
    test_b = len(sentences) - train - test_a
    train, test_a, test_b = sentences[0:train], sentences[train:train + test_a], sentences[train + test_a:]
    write_conll(train, 'hin.train')
    write_conll(test_a, 'hin.test_a')
    write_conll(test_b, 'hin.test_b')
