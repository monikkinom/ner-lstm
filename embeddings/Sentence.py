#!/usr/bin/python3.5
import sys

assert(len(sys.argv) == 3)

sentences = open(sys.argv[1]).read().split('\n')

for i in range(len(sentences)):
    try:
        if len(sentences[i]) == 0:
            del(sentences[i])
    except:
        pass

f = open(sys.argv[2], 'w')

assert(len(sentences)%3 == 0)

for i in range(0, len(sentences), 3):
    words = sentences[i].split()
    pos = sentences[i+1].split()
    chunk = sentences[i+2].split()

    if words[-1] != '.':
        words.append('.')
        pos.append('.')
        chunk.append('O')

    assert(len(words) == len(pos) == len(chunk))

    for i in range(len(words)):
        f.write(words[i]+' '+pos[i]+' '+chunk[i]+' -\n')
    f.write('\n')
