"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os.path
import struct

__EOS_ID = 0
__UNK_ID = 1


def __encode_line(vocab, line, remove_oov):
    if not remove_oov:
        return [vocab.get(word, __UNK_ID) for word in [w for w in line.strip().split()]]
    else:
        return [t for t in [vocab.get(word, -1) for word in [w for w in line.strip().split()]] if t > -1]


def run(vocab, phrase_path, token_path, remove_oov):
    if not os.path.isfile(token_path):
        print("creating token file")
        with open(phrase_path, mode='r') as phrase_file:
            with open(token_path, mode='wb') as tokens_file:
                counter = 0
                for line in phrase_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = __encode_line(vocab, line, remove_oov)
                    tokens.append(__EOS_ID)
                    for token in tokens:
                        tokens_file.write(struct.pack('<i', token))
