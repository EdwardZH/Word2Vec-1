"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os.path
import sys

__max_working_size = int(1e6)
__cull_level = 2
__vocab = {}


def __reduce_vocab():
    global __cull_level
    __cull_level -= 1
    while len(__vocab) > __max_working_size:
        __cull_level += 1
        print("reducing vocab below %d" % __cull_level)
        for k, v in list(__vocab.items()):
            if v < __cull_level:
                del __vocab[k]


def run(phrase_path, vocab_path, max_size):
    if not os.path.isfile(vocab_path):
        print("creating vocab file")
        __vocab["<EOS>"] = sys.maxsize
        with open(phrase_path, mode='r') as phrase_file:
            counter = 0
            for line in phrase_file:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                    if len(__vocab) > __max_working_size:
                        __reduce_vocab()
                for word in [w for w in line.split()]:
                    if word in __vocab:
                        __vocab[word] += 1
                    else:
                        __vocab[word] = 1
            print("finalising vocab")
            vocab_list = sorted(__vocab, key=__vocab.get, reverse=True)
            if len(vocab_list) > max_size:
                vocab_list = vocab_list[:max_size]
            print("writing vocab to file")
            with open(vocab_path, mode='w') as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")
