"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os.path

__EOS = "<EOS>"
__UNK = "<UNK>"
__NUM = "<NUM>"
__START_VOCAB = [__EOS, __UNK, __NUM]


def __reduce_vocab(vocab, threshold):
    print("reducing vocab")
    for k, v in list(vocab.items()):
        if v < threshold:
            del vocab[k]


def run(phrase_path, vocab_path, min_occurrence, max_size, remove_oov):
    if not os.path.isfile(vocab_path):
        print("creating vocab file")
        vocab = {}
        max_working_size = int(1e6)
        with open(phrase_path, mode='r') as phrase_file:
            counter = 0
            for line in phrase_file:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                for word in [w for w in line.split()]:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                if len(vocab) > max_working_size:
                    __reduce_vocab(vocab, min_occurrence)
            if __NUM in vocab:
                del vocab[__NUM]
            __reduce_vocab(vocab, min_occurrence)
            print("finalising vocab")
            if remove_oov:
                vocab_list = sorted(vocab, key=vocab.get, reverse=True)
            else:
                vocab_list = __START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_size:
                vocab_list = vocab_list[:max_size]
            print("writing vocab to file")
            with open(vocab_path, mode='w') as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")
