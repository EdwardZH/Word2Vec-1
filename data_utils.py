"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import collections
import os.path
import struct
import numpy as np
import random
import clean_builder
import vocab_builder
import token_builder
import phrase_builder


class Iterator(object):
    def __init__(self, token_path, batch_size, window):
        self.token_path = token_path
        self.batch_size = batch_size
        self.chunk_size = batch_size * 4
        self.window = window
        self.input_buffer = collections.deque(maxlen=2 * self.window + 1)

    def __iter__(self):
        inputs = np.ndarray(shape=self.batch_size * self.window, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size * self.window, 1), dtype=np.int32)
        with open(self.token_path, mode='rb') as tokens_file:
            for _ in range(2 * self.window + 1):
                self.input_buffer.append(struct.unpack('<i', tokens_file.read(4))[0])
            while True:
                chunk = tokens_file.read(self.chunk_size)
                if len(chunk) < self.chunk_size:
                    break
                for i in range(self.batch_size):
                    j = 0
                    offset = random.randint(0, self.window)
                    for t in range(self.window + 1):
                        target = t + offset
                        if target != self.window:
                            inputs[i * self.window + j] = self.input_buffer[self.window]
                            labels[i * self.window + j, 0] = self.input_buffer[target]
                            j += 1
                    self.input_buffer.append(struct.unpack('<i', chunk[i * 4:i * 4 + 4])[0])
                yield inputs, labels


def __load_vocab(vocab_path):
    print("initialising vocab")
    rev_vocab = []
    with open(vocab_path, mode='r') as vocab_file:
        rev_vocab.extend(vocab_file.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab


def prepare(data_path, max_vocab_size, corpus_file):
    corpus_path = os.path.join(data_path, corpus_file)
    clean_path = os.path.join(data_path, "clean.txt")
    phrase_path = os.path.join(data_path, "phrase.txt")
    vocab_path = os.path.join(data_path, "vocab.txt")
    token_path = os.path.join(data_path, "token.bin")
    clean_builder.run(corpus_path, clean_path)
    phrase_builder.run(clean_path, phrase_path)
    vocab_builder.run(phrase_path, vocab_path, max_vocab_size)
    vocab, rev_vocab = __load_vocab(vocab_path)
    token_builder.run(vocab, phrase_path, token_path)
    return vocab, rev_vocab
