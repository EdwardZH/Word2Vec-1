"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.2 and TensorFlow_gpu-0.12
"""
import collections
import os.path
import re
import struct

import numpy as np

_RE_APOSTROPHE = re.compile("(')")
_RE_BREAK = re.compile("[.?!]")
_RE_DIACRITICS_A = re.compile("([àáâãäå])")
_RE_DIACRITICS_C = re.compile("(ç)")
_RE_DIACRITICS_E = re.compile("([èéêë])")
_RE_DIACRITICS_I = re.compile("([ìíîï])")
_RE_DIACRITICS_N = re.compile("(ñ)")
_RE_DIACRITICS_O = re.compile("([òóôõöø])")
_RE_DIACRITICS_U = re.compile("([ùúûü])")
_RE_DIACRITICS_Y = re.compile("([ýÿ])")
_RE_DIGIT = re.compile("[0-9]+")
_RE_MULTI_SPACE = re.compile("\s\s+")
_RE_NON_ALPHA = re.compile("([^a-z0-9 ])")

_EOS = "<EOS>"
_UNK = "<UNK>"
_NUM = "<NUM>"
_START_VOCAB = [_EOS, _UNK, _NUM]

EOS_ID = 0
UNK_ID = 1
NUM_ID = 2


class Iterator(object):
    def __init__(self, token_path, batch_size, window):
        self.token_path = token_path
        self.batch_size = batch_size
        self.chunk_size = batch_size * 4
        self.window = window
        self.span = 2 * window + 1
        self.input_buffer = collections.deque(maxlen=self.span)

    def __iter__(self):
        inputs = np.ndarray(shape=self.batch_size * self.window * 2, dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size * self.window * 2, 1), dtype=np.int32)
        with open(self.token_path, 'rb') as tokens_file:
            for _ in range(self.span):
                self.input_buffer.append(struct.unpack('<i', tokens_file.read(4))[0])
            while True:
                chunk = tokens_file.read(self.chunk_size)
                if len(chunk) < self.chunk_size:
                    break
                for i in range(self.batch_size):
                    j = 0
                    for target in range(self.span):
                        if target != self.window:
                            inputs[i * self.window * 2 + j] = self.input_buffer[self.window]
                            labels[i * self.window * 2 + j, 0] = self.input_buffer[target]
                            j += 1
                    self.input_buffer.append(struct.unpack('<i', chunk[i * 4:i * 4 + 4])[0])
                yield inputs, labels


def clean_text(text):
    text = text.lower()
    text = _RE_APOSTROPHE.sub("", text)
    text = _RE_DIACRITICS_A.sub("a", text)
    text = _RE_DIACRITICS_C.sub("c", text)
    text = _RE_DIACRITICS_E.sub("e", text)
    text = _RE_DIACRITICS_I.sub("i", text)
    text = _RE_DIACRITICS_N.sub("n", text)
    text = _RE_DIACRITICS_O.sub("o", text)
    text = _RE_DIACRITICS_U.sub("u", text)
    text = _RE_DIACRITICS_Y.sub("y", text)
    text = _RE_NON_ALPHA.sub(" ", text)
    text = _RE_DIGIT.sub(" <NUM> ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def create_vocabulary(corpus_path, clean_path, vocab_path, min_occurrence, max_size, remove_oov):
    max_working_size = int(1e6)
    if not os.path.isfile(vocab_path):
        print("creating vocab")
        vocab = {}
        with open(corpus_path, 'r') as corpus_file:
            with open(clean_path, 'w') as clean_file:
                counter = 0
                for line in corpus_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    for sub_line in _RE_BREAK.split(line):
                        sub_line = clean_text(sub_line)
                        if len(sub_line) > 0:
                            clean_file.write(sub_line + "\n")
                            for word in [w for w in sub_line.split()]:
                                if word in vocab:
                                    vocab[word] += 1
                                else:
                                    vocab[word] = 1
                    if len(vocab) > max_working_size:
                        reduce_vocab(vocab, min_occurrence)
                if _NUM in vocab:
                    del vocab[_NUM]
                reduce_vocab(vocab, min_occurrence)
                print("finalising vocabulary")
                if remove_oov:
                    vocab_list = sorted(vocab, key=vocab.get, reverse=True)
                else:
                    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
                if len(vocab_list) > max_size:
                    vocab_list = vocab_list[:max_size]
                print("writing vocabulary to file")
                with open(vocab_path, 'w') as vocab_file:
                    for w in vocab_list:
                        vocab_file.write(w + "\n")


def reduce_vocab(vocab, threshold):
    print("reducing vocabulary")
    for k, v in list(vocab.items()):
        if v < threshold:
            del vocab[k]


def initialise_vocabulary(vocab_path):
    print("initialising vocab")
    rev_vocab = []
    with open(vocab_path, 'r') as vocab_file:
        rev_vocab.extend(vocab_file.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab


def words_to_tokens(vocab, clean_path, token_path, remove_oov):
    if not os.path.isfile(token_path):
        print("creating tokens")
        with open(clean_path, 'r') as data_file:
            with open(token_path, 'wb') as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenising line %d" % counter)
                    tokens = encode_line(vocab, line, remove_oov)
                    tokens.append(EOS_ID)
                    for token in tokens:
                        tokens_file.write(struct.pack('<i', token))


def encode_line(vocab, line, remove_oov):
    if not remove_oov:
        return [vocab.get(word, UNK_ID) for word in [w for w in line.strip().split()]]
    else:
        return [t for t in [vocab.get(word, -1) for word in [w for w in line.strip().split()]] if t > -1]


def prepare(data_path, min_vocab_occurrence, max_vocab_size, remove_oov, corpus_file):
    corpus_path = os.path.join(data_path, corpus_file)
    clean_path = os.path.join(data_path, "clean.txt")
    vocab_path = os.path.join(data_path, "vocab.txt")
    token_path = os.path.join(data_path, "tokens.bin")
    create_vocabulary(corpus_path, clean_path, vocab_path, min_vocab_occurrence, max_vocab_size, remove_oov)
    vocab, rev_vocab = initialise_vocabulary(vocab_path)
    words_to_tokens(vocab, clean_path, token_path, remove_oov)
    return vocab, rev_vocab
