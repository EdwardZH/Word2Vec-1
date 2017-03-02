"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os.path

__threshold = 100
__train_word_count = 0
__max_working_size = int(1e6)
__cull_level = 1
__vocab = {}


def __reduce_vocab():
    global __cull_level
    while len(__vocab) > __max_working_size:
        __cull_level += 1
        print("reducing vocab below %d" % __cull_level)
        for k, v in list(__vocab.items()):
            if v < __cull_level:
                del __vocab[k]
    __cull_level -= 1

def __learn(clean_path):
    global __train_word_count
    print("learning phrases")
    with open(clean_path, mode='r') as clean_file:
        counter = 0
        for line in clean_file:
            counter += 1
            if counter % 100000 == 0:
                print("processing line %d" % counter)
                if len(__vocab) > __max_working_size:
                    __reduce_vocab()
            last_word = None
            for word in [w for w in line.strip().split()]:
                __train_word_count += 1
                if word in __vocab:
                    __vocab[word] += 1
                else:
                    __vocab[word] = 1
                if (last_word is not None) & (last_word != "<NUM>") & (word != "<NUM>"):
                    bigram = last_word + "_" + word
                    if bigram in __vocab:
                        __vocab[bigram] += 1
                    else:
                        __vocab[bigram] = 1
                last_word = word
    __reduce_vocab()


def __save(clean_path, phrase_path):
    global __train_word_count
    print("creating phrase file")
    with open(clean_path, mode='r') as clean_file:
        with open(phrase_path, mode='w') as phrase_file:
            counter = 0
            for line in clean_file:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                word_count = 0
                last_word_count = 0
                bigram_count = 0
                last_word = None
                for word in [w for w in line.strip().split()]:
                    oov = False

                    if word not in __vocab:
                        oov = True
                    else:
                        word_count = __vocab[word]

                    if last_word is not None:
                        bigram = last_word + "_" + word
                        if bigram in __vocab:
                            bigram_count = __vocab[bigram]
                        else:
                            oov = True
                    else:
                        oov = True
                    score = 0

                    if not oov:
                        score = (bigram_count - __cull_level) / last_word_count / word_count * __train_word_count

                    if score > __threshold:
                        phrase_file.write('_' + word)
                    else:
                        phrase_file.write(' ' + word)

                    last_word = word
                    last_word_count = word_count

                phrase_file.write('\n')


def run(clean_path, phrase_path):
    if not os.path.isfile(phrase_path):
        __learn(clean_path)
        __save(clean_path, phrase_path)
