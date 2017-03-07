"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf

embedding_size = 128


def load_vocab(vocab_path):
    print("initialising vocab")
    rev_vocab = []
    with open(vocab_path, mode='r') as vocab_file:
        rev_vocab.extend(vocab_file.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab


def main(_):
    data_path = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "model")
    vocab_path = os.path.join(data_path, "vocab.txt")

    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    csv = os.path.join(data_path, "classes.csv")
    df = pd.read_csv(csv)
    class_name = df.columns.values.tolist()[1:]
    no_of_classes = len(class_name)
    data = df.as_matrix()

    xs = []
    exclude = []
    for row in data:
        exclude.append(row[0])
        xs.append(vocab.get(row[0], -1))
    ys = (data[:, 1:])
    no_of_rows = len(xs)
    y_negs = np.array([np.array([0, 0, 0, 0, 0, 0, 0, 1]), ] * no_of_rows)
    ys = np.concatenate((ys, y_negs), axis=0)

    keep_prob = tf.Variable(0.5, trainable=False)
    embeddings = tf.Variable(tf.zeros([vocab_size, embedding_size]), trainable=False, name="embeddings")

    x_id = tf.placeholder(tf.int32, shape=[None])
    x = tf.nn.embedding_lookup(embeddings, x_id)
    y_ = tf.placeholder(tf.int32, shape=[None, no_of_classes])

    w1 = tf.Variable(tf.truncated_normal([embedding_size, embedding_size]))
    b1 = tf.Variable(tf.zeros([embedding_size]))
    y1 = tf.nn.dropout(tf.sigmoid(tf.matmul(x, w1) + b1), keep_prob)

    w2 = tf.Variable(tf.truncated_normal([embedding_size, no_of_classes]))
    b2 = tf.Variable(tf.zeros([no_of_classes]))
    y2 = tf.nn.softmax(tf.matmul(y1, w2) + b2)

    loss = tf.losses.log_loss(y_, y2)
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        saver = tf.train.Saver({"embeddings": embeddings})
        saver.restore(session, latest_checkpoint)

        for i in range(25000):
            x_negs = np.random.randint(vocab_size, size=no_of_rows)
            avg_loss = session.run([train_op, loss], {x_id: np.concatenate((xs, x_negs), axis=0), y_: ys})[1]
            print(avg_loss / no_of_rows)

        candidate = {}
        session.run(tf.assign(keep_prob, 1.0))
        for word in vocab:
            word_id = vocab.get(word, -1)
            vals = session.run(y2, {x_id: [word_id]})[0]
            best_val = vals[:no_of_classes - 1].argmax(axis=0)
            if vals[best_val] > 0.85:
                if word not in exclude:
                    candidate[class_name[best_val] + ": " + word] = vals[best_val]
    candidate_list = sorted(candidate, key=candidate.get, reverse=True)
    candidate_list = candidate_list[:20]

    for item in candidate_list:
        print(item)


if __name__ == "__main__":
    tf.app.run()
