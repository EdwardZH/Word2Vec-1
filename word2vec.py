"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.2 and TensorFlow_gpu-0.12
"""

import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE

import data_utils

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_string("save_path", "model.ckpt", "Base name of checkpoint files.")
flags.DEFINE_string("train_file", "corpus.txt", "Name of the training data file.")
flags.DEFINE_boolean("remove_oov", True, "Remove out of vocabulary word labels from training.")
flags.DEFINE_integer("batch_size", 128, "Number of training examples each step processes.")
flags.DEFINE_integer("embedding_size", 128, "Embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 4, "Number of epochs to train.")
flags.DEFINE_integer("min_occurrence", 8, "Minimum number of times a word should occur in vocabulary.")
flags.DEFINE_integer("max_vocab", 65536, "Maximum size of the vocabulary.")
flags.DEFINE_integer("num_neg_samples", 8, "Negative samples per training example.")
flags.DEFINE_integer("plot_count", 256, "Number of items to plot on TSNE graph.")
flags.DEFINE_integer("window_size", 4, "Number of words to predict to the left and right of the target word.")
FLAGS = flags.FLAGS


def plot_tsne(final_embeddings, rev_vocab, tsne_path):
    tsne = TSNE(init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(final_embeddings[:FLAGS.plot_count, :])
    labels = [rev_vocab[i] for i in range(FLAGS.plot_count)]
    plt.figure(figsize=(19.2, 10.8))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(tsne_path)


def main(_):
    data_path = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "model")
    checkpoint = os.path.join(model_path, FLAGS.save_path)
    tsne_path = os.path.join(model_path, "tsne.png")
    rev_vocab = data_utils.prepare(data_path, FLAGS.min_occurrence, FLAGS.max_vocab, FLAGS.remove_oov, FLAGS.train_file)
    vocab_size = len(rev_vocab)

    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * FLAGS.window_size * 2])
        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * FLAGS.window_size * 2, 1])
        embeddings = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0))
        input_embeddings = tf.nn.embedding_lookup(embeddings, inputs)
        nce_weights = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))
        nce_loss = tf.nn.nce_loss(nce_weights, nce_biases, input_embeddings, labels, FLAGS.num_neg_samples, vocab_size)
        loss = tf.reduce_mean(nce_loss)
        learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(FLAGS.epochs_to_train):
            lr = (FLAGS.learning_rate * FLAGS.epochs_to_train - epoch) / FLAGS.epochs_to_train
            session.run(tf.assign(learning_rate, lr))
            print("\nepoch %d" % epoch)
            avg_loss = 0
            batch = 0
            start = time.time()
            for batch_inputs, batch_labels in data_utils.iterator(data_path, FLAGS.batch_size, FLAGS.window_size):
                feed_dict = {inputs: batch_inputs, labels: batch_labels}
                avg_loss += session.run([optimizer, loss], feed_dict=feed_dict)[1]
                batch += 1
                if batch % 1000 == 0:
                    current = time.time()
                    avg_loss /= 1000
                    rate = 0.001 * batch * FLAGS.batch_size / (current - start)
                    print("batch: %d  loss: %.2f  words/sec: %.2fk" % (batch, avg_loss, rate))
                    avg_loss = 0
            tf.train.Saver().save(session, checkpoint)
        final_embeddings = normalized_embeddings.eval()
    plot_tsne(final_embeddings, rev_vocab, tsne_path)


if __name__ == "__main__":
    tf.app.run()
