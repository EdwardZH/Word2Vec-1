"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""

import os
import threading
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import decomposition

import data_utils

epoch = 0

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_string("save_path", "model.ckpt", "Base name of checkpoint files.")
flags.DEFINE_string("train_file", "corpus.txt", "Name of the training data file.")
flags.DEFINE_boolean("plot", True, "Set to true to plot example pca graph after training.")
flags.DEFINE_boolean("run_training", True, "Set to false to bypass training and query from saved model.")
flags.DEFINE_boolean("remove_oov", True, "Remove out of vocabulary word labels from training.")
flags.DEFINE_integer("batch_size", 512, "Number of training examples each step processes.")
flags.DEFINE_integer("embedding_size", 128, "Embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 4, "Number of epochs to train.")
flags.DEFINE_integer("min_occ", 8, "Minimum number of times a word should occur in vocabulary.")
flags.DEFINE_integer("max_vocab", 65536, "Maximum size of the vocabulary.")
flags.DEFINE_integer("num_neg_samples", 8, "Negative samples per training example.")
flags.DEFINE_integer("window_size", 4, "Number of words to predict to the left and right of the target word.")
FLAGS = flags.FLAGS


def plot_graph(embeddings, labels, path):
    pca = decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    values = pca.transform(embeddings)
    plt.figure(figsize=(19.2, 10.8))
    for i, label in enumerate(labels):
        x, y = values[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.savefig(path)


def load(session, data_path, close_op, enqueue_op, queue_inputs, queue_labels):
    global epoch
    token_path = os.path.join(data_path, "token.bin")
    for epoch in range(FLAGS.epochs_to_train):
        for batch_inputs, batch_labels in iter(data_utils.Iterator(token_path, FLAGS.batch_size, FLAGS.window_size)):
            session.run(enqueue_op, feed_dict={queue_inputs: batch_inputs, queue_labels: batch_labels})
    session.run(close_op)


def main(_):
    global epoch
    data_path = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    plot_path = os.path.join(model_path, "plot.png")
    checkpoint = os.path.join(model_path, FLAGS.save_path)
    vocab, rev_vocab = data_utils.prepare(data_path, FLAGS.min_occ, FLAGS.max_vocab, FLAGS.remove_oov, FLAGS.train_file)
    vocab_size = len(vocab)

    # Add queue ops to graph.
    q_inputs = tf.placeholder(tf.int32, shape=(FLAGS.batch_size * FLAGS.window_size * 2), name="inputs")
    q_labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size * FLAGS.window_size * 2, 1), name="labels")
    queue = tf.FIFOQueue(50, [tf.int32, tf.int32], name="fifo_queue")
    close_op = queue.close()
    enqueue_op = queue.enqueue([q_inputs, q_labels])

    # Add training ops to graph.
    inputs, labels = queue.dequeue()
    embeddings = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0), name="embeddings")
    lookup = tf.nn.embedding_lookup(embeddings, inputs, name="lookup")
    weights = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0), name="nce_weights")
    biases = tf.Variable(tf.zeros([vocab_size]), name="biases")
    nce_loss = tf.nn.nce_loss(weights, biases, labels, lookup, FLAGS.num_neg_samples, vocab_size, name="nce_loss")
    loss = tf.reduce_mean(nce_loss)
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False, name="learning_rate")
    global_step = tf.Variable(0, trainable=False, name="global_step")
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Add query ops to graph.
    query_id = tf.placeholder(tf.int32, shape=[1], name="query_id")
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    norm_embeddings = embeddings / norm
    query_embedding = tf.gather(norm_embeddings, query_id)
    cosine_similarity = tf.matmul(query_embedding, norm_embeddings, transpose_b=True)
    top_items = tf.nn.top_k(cosine_similarity, 8)[1]
    result_embedding = tf.gather(norm_embeddings, top_items)

    with tf.Session() as session:
        latest_checkpoint = tf.train.latest_checkpoint(model_path)
        if latest_checkpoint is not None:
            tf.train.Saver().restore(session, latest_checkpoint)
            print("Checkpoint loaded")
        else:
            session.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(model_path, graph=session.graph)

        if FLAGS.run_training:
            thread = threading.Thread(target=load, args=(session, data_path, close_op, enqueue_op, q_inputs, q_labels))
            thread.isDaemon()
            thread.start()

            last_epoch = 0
            start = time.time()
            avg_loss = 0
            batch = 0

            try:
                while True:
                    if epoch != last_epoch:
                        last_epoch = epoch
                        tf.train.Saver().save(session, checkpoint)
                        lr = (FLAGS.learning_rate * FLAGS.epochs_to_train - epoch) / FLAGS.epochs_to_train
                        session.run(tf.assign(learning_rate, lr))
                        start = time.time()
                        avg_loss = 0
                        batch = 0

                    batch += 1
                    avg_loss += session.run([train_op, loss])[1]

                    if batch % 1000 == 0:
                        current = time.time()
                        avg_loss /= 1000
                        rate = 0.001 * batch * FLAGS.batch_size / (current - start)
                        print("epoch: %d  batch: %d  loss: %.2f  words/sec: %.2fk" % (epoch, batch, avg_loss, rate))
                        summary = session.run(summary_op)
                        writer.add_summary(summary, epoch * batch)
                        avg_loss = 0

                    if batch % 500000 == 0:
                        tf.train.Saver().save(session, checkpoint, global_step=global_step)

            except tf.errors.OutOfRangeError:
                tf.train.Saver().save(session, checkpoint, global_step=global_step)
                writer.close()

        if FLAGS.plot:
            seeds = ("berlin", "john", "november", "cancer", "blue", "school")
            seed_ids = []
            for seed in seeds:
                word_id = vocab.get(seed, -1)
                if word_id != -1:
                    seed_ids.append(word_id)
            labels = []
            array = []
            for word_id in seed_ids:
                ids, r_embeddings = session.run([top_items, result_embedding], {query_id: [word_id]})
                for id in ids.ravel():
                    labels.append(rev_vocab[id])
                for e in r_embeddings:
                    array.extend(e)
            plot_graph(array, labels, plot_path)

        while True:
            word = input("\nQuery word: ").strip()
            word_id = vocab.get(word, -1)
            if word_id == -1:
                print("Unknown word!")
            else:
                ids, similarity = session.run([top_items, cosine_similarity], {query_id: [word_id]})
                score = similarity.ravel()
                for id in ids.ravel():
                    print("%f %s" % (score[id], rev_vocab[id]))


if __name__ == "__main__":
    tf.app.run()
