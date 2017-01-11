"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.2 and TensorFlow_gpu-0.12
"""

import os
import threading
import time

import tensorflow as tf

import data_utils

epoch = 0

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 1.0, "Initial learning rate.")
flags.DEFINE_string("save_path", "model.ckpt", "Base name of checkpoint files.")
flags.DEFINE_string("train_file", "corpus.txt", "Name of the training data file.")
flags.DEFINE_boolean("query", True, "Set to true to query from saved model.")
flags.DEFINE_boolean("remove_oov", True, "Remove out of vocabulary word labels from training.")
flags.DEFINE_integer("batch_size", 256, "Number of training examples each step processes.")
flags.DEFINE_integer("embedding_size", 128, "Embedding dimension size.")
flags.DEFINE_integer("epochs_to_train", 4, "Number of epochs to train.")
flags.DEFINE_integer("min_occ", 8, "Minimum number of times a word should occur in vocabulary.")
flags.DEFINE_integer("max_vocab", 65536, "Maximum size of the vocabulary.")
flags.DEFINE_integer("num_neg_samples", 8, "Negative samples per training example.")
flags.DEFINE_integer("window_size", 4, "Number of words to predict to the left and right of the target word.")
FLAGS = flags.FLAGS


def load(session, data_path, close_op, enqueue_op, queue_inputs, queue_labels):
    global epoch
    token_path = os.path.join(data_path, "tokens.bin")
    for epoch in range(FLAGS.epochs_to_train):
        for batch_inputs, batch_labels in iter(data_utils.Iterator(token_path, FLAGS.batch_size, FLAGS.window_size)):
            session.run(enqueue_op, feed_dict={queue_inputs: batch_inputs, queue_labels: batch_labels})
    session.run(close_op)


def main(_):
    global epoch
    data_path = os.path.join(os.getcwd(), "data")
    model_path = os.path.join(os.getcwd(), "model")
    checkpoint = os.path.join(model_path, FLAGS.save_path)
    vocab, rev_vocab = data_utils.prepare(data_path, FLAGS.min_occ, FLAGS.max_vocab, FLAGS.remove_oov, FLAGS.train_file)
    vocab_size = len(vocab)

    # Add queue ops to graph.
    q_inputs = tf.placeholder(tf.int32, shape=(FLAGS.batch_size * FLAGS.window_size * 2))
    q_labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size * FLAGS.window_size * 2, 1))
    q = tf.FIFOQueue(50, [tf.int32, tf.int32])
    close_op = q.close()
    enqueue_op = q.enqueue([q_inputs, q_labels])

    # Add training ops to graph.
    inputs, labels = q.dequeue()
    embeddings = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0))
    input_embeddings = tf.nn.embedding_lookup(embeddings, inputs)
    nce_weights = tf.Variable(tf.random_uniform([vocab_size, FLAGS.embedding_size], -1.0, 1.0))
    nce_biases = tf.Variable(tf.zeros([vocab_size]))
    nce_loss = tf.nn.nce_loss(nce_weights, nce_biases, input_embeddings, labels, FLAGS.num_neg_samples, vocab_size)
    loss = tf.reduce_mean(nce_loss)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Add query ops to graph.
    query_id = tf.placeholder(tf.int32, shape=[1])
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    norm_embeddings = embeddings / norm
    query_embedding = tf.gather(norm_embeddings, query_id)
    dist = tf.matmul(query_embedding, norm_embeddings, transpose_b=True)
    top_items = tf.nn.top_k(dist, 8)[1]

    with tf.Session() as session:
        if not FLAGS.query:
            session.run(tf.global_variables_initializer())
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
                    if batch % 100 == 0:
                        current = time.time()
                        avg_loss /= 100
                        rate = 0.001 * batch * FLAGS.batch_size / (current - start)
                        print("epoch: %d  batch: %d  loss: %.2f  words/sec: %.2fk" % (epoch, batch, avg_loss, rate))
                        avg_loss = 0

            except tf.errors.OutOfRangeError:
                tf.train.Saver().save(session, checkpoint)
        else:
            tf.train.Saver().restore(session, tf.train.latest_checkpoint(model_path))

        while True:
            word = input("\nQuery word: ").strip()
            word_id = vocab.get(word, -1)
            if word_id == -1:
                print("Unknown word!")
            else:
                items = session.run(top_items, {query_id: [word_id]})
                for item in items.ravel():
                    print(rev_vocab[item])


if __name__ == "__main__":
    tf.app.run()
