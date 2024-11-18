#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import argparse
from tensorflow.python.client import device_lib

# Uncomment the line below to disable GPU entirely
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

# Data loading params
parser.add_argument("--dev_sample_percentage", type=float, default=0.1, help="Percentage of the training data to use for validation")
parser.add_argument("--positive_data_file", type=str, default="/hdd/user4/cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.pos", help="Data source for the positive data.")
parser.add_argument("--negative_data_file", type=str, default="/hdd/user4/cnn-text-classification-tf-master/cnn-text-classification-tf-master/data/rt-polaritydata/rt-polarity.neg", help="Data source for the negative data.")

# Model Hyperparameters
parser.add_argument("--embedding_dim", type=int, default=128, help="Dimensionality of character embedding (default: 128)")
parser.add_argument("--filter_sizes", type=str, default="3,4,5", help="Comma-separated filter sizes (default: '3,4,5')")
parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per filter size (default: 128)")
parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
parser.add_argument("--l2_reg_lambda", type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")

# Training parameters
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size (default: 64)")
parser.add_argument("--num_epochs", type=int, default=200, help="Epoch Size (default: 200)")
parser.add_argument("--evaluate_every", type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
parser.add_argument("--checkpoint_every", type=int, default=100, help="Save model after this many steps (default: 100)")
parser.add_argument("--num_checkpoints", type=int, default=5, help="Number of checkpoints to store (default: 5)")

# Augment Parsing
FLAGS = parser.parse_args()

# Preprocessing
def preprocess():
    ### Load data
    print("Data loading...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    
    ### Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_text)
    x = tokenizer.texts_to_sequences(x_text)
    x = pad_sequences(x, maxlen=max_document_length, padding="post")
    x = np.array(x)
    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    
    del x, y, x_shuffled, y_shuffled
    
    print("Vocabulary Size: {:d}".format(len(tokenizer.word_index) + 1))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    return x_train, y_train, tokenizer, x_dev, y_dev

# Training
def train(x_train, y_train, tokenizer, x_dev, y_dev):
    timestamp = str(int(time.time()))
    out_dir = "/hdd/user4/cnn/output/"
    print("Writing to {}\n".format(out_dir))
    
    cnn = TextCNN(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_size=FLAGS.embedding_dim,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        l2_reg_lambda=FLAGS.l2_reg_lambda
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(out_dir, "summaries", "train"))
    dev_summary_writer = tf.summary.create_file_writer(os.path.join(out_dir, "summaries", "dev"))

    @tf.function
    def train_step(x_batch, y_batch):
        x_batch = tf.expand_dims(x_batch, -1)
        with tf.GradientTape() as tape:
            logits = cnn(x_batch, training=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
        gradients = tape.gradient(loss, cnn.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_batch, 1)), tf.float32))
        
        with train_summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=optimizer.iterations)
            tf.summary.scalar("accuracy", accuracy, step=optimizer.iterations)
        
        return loss, accuracy

    @tf.function
    def dev_step(x_batch, y_batch):
        x_batch = tf.expand_dims(x_batch, -1)
        
        logits = cnn(x_batch, training=False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=logits))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_batch, 1)), tf.float32))
        
        with dev_summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=optimizer.iterations)
            tf.summary.scalar("accuracy", accuracy, step=optimizer.iterations)
        
        return loss, accuracy

    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        loss, accuracy = train_step(x_batch, y_batch)
        current_step = optimizer.iterations.numpy()
        
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_loss, dev_accuracy = dev_step(x_dev, y_dev)
            print("step {}, loss {:g}, acc {:g}".format(current_step, dev_loss, dev_accuracy))
            print("")

        if current_step % FLAGS.checkpoint_every == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            print("Saved model checkpoint to {}\n".format(checkpoint_prefix))

if __name__ == '__main__':
    x_train, y_train, tokenizer, x_dev, y_dev = preprocess()
    train(x_train, y_train, tokenizer, x_dev, y_dev)
