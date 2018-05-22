import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

class LSTM(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self
                 , sequence_length
                 , vocab_size
                 , batch_size
                 , hidden_size=128
                 , embedding_size=None
                 , pretrained_embedding=None
                 , l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, sequence_length], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding_size:
                W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="W")
            else:
                W = tf.Variable(
                        tf.constant(pretrained_embedding, dtype=np.float32),
                        name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

        cell = rnn.BasicLSTMCell(hidden_size)
        self.initial_state = cell.zero_state(batch_size, self.data_type())
        cell_outputs = []
        with tf.variable_scope("LSTM"):
            for start_position in range(sequence_length):
                state = self.initial_state
                output = None
                for i in range(0, sequence_length):
                    if i > 0: tf.get_variable_scope().reuse_variables()
                    output, state = cell(self.embedded_chars[:, i, :], state)
                cell_outputs.append(output)

        batch_outputs = tf.stack(cell_outputs, 1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_out = tf.get_variable(
                "W_out",
                shape=[hidden_size, vocab_size],
                initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.Variable(tf.constant(0.1, shape=[vocab_size]), name="b_out")
            batch_out_r = tf.reshape(batch_outputs, (-1, hidden_size))
            self.scores = tf.nn.xw_plus_b(batch_out_r, W_out, b_out, "scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(self.input_y, (-1, 1)), logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.reshape(self.input_y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def data_type(self):
        return tf.float32