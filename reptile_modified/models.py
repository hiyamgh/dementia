"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags


DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)
FLAGS = flags.FLAGS

activations = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'softmax': tf.nn.softmax,
    'swish': tf.nn.swish
}


# More on batch normalization: https://stackoverflow.com/questions/29979251/batch-normalization-in-neural-network
# Above link from: https://stackoverflow.com/questions/41269570/what-is-batch-normalizaiton-why-using-it-how-does-it-affect-prediction

class StructuredModel:
    """
    A model for Structured/Tabular 2D datasets
    """
    def __init__(self, num_classes, dim_input, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, dim_input))

        # get the number of hidden layers and the number of nodes in each
        dim_hidden = list(map(int, list(FLAGS.dim_hidden.split(", "))))
        ac_fn = activations[FLAGS.activation_fn]
        # start building the model
        out = tf.layers.dense(self.input_ph, dim_hidden[0])
        out = tf.nn.relu(out)
        for i in range(1, len(dim_hidden)):
            out = tf.layers.dense(out, dim_hidden[i])
            out = tf.layers.batch_normalization(out, training=True)
            out = ac_fn(out)

        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        if FLAGS.cost_sensitive == 1:
            ws = list(map(int, list(FLAGS.weights_vector.split(", "))))
            weights_vector = np.array(ws)
            scaled_logits = tf.math.multiply(self.logits, weights_vector)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=scaled_logits)
        else:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                       logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
        self.probas = tf.nn.softmax(self.logits)


# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)
