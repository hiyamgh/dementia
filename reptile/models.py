"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf
from main_run import args
DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
activations = {
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'softmax': tf.nn.softmax,
    'swish': tf.nn.swish
}


class DataModel:
    ''' Class created to design a feed forward neural network for tabular data '''
    def __init__(self, num_classes, num_features, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float64, shape=[None, num_features])
        out = self.input_ph

        dim_layers = args.dim_hidden
        dim_hidden = list(map(int, list(dim_layers.split(", "))))
        act_fn = args.activation_fn
        for i in range(len(dim_hidden)):
            out = tf.layers.dense(inputs=out, units=int(dim_hidden[i]), activation=activations[act_fn])
        self.logits = tf.layers.dense(out, num_classes)

        # fc1 = tf.layers.Dense(128, activation=tf.nn.relu)  # Hidden layer 2
        # fc2 = tf.layers.Dense(64, activation=tf.nn.relu)  # Hidden layer 3
        # fc3 = tf.layers.Dense(num_classes)  # Hidden layer 4
        # out = self.input_ph
        #
        # h1 = fc1(out)  # Get the output of hidden layer 1
        # h2 = fc2(h1)  # Get the output of hidden layer 2
        # h3 = fc3(h2)  # Get the output of hidden layer 3
        #
        # self.logits = tf.layers.dense(h3, num_classes)

        self.label_ph = tf.placeholder(tf.int64, shape=(None,))
        if args.cost_sensitive == 1:
            ws = list(map(int, list(args.weights_vector.split(", "))))
            weights_vector = np.array(ws)
            scaled_logits = tf.math.multiply(self.logits, weights_vector)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                      logits=scaled_logits)
        else:
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                       logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(**optim_kwargs).minimize(self.loss)


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
