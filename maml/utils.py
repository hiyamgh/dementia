""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from itertools import product

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    inp = tf.cast(inp, tf.float32)
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    if FLAGS.cost_sensitive:
        if FLAGS.cost_sensitive_type == 'miss-classification':
            weights = np.array(FLAGS.cost_matrix)
            nb_cl = FLAGS.num_classes
            final_mask = tf.zeros_like(pred[:, 0])
            y_pred_max = tf.argmax(pred, axis=1)
            y_pred_max = tf.expand_dims(y_pred_max, 1)
            y_pred_max_mat = tf.equal(pred, tf.cast(y_pred_max, tf.float64))
            for c_p, c_t in product(range(nb_cl), range(nb_cl)):
                final_mask += tf.cast(weights[c_t, c_p], tf.float64) * tf.cast(y_pred_max_mat[:, c_p], tf.float64) * tf.cast(
                    label[:, c_t], tf.float64)
            return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) * final_mask / FLAGS.update_batch_size
        else:
            # if cost_sensitive_type == 'weighted':
            weights_vector = np.array(FLAGS.weights_vector)
            scaled_logits = tf.math.multiply(pred, weights_vector)
            return tf.nn.softmax_cross_entropy_with_logits(logits=scaled_logits, labels=label) / FLAGS.update_batch_size
    else:
        return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
