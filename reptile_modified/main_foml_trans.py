"""
Train a model on Omniglot.
"""

import random
from functools import partial
import tensorflow as tf
from reptile import Reptile, FOML
from eval import evaluate
from models import StructuredModel
from train import train
from data_handler import generate_train_test
from tensorflow.python.platform import flags
import os, time

FLAGS = flags.FLAGS

# dataset related
flags.DEFINE_string('training_data_path', 'input/train_imputed_scaled.csv', 'path to training data')
flags.DEFINE_string('testing_data_path', 'input/test_imputed_scaled.csv', 'path to testing data')
flags.DEFINE_string('target_variable', 'dem1066', 'name of the target variable column')
flags.DEFINE_list('cols_drop', None, 'list of column to drop from data, if any')
flags.DEFINE_string('special_encoding', None, 'special encoding needed to read the data, if any')
flags.DEFINE_string('scaling', None, 'scaling done to the dataset, if any')
flags.DEFINE_string('categorical_columns', 'input/categorical.p', 'path to list of categorical columns in the dataset')
flags.DEFINE_string('categorical_encoding', 'target', 'categorical encoding mechanism')

# training - related
flags.DEFINE_boolean('pretrained', False, 'evaluate a pre-trained model')
flags.DEFINE_integer('seed', 0, 'random seed')
flags.DEFINE_string('logdir', 'trained_models', 'checkpoint directory')

# meta learning - related
flags.DEFINE_integer('classes', 2, 'number of classes per inner task')
flags.DEFINE_integer('shots', 5, 'number of examples per class')
flags.DEFINE_integer('train_shots', 0, 'shots in a training batch')
flags.DEFINE_integer('inner_batch', 5, 'inner batch size')
flags.DEFINE_integer('inner_iters', 20, 'inner iterations')
flags.DEFINE_boolean('replacement', False, 'sample with replacement')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam step size')
flags.DEFINE_float('meta_step', 0.1, 'meta-training step size')
flags.DEFINE_float('meta_step_final', 0.1, 'meta-training step size by the end')
# flags.DEFINE_integer('meta_batch', 1, 'meta-training batch size')
flags.DEFINE_integer('meta_batch', 5, 'meta-training batch size')
flags.DEFINE_integer('meta_iters', 1000, 'meta-training iterations')
flags.DEFINE_integer('eval_batch', 10, 'eval inner batch size')
flags.DEFINE_integer('eval_iters', 50, 'eval inner iterations')
flags.DEFINE_integer('eval_samples', 1000, 'evaluation samples')
flags.DEFINE_integer('eval_interval', 10, 'train steps per eval')
flags.DEFINE_float('weight_decay', 1, 'weight decay rate')
flags.DEFINE_boolean('transductive', True, 'evaluate all samples at once')
flags.DEFINE_boolean('foml', True, 'use FOML instead of Reptile')
flags.DEFINE_integer('foml_tail', 5, 'number of shots for the final mini-batch in FOML')
flags.DEFINE_boolean('sgd', False, 'use vanilla SGD instead of Adam')

## Base model hyper parameters
flags.DEFINE_string('dim_hidden', '128, 64', 'number of neurons in each hidden layer')
flags.DEFINE_string('activation_fn', 'relu', 'activation function used')
flags.DEFINE_integer('model_num', 1, 'model number to store trained model. Better for tracking')

# cost sensitive hyper parameters
flags.DEFINE_integer('cost_sensitive', 1, 'whether to imply cost sensitive learning or not')
flags.DEFINE_string('weights_vector', "1, 100", 'if class_weights is used, then this are the respective weights'
                                                'of each classs')
flags.DEFINE_string('sampling_strategy', 'all', 'how to resample data, only done when cost sensitive is True')
flags.DEFINE_integer('top_features', 20, 'top features selected by feature selection')


def model_kwargs():
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    res = {'learning_rate': FLAGS.learning_rate}
    if FLAGS.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    return res


def train_kwargs():
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': FLAGS.classes,
        'num_shots': FLAGS.shots,
        'train_shots': (FLAGS.train_shots or None),
        'inner_batch_size': FLAGS.inner_batch,
        'inner_iters': FLAGS.inner_iters,
        'replacement': FLAGS.replacement,
        'meta_step_size': FLAGS.meta_step,
        'meta_step_size_final': FLAGS.meta_step_final,
        'meta_batch_size': FLAGS.meta_batch,
        'meta_iters': FLAGS.meta_iters,
        'eval_inner_batch_size': FLAGS.eval_batch,
        'eval_inner_iters': FLAGS.eval_iters,
        'eval_interval': FLAGS.eval_interval,
        'weight_decay_rate': FLAGS.weight_decay,
        'transductive': FLAGS.transductive,
        'reptile_fn': _args_reptile(FLAGS)
    }


def evaluate_kwargs():
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': FLAGS.classes,
        'num_shots': FLAGS.shots,
        'eval_inner_batch_size': FLAGS.eval_batch,
        'eval_inner_iters': FLAGS.eval_iters,
        'replacement': FLAGS.replacement,
        'weight_decay_rate': FLAGS.weight_decay,
        'num_samples': FLAGS.eval_samples,
        'transductive': FLAGS.transductive,
        'reptile_fn': _args_reptile(FLAGS)
    }


def _args_reptile(FLAGS):
    if FLAGS.foml:
        return partial(FOML, tail_shots=FLAGS.foml_tail)
    return Reptile


def main():
    """
    Load data and train a model on it.
    """
    random.seed(FLAGS.seed)

    X_train, y_train, X_test, y_test = generate_train_test()
    model = StructuredModel(FLAGS.classes, X_test.shape[1], **model_kwargs())

    with tf.Session() as sess:
        if not FLAGS.pretrained:
            print('Training...')
            exp_string = os.path.join(FLAGS.logdir, 'model_{}'.format(FLAGS.model_num))
            t1 = time.time()
            train(sess, model, X_train, y_train, X_test, y_test, exp_string, **train_kwargs())
            t2 = time.time()
            training_time = (t2 - t1) / 60
            print('training time: {:.3f} mins'.format(training_time))
            with open(os.path.join(exp_string, 'statistics.txt'), 'w') as f:
                f.write('training time: {:.3f} mins'.format(training_time))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs()
        print('Train accuracy: ' + str(evaluate(sess, model, X_train, y_train, evaluate_testing=False, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, X_test, y_test, evaluate_testing=True, **eval_kwargs)))
    return


if __name__ == '__main__':
    main()
