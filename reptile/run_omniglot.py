"""
Train a model on Omniglot.
"""

import random

import tensorflow as tf
import pandas as pd
import numpy as np
from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs, dim_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import OmniglotModel, DataModel
from supervised_reptile.omniglot import read_dataset, split_dataset, augment_dataset, load_datasets
from supervised_reptile.train import train
import warnings
warnings.filterwarnings('always')

# DATA_DIR = 'data/omniglot'
# DATA_DIR = 'E:/omniglot/'
# DATA_DIR = 'E:/omniglot/images_evaluation/images_evaluation/'


def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)
    dk = dim_kwargs(args)

    X_train, y_train, X_test, y_test = load_datasets(args)
    model = DataModel(args.classes, X_test.shape[1], **model_kwargs(args))

    # train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    # train_set = list(augment_dataset(train_set))
    # test_set = list(test_set)
    #
    # model = OmniglotModel(args.classes, **model_kwargs(args))

    # My question is that why in train() function we are passing both training and testing datasets?
    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            # train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
            train(sess, model, X_train, y_train, X_test, y_test, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        # print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        # print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))
        # print('Train accuracy: ' + str(evaluate(sess, model, X_train, y_train, **eval_kwargs)))
        # print('Test accuracy: ' + str(evaluate(sess, model, X_test, y_test, **eval_kwargs)))

        if args.cost_sensitive == 0:
            num_correct, _ = evaluate(sess, model, X_train, y_train, **eval_kwargs)
            print('Train accuracy: ' + num_correct)
            num_correct, res_class = evaluate(sess, model, X_test, y_test, **eval_kwargs)
            print('Test accuracy: ' + num_correct)
            for k, v in res_class.items():
                print('Avg. {}: {}'.format(k, v))
        else:
            num_correct, _ = evaluate(sess, model, X_train, y_train, **eval_kwargs)
            print('Train accuracy: ' + str(num_correct))
            num_correct, res_cost = evaluate(sess, model, X_test, y_test, **eval_kwargs)
            print('Test accuracy: ' + str(num_correct))
            for k, v in res_cost.items():
                print('Avg. {}: {}'.format(k, v))


if __name__ == '__main__':
    main()

# Evaluating...
# Train accuracy: 0.922
# Test accuracy: 0.8129