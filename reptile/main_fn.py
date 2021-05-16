"""
Train a model on Omniglot.
"""

import tensorflow as tf
import argparse
from functools import partial
from eval import evaluate
from omniglot import load_datasets
from reptile import Reptile, FOML
from train import train
import os
import pandas as pd
import warnings
warnings.filterwarnings('always')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pretrained', help='evaluate a pre-trained model', action='store_true', default=False)
parser.add_argument('--seed', help='random seed', default=0, type=int)
parser.add_argument('--save_dir', help='checkpoint directory', default='logs/')
parser.add_argument('--classes', help='number of classes per inner task', default=2, type=int)

# Data Pre-Processing related arguments
parser.add_argument('--training_data_path', help='path to training data', default='input/feature_extraction_train_updated.csv')
parser.add_argument('--testing_data_path', help='path to testing data', default='input/feature_extraction_test_updated.csv')
parser.add_argument('--target_variable', help='name of the column that acts as the target variable', default='label')
parser.add_argument('--cols_drop', help='list of columns to drop from the dataset', default=['article_title', 'article_content', 'source', 'source_category', 'unit_id'])
parser.add_argument('--special_encoding', help='special encoding while reading pandas dataframe, if any is needed', default='latin-1')
parser.add_argument('--scaling', help='scaling mechanism needed for scaling the data (without data leakage)', default='z-score')
parser.add_argument('--categorical_columns', help='path to list of categorical columns in the dataset', default=None)
parser.add_argument('--categorical_encoding', help='categorical encoding mechanism', default=None)

# Model-related arguments
parser.add_argument('--dim_hidden', help='number of neurons in each hidden layer', default='128, 64, 64')
parser.add_argument('--dim_name', help='unique index name for the list of hidden layers (above)', default='dim0')
parser.add_argument('--activation_fn', help='activation function used', default='relu')
parser.add_argument('--model_num', help='model number to store trained model. Better for tracking', default=1)

# cost-sensitive related arguments
parser.add_argument('--cost_sensitive', help='whether to evaluate in cost sensitive mode or not', default=0)
parser.add_argument('--cost_sensitive_type', help='type of cost applied', default='weighted')
parser.add_argument('--weights_vector', help='respective weights of each class if cost_sensitive_type=weighted', default="10, 1")
parser.add_argument('--cost_matrix', help='cost matrix if cost_sensitive_type=miss-classification', default=[[1, 2.15], [2.15, 1]])
parser.add_argument('--sampling_strategy', help='how to resample data, only done when cost_sensitive=1', default=None)
parser.add_argument('--top_features', help='number of top features (by feature selection) to consider', default=None)

# Other arguments
parser.add_argument('--shots', help='number of examples per class', default=32, type=int)
parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
parser.add_argument('--replacement', help='sample with replacement', action='store_true')
parser.add_argument('--learning-rate', help='Adam step size', default=1e-3, type=float)
parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
parser.add_argument('--meta-step-final', help='meta-training step size by the end', default=0.1, type=float)
parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
# parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
parser.add_argument('--meta-iters', help='meta-training iterations', default=1000, type=int)
parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
parser.add_argument('--weight-decay', help='weight decay rate', default=1, type=float)
parser.add_argument('--transductive', help='evaluate all samples at once', action='store_true')
parser.add_argument('--foml', help='use FOML instead of Reptile', action='store_true')
parser.add_argument('--foml-tail', help='number of shots for the final mini-batch in FOML', default=None, type=int)
parser.add_argument('--sgd', help='use vanilla SGD instead of Adam', action='store_true')
args = parser.parse_args()


def main():
    """
    Load data and train a model on it.
    """

    X_train, y_train, X_test, y_test = load_datasets(args)
    from models_fn import DataModel
    model = DataModel(args.classes, X_test.shape[1], **model_kwargs(args))

    # train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    # train_set = list(augment_dataset(train_set))
    # test_set = list(test_set)
    #
    # model = OmniglotModel(args.classes, **model_kwargs(args))

    checkpoint = os.path.join(args.save_dir, 'model_{}/'.format(args.model_num))
    # My question is that why in train() function we are passing both training and testing datasets?
    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            # train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
            train(sess, model, X_train, y_train, X_test, y_test, checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        # print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        # print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))
        # print('Train accuracy: ' + str(evaluate(sess, model, X_train, y_train, **eval_kwargs)))
        # print('Test accuracy: ' + str(evaluate(sess, model, X_test, y_test, **eval_kwargs)))

        if args.cost_sensitive == 0:
            num_correct, _ = evaluate(sess, model, X_train, y_train, **eval_kwargs)
            print('Train accuracy: ' + str(num_correct))
            num_correct, res_class, y_test, y_pred, probas = evaluate(sess, model, X_test, y_test, **eval_kwargs)
            print('Test accuracy: ' + str(num_correct))
            for k, v in res_class.items():
                print('Avg. {}: {}'.format(k, v))
        else:
            num_correct, _ = evaluate(sess, model, X_train, y_train, **eval_kwargs)
            print('Train accuracy: ' + str(num_correct))
            num_correct, res_cost, y_test, y_pred, probas = evaluate(sess, model, X_test, y_test, **eval_kwargs)
            print('Test accuracy: ' + str(num_correct))
            for k, v in res_cost.items():
                print('Avg. {}: {}'.format(k, v))

        risk_df = pd.DataFrame()
        risk_df['test_indices'] = list(range(len(y_test)))
        risk_df['y_test'] = y_test
        risk_df['y_pred'] = y_pred
        risk_df['risk_scores'] = probas

        # sort by ascending order of risk score
        risk_df = risk_df.sort_values(by='risk_scores', ascending=False)
        risk_df.to_csv(os.path.join(checkpoint, 'risk_df.csv'), index=False)


def dim_kwargs(parsed_args):
    """
    Build the dim_kwargs for the model for hyper parameterizing the
    number of layers and nodes from the parsed command-line arguments
    """
    res = {}
    res['dim_hidden'] = parsed_args.dim_hidden
    res['activation_fn'] = parsed_args.activation_fn
    res['cost_sensitive'] = parsed_args.cost_sensitive
    res['cost_sensitive_type'] = parsed_args.cost_sensitive_type
    res['weights_vector'] = parsed_args.weights_vector
    res['cost_matrix'] = parsed_args.cost_matrix
    return res


def model_kwargs(parsed_args):
    """
    Build the kwargs for model constructors from the
    parsed command-line arguments.
    """
    res = {'learning_rate': parsed_args.learning_rate}
    if parsed_args.sgd:
        res['optimizer'] = tf.train.GradientDescentOptimizer
    return res


def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_step_size_final': parsed_args.meta_step_final,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'weight_decay_rate': parsed_args.weight_decay,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'cost_sensitive': parsed_args.cost_sensitive
    }


def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'weight_decay_rate': parsed_args.weight_decay,
        'num_samples': parsed_args.eval_samples,
        'transductive': parsed_args.transductive,
        'reptile_fn': _args_reptile(parsed_args),
        'cost_sensitive': parsed_args.cost_sensitive
    }


def _args_reptile(parsed_args):
    if parsed_args.foml:
        return partial(FOML, tail_shots=parsed_args.foml_tail)
    return Reptile


if __name__ == '__main__':
    main()