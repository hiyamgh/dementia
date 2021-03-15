"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.

    Note that better sinusoid results can be achieved by using a larger network.
"""
import csv
import numpy as np
import pandas as pd
import pickle
import random
import os
import tensorflow as tf

# from data_generator_correct import DataGenerator
from data_generator_hiyam2 import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'omniglot', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 2, 'number of classes used in classification (e.g. 5-way classification).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Data options - FAKE NEWS
# flags.DEFINE_string('training_data_path', 'feature_extraction_train_updated.csv', 'path to training data')
# flags.DEFINE_string('testing_data_path', 'feature_extraction_test_updated.csv', 'path to testing data')
# flags.DEFINE_string('target_variable', 'label', 'name of the target variable column')
# flags.DEFINE_string('fp_file', 'fake_news_fps/fps_fakenews_0.7.pickle', 'path to file containing the frequent patterns')
# flags.DEFINE_list('cols_drop', ['article_title', 'article_content', 'source', 'source_category', 'unit_id'], 'list of column to drop from data, if any')
# flags.DEFINE_string('special_encoding', 'latin-1', 'special encoding needed to read the data, if any')
# flags.DEFINE_string('scaling', 'z-score', 'scaling done to the dataset, if any')
flags.DEFINE_string('training_data_path', 'feature_extraction_train_updated.csv', 'path to training data')
flags.DEFINE_string('testing_data_path', 'feature_extraction_test_updated.csv', 'path to testing data')
flags.DEFINE_string('target_variable', 'label', 'name of the target variable column')
flags.DEFINE_string('include_fp', '1', 'whether to include frequent pattern in mining tasks or not, if yes 1, if no 0')
flags.DEFINE_string('fp_file', 'fake_news_fps_colsmeta/fps_fakenews_0.7.pickle', 'path to file containing the frequent patterns')
flags.DEFINE_string('colsmeta_file', 'fake_news_fps_colsmeta/colsmeta_fakenews_0.7.pickle', 'path to file containing the colsmeta')
flags.DEFINE_list('cols_drop', ['article_title', 'article_content', 'source', 'source_category', 'unit_id'], 'list of column to drop from data, if any')
flags.DEFINE_string('special_encoding', 'latin-1', 'special encoding needed to read the data, if any')
flags.DEFINE_string('scaling', 'z-score', 'scaling done to the dataset, if any')

## Training options
# flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
# flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
# flags.DEFINE_integer('meta_batch_size', 4, 'number of tasks sampled per meta-update')
# flags.DEFINE_float('meta_lr', 0.1, 'the base learning rate of the generator')
# flags.DEFINE_integer('update_batch_size', 16, 'number of examples used for inner gradient update (K for K-shot learning).')
# flags.DEFINE_float('update_lr', 0.1, 'step size alpha for inner gradient update.') # 0.1 for omniglot
# flags.DEFINE_integer('num_updates', 4, 'number of inner gradient updates during training.')

flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 1000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 1e-1, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 16, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-1, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 4, 'number of inner gradient updates during training.')
flags.DEFINE_float('supp_fp', 0.7, 'support value for fp growth')
# Metric: accuracy
# [support_t0, query_t0 - 			K]
# mean: [0.84472656 0.82958984 0.83105469 0.83398438 0.83544922 0.83105469
#  0.83154297 0.83056641 0.83105469 0.83056641 0.83203125]
# mean of all accuracy: 0.8328746448863636 +- 0.0

## Base model hyper parameters
flags.DEFINE_string('dim_hidden', '256, 128, 64', 'number of neurons in each hidden layer')
flags.DEFINE_string('dim_name', 'dim0', 'unique index name for the list of hidden layers (above)')
flags.DEFINE_string('activation_fn', 'relu', 'activation function used')
flags.DEFINE_integer('model_num', 1, 'model number to store trained model. Better for tracking')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', False, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## define options for cost-sensitive learning
flags.DEFINE_boolean('cost_sensitive', False, 'whether to apply cost sensitive learning or not')
# flags.DEFINE_string('cost_sensitive_type', 'miss-classification', 'type of cost applied')
flags.DEFINE_string('cost_sensitive_type', 'weighted', 'type of cost applied')
flags.DEFINE_string('weights_vector', "1, 100", 'if class_weights is used, then this are the respective weights'
                                                'of each classs')
flags.DEFINE_list('cost_matrix', [[1, 2.15], [2.15, 1]], 'cost matrix used, only applicable when using'
                                                       'miss-classification cost sensitive method')
flags.DEFINE_string('sampling_strategy', None, 'how to resample data, only done when cost sensitive is True')
flags.DEFINE_integer('top_features', 20, 'top features selected by feature selection')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'fake_news/', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

# 0.84375 accuracy (first test)
# 0.80113 accuracy (second test) closed then opened pycharm
# 0.81534 accuracy (third test)
####################################################
# WITHOUT FP GROWTH ==> data_generator_correct
# latest setting 0.83806
# latest setting Mean validation accuracy/loss, stddev, and confidence intervals
# (array([0.875  , 0.8125 , 0.8125 , 0.75   , 0.8125 , 0.875  , 0.875  ,
#        0.875  , 0.875  , 0.84375, 0.8125 ]),
#        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
#        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
#        Mean validation accuracy/loss, stddev, and confidence intervals
# (array([0.875  , 0.8125 , 0.8125 , 0.75   , 0.8125 , 0.875  , 0.875  ,
#        0.875  , 0.875  , 0.84375, 0.8125 ]),
#        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
#        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
#
# Process finished with exit code 0

#######################################################################
# 0.78125 Mean validation accuracy/loss, stddev, and confidence intervals
# (array([0.6875 , 0.8125 , 0.84375, 0.8125 , 0.78125, 0.78125, 0.78125,
#        0.78125, 0.75   , 0.78125, 0.78125]),
#        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
#
# Process finished with exit code 0
def compute_metrics(predictions, labels):
	'''compute metrics - regular and cos sensitive  '''
	accuracy = accuracy_score(labels, predictions)
	precision = precision_score(labels, predictions)
	recall = recall_score(labels, predictions)
	f1score = f1_score(labels, predictions)
	roc = roc_auc_score(labels, predictions)

	return accuracy, precision, recall, f1score, roc


def compute_cost_sensitive_metrics(predictions, labels):
    f2 = fbeta_score(labels, predictions, beta=2)
    gmean=geometric_mean_score(labels, predictions, average='weighted')
    bss = brier_skill_score(labels, predictions)
    pr_auc = average_precision_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
    sensitivity = tp/(tp + fn)
    specificity = tn/(tn + fp)
    return f2, gmean, bss, pr_auc, sensitivity, specificity


def brier_skill_score(y, yhat):
    probabilities = [0.01 for _ in range(len(y))]
    brier_ref= brier_score_loss(y, probabilities)
    bs= brier_score_loss(y, yhat)
    return 1.0 - (bs / brier_ref)

def unstack_prediction_probabilities(support_predicted, support_actual, support_probas,
                                     query_predicted, query_actual, query_probas,
                                     exp_string):
    """ craetes the risk data frame needed for advanced ML Evaluation """
    risk_df = pd.DataFrame()
    y_test, y_pred, probas = [], [], []
    for mini_batch in range(len(support_predicted)):
        y_pred.extend(support_predicted[mini_batch])
        y_test.extend(support_actual[mini_batch])
        # add the probability of the positive class
        probas.extend(support_probas[mini_batch][:, 1])

    for num_update in range(len(query_predicted)):
        for mini_batch in range(len(query_predicted[num_update])):
            y_pred.extend(query_predicted[num_update][mini_batch])
            y_test.extend(query_actual[num_update][mini_batch])
            # add the probability of the positive class
            probas.extend(query_probas[num_update][mini_batch][:, 1])

    # 'test_indices', 'y_test', 'y_pred', 'risk_scores'
    risk_df['test_indices'] = list(range(len(y_test)))
    risk_df['y_test'] = y_test
    risk_df['y_pred'] = y_pred
    risk_df['risk_scores'] = probas

    # sort by ascending order of risk score
    risk_df = risk_df.sort_values(by='risk_scores', ascending=False)
    risk_df.to_csv(os.path.join(FLAGS.logdir + '/' + exp_string, 'risk_df.csv'), index=False)


def evaluate_cost_sensitive(support_predicted, support_actual, query_predicted, query_actual):
    # gmean, bss, pr_auc
    support_f2s, support_gmeans, support_bsss, support_pr_aucs, \
    support_sensitivities, support_specificities = [], [], [], [], [], []
    query_total_f2s, query_total_gmeans, query_total_bsss, query_total_pr_aucs,\
        query_total_sensitivities, query_total_specificities = [], [], [], [], [], []

    for i in range(len(support_predicted)):
        # sn: sensitivity, sp: specificity
        f2, gmean, bss, pr_auc, sn, sp = compute_cost_sensitive_metrics(predictions=np.int64(support_predicted[i]),
                                                            labels=np.int64(support_actual[i]))

        support_f2s.append(f2)
        support_gmeans.append(gmean)
        support_bsss.append(bss)
        support_pr_aucs.append(pr_auc)
        support_sensitivities.append(sn)
        support_specificities.append(sp)

    support_f2 = np.mean(support_f2s)
    support_gmean = np.mean(support_gmeans)
    support_bss = np.mean(support_bsss)
    support_pr_auc = np.mean(support_pr_aucs)
    support_sensitivity = np.mean(support_sensitivities)
    support_specificity = np.mean(support_specificities)

    for k in range(len(query_predicted)):
        query_f2s, query_gmeans, query_bsss, query_pr_aucs = [], [], [], []
        query_sensitivities, query_specificities = [], []
        mini_batch = query_predicted[k]
        for i in range(len(mini_batch)):
            f2, gmean, bss, pr_auc, sn, sp = compute_cost_sensitive_metrics(predictions=np.int64(query_predicted[k][i]),
                                                                labels=np.int64(query_actual[k][i]))
            query_f2s.append(f2)
            query_gmeans.append(gmean)
            query_bsss.append(bss)
            query_pr_aucs.append(pr_auc)
            query_sensitivities.append(sn)
            query_specificities.append(sp)

        query_total_f2s.append(np.mean(query_f2s))
        query_total_gmeans.append(np.mean(query_gmeans))
        query_total_bsss.append(np.mean(query_bsss))
        query_total_pr_aucs.append(np.mean(query_pr_aucs))
        query_total_sensitivities.append(np.mean(query_sensitivities))
        query_total_specificities.append(np.mean(query_specificities))

    results = {
        'f2': [support_f2] + query_total_f2s,
        'gmean': [support_gmean] + query_total_gmeans,
        'bss': [support_bss] + query_total_bsss,
        'pr_auc': [support_pr_auc] + query_total_pr_aucs,
        'sensitivity': [support_sensitivity] + query_total_sensitivities,
        'specificity': [support_specificity] + query_total_specificities
    }

    return results


def evaluate(support_predicted, support_actual, query_predicted, query_actual):
    support_accuracies = []
    support_precisions, support_recalls, support_f1s, support_aucs = [], [], [], []

    query_total_accuracies = []
    query_total_precisions, query_total_recalls, query_total_f1s, query_total_aucs = [], [], [], []

    for i in range(len(support_predicted)):
        accuracy, precision, recall, f1score, auc = compute_metrics(predictions=np.int64(support_predicted[i]),
                                                                    labels=np.int64(support_actual[i]))

        support_accuracies.append(accuracy)
        support_precisions.append(precision)
        support_recalls.append(recall)
        support_f1s.append(f1score)
        support_aucs.append(auc)

    support_accuracy = np.mean(support_accuracies)
    support_precision = np.mean(support_precisions)
    support_recall = np.mean(support_recalls)
    support_f1 = np.mean(support_f1s)
    support_auc = np.mean(support_aucs)

    for k in range(len(query_predicted)):
        query_accuracies = []
        query_precisions, query_recalls, query_f1s, query_rocs = [], [], [], []
        mini_batch = query_predicted[k]
        for i in range(len(mini_batch)):
            accuracy, precision, recall, f1score, roc = compute_metrics(predictions=np.int64(query_predicted[k][i]),
                                                                        labels=np.int64(query_actual[k][i]))
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1score)
            query_rocs.append(roc)

        query_total_accuracies.append(np.mean(query_accuracies))
        query_total_precisions.append(np.mean(query_precisions))
        query_total_recalls.append(np.mean(query_recalls))
        query_total_f1s.append(np.mean(query_f1s))
        query_total_aucs.append(np.mean(query_rocs))

    results = {
        'accuracy': [support_accuracy] + query_total_accuracies,
        'precision': [support_precision] + query_total_precisions,
        'recall': [support_recall] + query_total_recalls,
        'f1': [support_f1] + query_total_f1s,
        'roc': [support_auc] + query_total_aucs
    }

    return results


def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 1000
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            if FLAGS.baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :] # b used for testing
            labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set.
        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 and FLAGS.datasource !='sinusoid':
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                if model.classification:
                    input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
                else:
                    input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]
            else:
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes*FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes*FLAGS.update_batch_size:, :]
                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                if model.classification:
                    input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]]
                else:
                    input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates-1]]

            result = sess.run(input_tensors, feed_dict)
            print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

# calculated for omniglot
NUM_TEST_POINTS = 600


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    metaval_accuracies2, metaval_precisions, metaval_recalls, metaval_f1s, metaval_aucs = [], [], [], [], []

    # cost sensitive metavals, if any
    metaval_f2s, metaval_gmeans, metaval_bsss, metaval_pr_aucs = [], [], [], []
    metaval_sensitivities, metaval_specificities = [], []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
            support_predicted, support_actual = sess.run([model.pred1, model.actual1], feed_dict)
            query_predicted, query_actual = sess.run([model.pred2, model.actual2], feed_dict)
            support_probabilities, query_probabilities = sess.run([model.proba1, model.proba2], feed_dict)
            metrics = evaluate(support_predicted, support_actual, query_predicted, query_actual)
            if FLAGS.cost_sensitive:
                metrics_cost_sensitive = evaluate_cost_sensitive(support_predicted, support_actual, query_predicted, query_actual)
            unstack_prediction_probabilities(support_predicted, support_actual, support_probabilities,
                                             query_predicted, query_actual, query_probabilities,
                                             exp_string)

        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)
        metaval_accuracies2.append(metrics['accuracy'])
        metaval_precisions.append(metrics['precision'])
        metaval_recalls.append(metrics['recall'])
        metaval_f1s.append(metrics['f1'])
        metaval_aucs.append(metrics['roc'])

        if FLAGS.cost_sensitive:
            metaval_f2s.append(metrics_cost_sensitive['f2'])
            metaval_gmeans.append(metrics_cost_sensitive['gmean'])
            metaval_bsss.append(metrics_cost_sensitive['bss'])
            metaval_pr_aucs.append(metrics_cost_sensitive['pr_auc'])
            metaval_sensitivities.append(metrics_cost_sensitive['sensitivity'])
            metaval_specificities.append(metrics_cost_sensitive['specificity'])

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    if FLAGS.cost_sensitive:
        results_final = {
            'accuracy': metaval_accuracies2,
            'precision': metaval_precisions,
            'recall': metaval_recalls,
            'f1': metaval_f1s,
            'roc': metaval_aucs,
            'f2': metaval_f2s,
            'gmean': metaval_gmeans,
            'bss': metaval_bsss,
            'pr_auc': metaval_pr_aucs,
            'sensitivity': metaval_sensitivities,
            'specificity': metaval_specificities
        }
    else:
        results_final = {
            'accuracy': metaval_accuracies2,
            'precision': metaval_precisions,
            'recall': metaval_recalls,
            'f1': metaval_f1s,
            'roc': metaval_aucs,
        }

    results_save = {}
    stds_save = {}
    print('\n============================ Results -- Evaluation ============================ ')
    for metric in results_final:
        means = np.mean(results_final[metric], 0)
        stds = np.std(results_final[metric], 0)
        ci95 = 1.96 * stds / np.sqrt(NUM_TEST_POINTS)

        print('\nMetric: {}'.format(metric))
        print('[support_t0, query_t0 - \t\t\tK] ')
        print('mean:', means)
        # print('stds:', stds)
        # print('ci95:', ci95)
        print('mean of all {}: {} +- {}'.format(metric, np.mean(means), np.mean(ci95)))
        results_save[metric] = '{:.5f}'.format(np.mean(means))
        stds_save[metric] = ':.5f'.format(np.std(means))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

    path_to_save = FLAGS.logdir + '/' + exp_string + '/'
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)

    # save dictionary of results
    with open(os.path.join(path_to_save, 'error_metrics.p'), 'wb') as f:
        pickle.dump(results_save, f, pickle.HIGHEST_PROTOCOL)
    # save dictionary of standard deviations
    with open(os.path.join(path_to_save, 'std_metrics.p'), 'wb') as f:
        pickle.dump(stds_save, f, pickle.HIGHEST_PROTOCOL)


def main():
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    if FLAGS.cost_sensitive:
        print('cost sensitive learning - turned on')
    else:
        print('cost sensitive learning - turned off')

    if FLAGS.datasource == 'sinusoid':
        if FLAGS.train:
            test_num_updates = 5
        else:
            test_num_updates = 10
    else:
        if FLAGS.datasource == 'miniimagenet':
            if FLAGS.train == True:
                test_num_updates = 1  # eval on at least one update during training
            else:
                test_num_updates = 10
        else:
            test_num_updates = 10

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)
    else:
        if FLAGS.metatrain_iterations == 0 and FLAGS.datasource == 'miniimagenet':
            assert FLAGS.meta_batch_size == 1
            assert FLAGS.update_batch_size == 1
            data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
        else:
            if FLAGS.datasource == 'miniimagenet': # TODO - use 15 val examples for imagenet?
                if FLAGS.train:
                    data_generator = DataGenerator(FLAGS.update_batch_size+15, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
            else:
                data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    if FLAGS.datasource == 'miniimagenet' or FLAGS.datasource == 'omniglot':
        tf_data_load = True
        num_classes = data_generator.num_classes

        if FLAGS.train: # only construct training model if needed
            random.seed(5)
            image_tensor, label_tensor = data_generator.make_data_tensor()
            inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
            input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

        random.seed(6)
        image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        labelb = tf.slice(label_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
    else:
        tf_data_load = False
        input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = 'model_{}'.format(FLAGS.model_num)
    # exp_string = 'miter_' + str(FLAGS.metatrain_iterations) +\
    #              '.mbs_'+str(FLAGS.meta_batch_size) + \
    #              '.ubs_' + str(FLAGS.train_update_batch_size) + \
    #              '.numup_' + str(FLAGS.num_updates) +\
    #              '.metalr_' + str(FLAGS.meta_lr) +\
    #              '.updatelr_' + str(FLAGS.train_update_lr) +\
    #              '.sfp_'+str(FLAGS.supp_fp) +\
    #              '.dn_'+str(FLAGS.dim_name) +\
    #              '.actfn_'+str(FLAGS.activation_fn)
    # if FLAGS.cost_sensitive:
    #     if FLAGS.sampling_strategy is not None:
    #         if FLAGS.top_features is not None:
    #             exp_string += '.ss_'+str(FLAGS.sampling_strategy) +\
    #                 '.topf_' + str(FLAGS.top_features) +\
    #                 '.weights_' + "_".join(list(map(str, FLAGS.weights_vector)))

    # exp_string = 'cls_'+str(FLAGS.num_classes)+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)
    # exp_string += 'miter_' + str(FLAGS.metatrain_iterations)
    #
    # if FLAGS.num_filters != 64:
    #     exp_string += 'hidden' + str(FLAGS.num_filters)
    # if FLAGS.max_pool:
    #     exp_string += 'maxpool'
    #
    # exp_string += ''
    # if FLAGS.stop_grad:
    #     exp_string += 'stopgrad'
    # if FLAGS.baseline:
    #     exp_string += FLAGS.baseline
    # if FLAGS.norm == 'batch_norm':
    #     exp_string += 'batchnorm'
    # elif FLAGS.norm == 'layer_norm':
    #     exp_string += 'layernorm'
    # elif FLAGS.norm == 'None':
    #     exp_string += 'nonorm'
    # else:
    #     print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    # if FLAGS.train:
    #     train(model, saver, sess, exp_string, data_generator, resume_itr)
    # else:
    #     test(model, saver, sess, exp_string, data_generator, test_num_updates)

    FLAGS.train = True
    train(model, saver, sess, exp_string, data_generator, resume_itr)
    FLAGS.train = False
    test(model, saver, sess, exp_string, data_generator, test_num_updates)


if __name__ == "__main__":
    main()
