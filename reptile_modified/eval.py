"""
Helpers for evaluating models.
"""

from reptile import Reptile
from variables import weight_decay
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score
from tensorflow.python.platform import flags
import pandas as pd
import os, pickle
FLAGS = flags.FLAGS


# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             X, y,
             evaluate_testing=True,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    exp_string = os.path.join(FLAGS.logdir, 'model_{}'.format(FLAGS.model_num))
    test_preds, test_actuals, probabilities = [], [], []
    total_correct = 0
    for _ in range(num_samples):
        correct, y_test, y_pred, probas = reptile.evaluate(X, y, model.input_ph, model.label_ph,
                                          model.minimize_op, model.predictions, model.probas,
                                          num_classes=num_classes, num_shots=num_shots,
                                          inner_batch_size=eval_inner_batch_size,
                                          inner_iters=eval_inner_iters, replacement=replacement)
        total_correct += correct
        test_actuals.extend(y_test)
        test_preds.extend(y_pred)
        probabilities.extend(probas)

    accuracy, precision, recall, f1score, roc = evaluate_predictions(test_actuals, test_preds)
    if FLAGS.cost_sensitive == 1:
        f2, gmean, bss, pr_auc, sensitivity, specificity = evaluate_predictions_cost_sensitive(test_actuals, test_preds)
        results = compile_results_cost_sensitive(accuracy, precision, recall, f1score, roc,
                                                 f2, gmean, bss, pr_auc, sensitivity, specificity)
    else:
        results = compile_results(accuracy, precision, recall, f1score, roc)

    # save risk data frame only when evaluating testing
    if evaluate_testing:
        risk_df = pd.DataFrame()
        risk_df['test_indices'] = list(range(len(test_actuals)))
        risk_df['y_test'] = test_actuals
        risk_df['y_pred'] = test_preds
        risk_df['risk_scores'] = probabilities

        # sort by ascending order of risk score
        risk_df = risk_df.sort_values(by='risk_scores', ascending=False)
        risk_df.to_csv(os.path.join(exp_string, 'risk_df.csv'), index=False)

    print('\nEvaluation Results:')
    for k, v in results.items():
        print('{}: {}'.format(k, v))

    if evaluate_testing:
        # save dictionary of results
        with open(os.path.join(exp_string, 'testing_error_metrics.p'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    else:
        # save dictionary of results
        with open(os.path.join(exp_string, 'training_error_metrics.p'), 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    return total_correct / (num_samples * num_classes)


def evaluate_predictions(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)

    return accuracy, precision, recall, f1score, roc


def evaluate_predictions_cost_sensitive(y_test, y_pred):
    f2 = fbeta_score(y_test, y_pred, beta=2)
    gmean = geometric_mean_score(y_test, y_pred, average='weighted')
    bss = brier_skill_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return f2, gmean, bss, pr_auc, sensitivity, specificity


def brier_skill_score(y, yhat):
    probabilities = [0.01 for _ in range(len(y))]
    brier_ref = brier_score_loss(y, probabilities)
    bs = brier_score_loss(y, yhat)
    return 1.0 - (bs / brier_ref)


def compile_results(accuracy, precision, recall, f1score, roc):
    return {
        'accuracy': '{:.5f}'.format(accuracy),
        'precision': '{:.5f}'.format(precision),
        'recall': '{:.5f}'.format(recall),
        'f1': '{:.5f}'.format(f1score),
        'roc': '{:.5f}'.format(roc),
    }


def compile_results_cost_sensitive(accuracy, precision, recall, f1score, roc,
                                    f2, gmean, bss, pr_auc, sensitivity, specificity):
    return {
        'accuracy': '{:.5f}'.format(accuracy),
        'precision': '{:.5f}'.format(precision),
        'recall': '{:.5f}'.format(recall),
        'f1': '{:.5f}'.format(f1score),
        'roc': '{:.5f}'.format(roc),
        'f2': '{:.5f}'.format(f2),
        'gmean': '{:.5f}'.format(gmean),
        'bss': '{:.5f}'.format(bss),
        'pr_auc': '{:.5f}'.format(pr_auc),
        'sensitivity': '{:.5f}'.format(sensitivity),
        'specificity': '{:.5f}'.format(specificity),
    }

