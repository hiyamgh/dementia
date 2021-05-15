"""
Helpers for evaluating models.
"""

from reptile import Reptile
from variables import weight_decay
import numpy as np


# pylint: disable=R0913,R0914
def evaluate(sess,
             model,
             X, y,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile,
             cost_sensitive=0):
    """
    Evaluate a model on a dataset.
    """
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    total_correct = 0
    accuracies, precisions, recalls, f1s, rocs = [], [], [], [], []
    if cost_sensitive == 0:
        for _ in range(num_samples):
            num_correct, res_class = reptile.evaluate(X, y, model.input_ph, model.label_ph,
                                              model.minimize_op, model.predictions,
                                              num_classes=num_classes, num_shots=num_shots,
                                              inner_batch_size=eval_inner_batch_size,
                                              inner_iters=eval_inner_iters, replacement=replacement,
                                              cost_sensitive=cost_sensitive)
            total_correct += num_correct
            accuracies.append(float(res_class['accuracy']))
            precisions.append(float(res_class['precision']))
            recalls.append(float(res_class['recall']))
            f1s.append(float(res_class['f1']))
            rocs.append(float(res_class['roc']))

        final_res_class = {
            'accuracy': '{:.5f}'.format(np.mean(accuracies)),
            'precision': '{:.5f}'.format(np.mean(precisions)),
            'recall': '{:.5f}'.format(np.mean(recalls)),
            'f1': '{:.5f}'.format(np.mean(f1s)),
            'roc': '{:.5f}'.format(np.mean(rocs)),
        }
        return total_correct / (num_samples * num_classes), final_res_class

    else:
        f2s, gmeans, bsss, pr_aucs, sensitivities, specificities = [], [], [], [], [], []
        for _ in range(num_samples):
            num_correct, res_class, res_cost = reptile.evaluate(X, y, model.input_ph, model.label_ph,
                                                      model.minimize_op, model.predictions,
                                                      num_classes=num_classes, num_shots=num_shots,
                                                      inner_batch_size=eval_inner_batch_size,
                                                      inner_iters=eval_inner_iters, replacement=replacement,
                                                      cost_sensitive=cost_sensitive)
            total_correct += num_correct

            # classification results
            accuracies.append(float(res_class['accuracy']))
            precisions.append(float(res_class['precision']))
            recalls.append(float(res_class['recall']))
            f1s.append(float(res_class['f1']))
            rocs.append(float(res_class['roc']))

            # cost sensitive results
            f2s.append(float(res_cost['f2']))
            gmeans.append(float(res_cost['gmean']))
            bsss.append(float(res_cost['bss']))
            pr_aucs.append(float(res_cost['pr_auc']))
            sensitivities.append(float(res_cost['sensitivity']))
            specificities.append(float(res_cost['specificity']))

        final_res_cost = {
            'accuracy': '{:.5f}'.format(np.mean(accuracies)),
            'precision': '{:.5f}'.format(np.mean(precisions)),
            'recall': '{:.5f}'.format(np.mean(recalls)),
            'f1': '{:.5f}'.format(np.mean(f1s)),
            'roc': '{:.5f}'.format(np.mean(rocs)),
            'f2': '{:.5f}'.format(np.mean(f2s)),
            'gmean': '{:.5f}'.format(np.mean(gmeans)),
            'bss': '{:.5f}'.format(np.mean(bsss)),
            'pr_auc': '{:.5f}'.format(np.mean(pr_aucs)),
            'sensitivity': '{:.5f}'.format(np.mean(sensitivities)),
            'specificity': '{:.5f}'.format(np.mean(specificities)),
        }

        return total_correct / (num_samples * num_classes), final_res_cost
