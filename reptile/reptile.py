"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score
from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)

class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
                                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   X_train, y_train,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            # mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots) # sample a mini-dataset in each batch (from each class in num_classes, take num_shot samples)
            mini_dataset = _sample_mini_dataset_tabular(X_train, y_train, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                inputs, labels = zip(*batch) # generate the X_train and y_train if you will (for each batch though)
                inputs = np.array(inputs)
                labels = np.array(labels)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels}) # run optimizer over loss
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

    def evaluate(self,
                 X, y,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement,
                 cost_sensitive=False):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        train_set, test_set = _split_train_test(
            _sample_mini_dataset_tabular(X, y, num_classes, num_shots+1)) # splitting the dataset again (which is already
        # a training or testing dataset .. into train and test ==> Splitting a few-shit task into train and test,
        # I assume like MAML split into support and query)
        # BUT WHY num_shots + 1 ?
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement): # for each batch in the train_set
            inputs, labels = zip(*batch) # get X_train and y_train basically
            inputs = np.array(inputs)
            labels = np.array(labels)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels}) # minimize loss of this batch
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        test_actual = [sample[1] for sample in test_set]
        if cost_sensitive == 1:
            accuracy, precision, recall, f1score, roc = self.evaluate_predictions(test_actual, test_preds)
            f2, gmean, bss, pr_auc, sensitivity, specificity = self.evaluate_predictions_cost_sensitive(test_actual, test_preds)
            res_class = self.compile_results(accuracy, precision, recall, f1score, roc)
            res_cost = self.compile_results_cost_sensitive(accuracy, precision, recall, f1score, roc,
                                           f2, gmean, bss, pr_auc, sensitivity, specificity)
            self._full_state.import_variables(old_vars)
            return num_correct, res_class, res_cost
        else:
            accuracy, precision, recall, f1score, roc = self.evaluate_predictions(test_actual, test_preds)
            res_class = self.compile_results(accuracy, precision, recall, f1score, roc)
            self._full_state.import_variables(old_vars)
            return num_correct, res_class

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive: # Not sure what transductive mode is ?
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set: # not sure what this is doing
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res

    def evaluate_predictions(self, y_test, y_pred):
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        return accuracy, precision, recall, f1score, roc

    def evaluate_predictions_cost_sensitive(self, y_test, y_pred):
        f2 = fbeta_score(y_test, y_pred, beta=2)
        gmean = geometric_mean_score(y_test, y_pred, average='weighted')
        bss = self.brier_skill_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        return f2, gmean, bss, pr_auc, sensitivity, specificity

    def brier_skill_score(self, y, yhat):
        probabilities = [0.01 for _ in range(len(y))]
        brier_ref = brier_score_loss(y, probabilities)
        bs = brier_score_loss(y, yhat)
        return 1.0 - (bs / brier_ref)

    def compile_results(self, accuracy, precision, recall, f1score, roc):
        return {
            'accuracy': '{:.5f}'.format(accuracy),
            'precision': '{:.5f}'.format(precision),
            'recall': '{:.5f}'.format(recall),
            'f1': '{:.5f}'.format(f1score),
            'roc': '{:.5f}'.format(roc),
        }

    def compile_results_cost_sensitive(self, accuracy, precision, recall, f1score, roc,
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


class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, *args, tail_shots=None, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch.
          kwargs: kwargs for Reptile.
        """
        super(FOML, self).__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail


def _sample_mini_dataset_tabular(X, y, num_classes, num_shots):
    """
        Sample a few shot task from a dataset.

        Returns:
          An iterable of (input, label) psampled airs.
    """

    X, y = shuffle(X, y, random_state=42)
    for class_idx in range(num_classes):
        idxs = [i for i in range(len(y)) if y[i] == class_idx]
        sampled_idxs = random.sample(idxs, num_shots)
        for sample in sampled_idxs:
            yield (X[sample], class_idx)


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset) # dataset is the list of 'Characters' which basically contains the path to each folder.
    random.shuffle(shuffled) # each folder is considered a 'class', and from each class (folder) w sample a certain number of images
    for class_idx, class_obj in enumerate(shuffled[:num_classes]): # since each 'Character' in the 'shuffled' is a different class
        for sample in class_obj.sample(num_shots): # get that class's path (omniglot folder) and sample num_shots images from there
            yield (sample, class_idx) # yield the sample and class_idx


def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    # the only difference I see between the two chunks below is that
    # the first one (with replacement) just samples randomly from 'samples'
    # while the second one does the same except that it 'shuffles' before
    samples = list(samples)
    if replacement: # if we're sampling with replacement, then we sample from the same set 'samples', num_batches times
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = [] # else if we're sampling without replacement:
    batch_count = 0
    while True:
        random.shuffle(samples) # shuffle the samples
        for sample in samples: # samples is the mini dataset we have, so for each sample in it
            cur_batch.append(sample) # add it to curr_batch (curr_batch becoming a list of lists)
            if len(cur_batch) < batch_size: # if the number of samples did not reach yet batch_size, continue
                continue
            yield cur_batch # if the number of samples reaches batch_size: yield it
            cur_batch = [] # empty up curr_batch
            batch_count += 1 # increment the batch count
            if batch_count == num_batches: # when the batch count is equal to num_batches
                return # we stop and return, otherwise, we shuffle the samples again

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
