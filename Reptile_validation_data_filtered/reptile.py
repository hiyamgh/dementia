"""
Supervised Reptile_validation_data learning and evaluation on arbitrary
datasets.
"""

import random
import tensorflow as tf
from variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)


class Reptile:
    """
    A meta-learning session.

    Reptile_validation_data can operate in two evaluation modes: normal
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
                   X, y,
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
        Perform a Reptile_validation_data training step.

        Args:
          X: input data
          y: output data
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
        old_vars = self._model_state.export_variables() # get latest weights
        new_vars = []
        for _ in range(meta_batch_size):
            # mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_dataset = _sample_mini_tasks(X, y, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                # minimize_op is optimizer(**optim_kwargs).minimize(self.loss)
                # so basically perform gradient decsent on this mini-batch
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
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
                 probas_op,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
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
        # train_set, test_set = _split_train_test(
        #     _sample_mini_dataset(dataset, num_classes, num_shots+1))
        train_set, test_set = _split_train_test(
            _sample_mini_tasks(X, y, num_classes, num_shots + 1))
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds, probas = self._test_predictions(train_set, test_set, input_ph, predictions, probas_op)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        # Since we are sampling only 1 example per class in testing, and we have 2
        # classes, i.e. 2 examples in total, its not logical to compute
        # precision, recall, etc. on 2 examples in each iteration
        # Thought: each iteration return y_test, and y_pred, and average over those when done with all iterations.
        self._full_state.import_variables(old_vars)
        y_test = [sample[1] for sample in test_set]
        y_pred = test_preds
        return num_correct, y_test, y_pred, probas

    def _test_predictions(self, train_set, test_set, input_ph, predictions, probas_op):
        if self._transductive:
            inputs, _ = zip(*test_set)
            res = self.session.run(predictions, feed_dict={input_ph: inputs})
            probas = self.session.run(probas_op, feed_dict={input_ph: inputs})[:, 1]
            return res, list(probas)

        res, probas = [], []
        for test_sample in test_set:
            inputs, _ = zip(*train_set) #
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
            probas.append(self.session.run(probas_op, feed_dict={input_ph: inputs})[:, 1][-1])
        return res, probas


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
                   X, y,
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
            # mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_dataset = _sample_mini_tasks(X, y, num_classes, num_shots)
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


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)


def _sample_mini_tasks(X, y, num_classes, num_shots):
    """ Samples a few shot task from X,y pairs

        Returns:
            An iterable of (input, label) pairs.
    """
    for class_idx in range(num_classes):
        idxs = [idx for idx in range(len(X)) if y[idx] == class_idx]
        idxs_chosen = random.sample(idxs, num_shots)
        for ic in idxs_chosen:
            yield (X[ic, :], class_idx)


def _mini_batches(samples, batch_size, num_batches, replacement):
    # inner_batch_size is batch_size, inner_iters is num_batches,
    # but batch_size must be less than the number of samples, or else:
    # ValueError: Sample larger than population or is negative

    # However, if without replacement inner batch size can be greater than number of samples, it works
    # because of shuffling (look at while loop).
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    # basically if num_batches is greater than the number of samples, its fine
    # because it will keep running (while(True)) and samples will be re-shuffled
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return


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