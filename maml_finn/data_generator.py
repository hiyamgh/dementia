""" Code for loading data. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

""""
n_way is the number of classes 
n_shot is the number of ?
n_query is the number of ?

n_train_iters 
n_test_iters

meta_batch
meta_lr

parser.add_argument('--n_train_iters', type=int, default=60000)
parser.add_argument('--n_test_iters', type=int, default=1000)

parser.add_argument('--dataset', type=str, default='omniglot')
parser.add_argument('--way', type=int, default=20) # number of classes
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=5)

for omniglot, he made n_way = 20 & self.N = 20,
n_shot=1 and n_query=5
"""


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, args):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """

        self.N = args.examples_per_class # examples/instances per class
        self.Ktr = args.Ktr # total number of training classes (in my case it will be always 2)
        self.Kte = args.Kte # total number of testing classes (in my case it will be always 2)

        # TODO fill the xtr and xte with data
        # self.xtr = args.df_train
        # self.xte = args.df_test

        self.xtr, self.xte = [[] * args.num_calsses], [[] * args.num_calsses]
        # we have an X for each class
        for i in range(args.num_calsses):
            self.xtr[i] = args.train[i]
            self.xte[i] = args.test[i]

    def generate_episode(self, args, training=True, n_episodes=1):
        '''
        n_way: num_classes
        n_shot: ?
        n_query: ?
        K: total number of classes in train/test
        N: examples/instances per class
        n_episodes was num_total_batches (finn)  i.e. number of tasks
        In our case K_tr == K_te == 2
        In our case n_way = 2
        In our case N is number of samples per class
        n_shot is number of instances in support set
        n_query is number of instances in query set
        '''
        # generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
        def generate_label(n_way, n_samp):
            labels = []
            for i in range(n_way):
                labels.extend([i]*n_samp)
            return labels

        n_way, n_shot, n_query = args.way, args.shot, args.query
        K = self.Ktr if training else self.Kte # total number of classes in training/testing
        # x0 = self.xtr0 if training else self.xte0 # training/testing class 0
        # x1 = self.xtr1 if training else self.xte1 # training/testing class 1
        x = self.xtr if training else self.xte

        # query and support sets (xs and ys for inputs and labels)
        xs, ys, xq, yq = [], [], [], []

        # number of episodes in number of tasks .. in cbfinn, it was called num_total_batches
        for t in range(n_episodes):
            # sample WAY classes
            classes = np.random.choice(range(K), size=n_way, replace=False) # choose n_way classes out of the
                                                                            # total number of classes
                                                                            # assuming there are like 500 classes
                                                                            # but in our case (binary classification)
                                                                            # we will choose always 2(n_way) out of 2(K)

            support_set = []
            query_set = []

            # from each 'sampled' class, create support and query sets
            for k in list(classes):
                # sample SHOT and QUERY instances
                idx = np.random.choice(range(self.N), size=n_shot + n_query, replace=False)
                x_k = x[k][idx]
                support_set.append(x_k[:n_shot])
                query_set.append(x_k[n_shot:])

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = generate_label(n_way, n_shot)
            yq_k = generate_label(n_way, n_query)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)
        return [xs, ys, xq, yq]
