""" Code for loading data. """
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_images

FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, num_samples_per_class, batch_size, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """

        # batch size 32, num_classes 1, num_samples_per_class 2
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = config.get('num_classes', FLAGS.num_classes)  # num classes now 5 (5-way)?

        # training and testing data
        self.training_path = FLAGS.training_path
        self.testing_path = FLAGS.testing_path
        self.target_variable = FLAGS.target_variable
        self.cols_drop = FLAGS.cols_drop
        self.special_encoding = FLAGS.special_encoding

        if self.cols_drop is not None:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding).drop(self.cols_drop, axis=1)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1).drop(self.cols_drop, axis=1)
        else:
            if self.special_encoding:
                self.df_train = pd.read_csv(self.training_path, encoding=self.special_encoding)
                self.df_test = pd.read_csv(self.testing_path, encoding=self.special_encoding)
            else:
                self.df_train = pd.read_csv(self.training_path).drop(self.cols_drop, axis=1)
                self.df_test = pd.read_csv(self.testing_path).drop(self.cols_drop, axis=1)
        
        # training and testing numpy arrays
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.df_train.columns == self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.df_test.columns == self.target_variable])


        # self.img_size = config.get('img_size', (28, 28))
        # self.dim_input = np.prod(self.img_size)
        # self.dim_output = self.num_classes
        # # data that is pre-resized using PIL with lanczos filter
        # data_folder = config.get('data_folder', './data/omniglot_resized/')
        #
        # character_folders = [os.path.join(data_folder, family, character) \
        #                      for family in os.listdir(data_folder) \
        #                      if os.path.isdir(os.path.join(data_folder, family)) \
        #                      for character in os.listdir(os.path.join(data_folder, family))]
        # random.seed(1)
        # random.shuffle(character_folders)
        # # num_val = 100
        # # num_train = config.get('num_train', 1200) - num_val
        # num_val = 20
        # num_train = config.get('num_train', 100) - num_val
        # self.metatrain_character_folders = character_folders[:num_train]
        # if FLAGS.test_set:
        #     self.metaval_character_folders = character_folders[num_train + num_val:]
        # else:
        #     self.metaval_character_folders = character_folders[num_train:num_train + num_val]
        # self.rotations = config.get('rotations', [0, 90, 180, 270])

    def generate_episode_data(self, x, y, n_shots):
        data = []
        labels = []
        for cls in range(self.num_classes):
            idxs_curr = [idx for idx in range(len(x)) if y[idx] == cls]
            idxs_chosen = np.random.choice(range(len(idxs_curr)), size=n_shots, replace=False)
            data.append(x[idxs_chosen])
            labels.append([cls] * len(idxs_chosen))

        all_data = ([data[i] for i in range(len(data))])

        return np.concatenate(all_data), np.array(labels).reshape(-1, 1)

    def make_data_tensor(self, train=True):
        if train:
            num_total_batches = 100
        else:
            num_total_batches = 60

        # make list of files
        print('Generating filenames')
        all_data = []
        all_data_labels = []
        # in each batch (episode we must say), for each class create num_samples_per_class
        for _ in range(self.batch_size):
            if train:
                data, labels = self.generate_episode_data(x=self.X_train, y=self.y_train, n_shots=self.num_samples_per_class)
            else:
                data, labels = self.generate_episode_data(x=self.X_test, y=self.y_test, n_shots=self.num_samples_per_class)

            all_data.append(data)
            all_data_labels.append(labels)

        return all_data, all_data_labels

        # # make queue for tensorflow to read from
        # filename_queue = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=False)
        # print('Generating image processing ops')
        # image_reader = tf.WholeFileReader()
        # _, image_file = image_reader.read(filename_queue)
        # if FLAGS.datasource == 'miniimagenet':
        #     image = tf.image.decode_jpeg(image_file, channels=3)
        #     image.set_shape((self.img_size[0],self.img_size[1],3))
        #     image = tf.reshape(image, [self.dim_input])
        #     image = tf.cast(image, tf.float32) / 255.0
        # else:
        #     image = tf.image.decode_png(image_file)
        #     image.set_shape((self.img_size[0],self.img_size[1],1))
        #     image = tf.reshape(image, [self.dim_input])
        #     image = tf.cast(image, tf.float32) / 255.0
        #     image = 1.0 - image  # invert
        # num_preprocess_threads = 1 # TODO - enable this to be set to >1
        # min_queue_examples = 256
        # examples_per_batch = self.num_classes * self.num_samples_per_class  # examples_per_class = 5 * 2 = 10
        # batch_image_size = self.batch_size  * examples_per_batch # batch size: 32 * 10 = 320
        # print('Batching images')
        # images = tf.train.batch(
        #         [image],
        #         batch_size = batch_image_size,
        #         num_threads=num_preprocess_threads,
        #         capacity=min_queue_examples + 3 * batch_image_size,
        #         ) # tensor of shape: (320, 784) ... 320 is the batch_image_size, 784 is 28*28
        # all_image_batches, all_label_batches = [], []
        # print('Manipulating image data to be right shape')
        # for i in range(self.batch_size):
        #     image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch] # tensor of shape (10, 784) (take 10 10 until 320)
        #
        #     if FLAGS.datasource == 'omniglot':
        #         # omniglot augments the dataset by rotating digits to create new classes
        #         # get rotation per class (e.g. 0,1,2,0,0 if there are 5 classes)
        #         rotations = tf.multinomial(tf.log([[1., 1.,1.,1.]]), self.num_classes)
        #     label_batch = tf.convert_to_tensor(labels) # labels are the 10 labels we generated
        #     new_list, new_label_list = [], []
        #     for k in range(self.num_samples_per_class):
        #         class_idxs = tf.range(0, self.num_classes) # tensor (list) of [0, 1, 2, 3, 4]
        #         class_idxs = tf.random_shuffle(class_idxs) # tensor (list) of [3, 2, 0, 1, 4]
        #
        #         true_idxs = class_idxs*self.num_samples_per_class + k # [2, 8, 6, 0, 4]
        #         new_list.append(tf.gather(image_batch,true_idxs))
        #         if FLAGS.datasource == 'omniglot': # and FLAGS.train:
        #             new_list[-1] = tf.stack([tf.reshape(tf.image.rot90(
        #                 tf.reshape(new_list[-1][ind], [self.img_size[0],self.img_size[1],1]),
        #                 k=tf.cast(rotations[0,class_idxs[ind]], tf.int32)), (self.dim_input,))
        #                 for ind in range(self.num_classes)])
        #         new_label_list.append(tf.gather(label_batch, true_idxs))
        #     new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
        #     new_label_list = tf.concat(new_label_list, 0)
        #     all_image_batches.append(new_list)
        #     all_label_batches.append(new_label_list)
        # all_image_batches = tf.stack(all_image_batches) # Tensor of shape (32, 10, 784)
        # all_label_batches = tf.stack(all_label_batches)
        # all_label_batches = tf.one_hot(all_label_batches, self.num_classes) # Tensor of shape  (32, 10, 5) (32 is batch size, 10 is examples per batch, 5 is nb of classes)
        # return all_image_batches, all_label_batches

    def generate_sinusoid_batch(self, train=True, input_idx=None):
        # Note train arg is not used (but it is used for omniglot method.
        # input_idx is used during qualitative testing --the number of examples used for the grad update
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [self.batch_size])
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [self.batch_size])
        outputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_output])
        init_inputs = np.zeros([self.batch_size, self.num_samples_per_class, self.dim_input])
        for func in range(self.batch_size):
            init_inputs[func] = np.random.uniform(self.input_range[0], self.input_range[1], [self.num_samples_per_class, 1])
            if input_idx is not None:
                init_inputs[:,input_idx:,0] = np.linspace(self.input_range[0], self.input_range[1], num=self.num_samples_per_class-input_idx, retstep=False)
            outputs[func] = amp[func] * np.sin(init_inputs[func]-phase[func])
        return init_inputs, outputs, amp, phase
