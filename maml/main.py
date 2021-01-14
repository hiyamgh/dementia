import os
import numpy as np
import pandas as pd
import argparse
import random
import sklearn
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=True, help='set for test, otherwise train')
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(model, saver, sess):
	"""

	:param model:
	:param saver:
	:param sess:
	:return:
	"""
	# write graph to tensorboard
	# tb = tf.summary.FileWriter(os.path.join('logs', 'mini'), sess.graph)
	prelosses, postlosses, preaccs, postaccs = [], [], [], []
	best_acc = 0

	# train for meta_iteartion epoches
	# for iteration in range(600000):
	for iteration in range(20000):
		# this is the main op
		ops = [model.meta_op]

		# add summary and print op
		if iteration % 200 == 0:
			ops.extend([model.summ_op,
			            model.query_losses[0], model.query_losses[-1],
			            model.query_accs[0], model.query_accs[-1]])

		# run all ops
		result = sess.run(ops)

		# summary
		if iteration % 200 == 0:
			# summ_op
			# tb.add_summary(result[1], iteration)
			# query_losses[0]
			prelosses.append(result[2])
			# query_losses[-1]
			postlosses.append(result[3])
			# query_accs[0]
			preaccs.append(result[4])
			# query_accs[-1]
			postaccs.append(result[5])

			print(iteration, '\tloss:', np.mean(prelosses), '=>', np.mean(postlosses),
			      '\t\tacc:', np.mean(preaccs), '=>', np.mean(postaccs))
			prelosses, postlosses, preaccs, postaccs = [], [], [], []

		# evaluation
		if iteration % 2000 == 0:
			# DO NOT write as a = b = [], in that case a=b
			# DO NOT use train variable as we have train func already.
			acc1s, acc2s = [], []
			# sample 20 times to get more accurate statistics.
			for _ in range(200):
				acc1, acc2 = sess.run([model.test_query_accs[0],
				                        model.test_query_accs[-1]])
				acc1s.append(acc1)
				acc2s.append(acc2)

			acc = np.mean(acc2s)
			print('>>>>\t\tValidation accs: ', np.mean(acc1s), acc, 'best:', best_acc, '\t\t<<<<')

			if acc - best_acc > 0.05 or acc > 0.4:
				saver.save(sess, os.path.join('ckpt', 'mini.mdl'))
				best_acc = acc
				print('saved into ckpt:', acc)


def evaluate(sess, model):
	input_tensors = [model.test_query_pred]
	metaval_target_preds = sess.run(input_tensors)

	target_vals = np.array(model.query_y).flatten()
	target_preds = np.array([np.argmax(preds, axis=2) for preds in metaval_target_preds]).flatten()

	target_auc, target_ap, target_ar, target_f1 = compute_metrics(target_preds, target_vals)
	print('precision: {:.3f}'.format(target_ap))
	print('recall: {:.3f}'.format(target_ar))
	print('F1: {:.3f}'.format(target_f1))
	print('AUC: {:.3f}'.format(target_auc))


def compute_metrics(predictions, labels):
	'''compute metrics score'''
	fpr, tpr, _ = sklearn.metrics.roc_curve(labels, predictions)
	auc = sklearn.metrics.auc(fpr, tpr)
	ncorrects = sum(predictions == labels)
	accuracy = sklearn.metrics.accuracy_score(labels, predictions)
	ap = sklearn.metrics.average_precision_score(labels, predictions, 'micro')
	ar = sklearn.metrics.recall_score(labels, predictions)
	f1score = sklearn.metrics.f1_score(labels, predictions, 'micro')
	return auc, ap, ar, f1score


def test(model, sess):

	K = 5
	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	test_precisions = []
	test_recalls = []


	# for i in range(600):
	for i in range(600):
		if i % 100 == 1:
			print(i)
		# extend return None!!!
		ops = [model.test_support_acc]
		ops.extend(model.test_query_accs)
		result = sess.run(ops)
		test_accs.append(result)

		ops = [model.test_support_prec]
		ops.extend(model.test_query_precisions)
		result = sess.run(ops)
		test_precisions.append(result)

		ops = [model.test_support_rec]
		ops.extend(model.test_query_recalls)
		result = sess.run(ops)
		test_recalls.append(result)

		if i == 599:
			ops = [model.test_query_tps, model.test_query_tns, model.test_query_fps, model.test_query_fns]
			tp, fp, tn, fn = sess.run(ops)

	# [600, K+1]
	for all_results in list(zip(['accuracy', 'precision', 'recall'], [test_accs, test_precisions, test_recalls])):

		metric = all_results[0]
		test_accs = all_results[1]
		test_accs = np.array(test_accs)
		# [K+1]
		means = np.mean(test_accs, 0)
		stds = np.std(test_accs, 0)
		ci95 = 1.96 * stds / np.sqrt(600)

		print('\nMetric: {}'.format(metric))
		print('[support_t0, query_t0 - \t\t\tK] ')
		print('mean:', means)
		print('stds:', stds)
		print('ci95:', ci95)

		print('mean of all {}: {}'.format(metric, np.mean(means)))

	print('\nTP={} \t\t FP={}'.format(tp, fp))
	print('FN={} \t\t TN={}'.format(fn, tn))

	# predicted = sess.run([model.test_query_pred])
	# target_vals = np.array(model.query_y).flatten()
	# # target_preds = np.array([np.argmax(preds, axis=2) for preds in metaval_target_preds]).flatten()
	#
	# target_auc, target_ap, target_ar, target_f1 = compute_metrics(target_preds, target_vals)
	# print('precision: {:.3f}'.format(target_ap))
	# print('recall: {:.3f}'.format(target_ar))
	# print('F1: {:.3f}'.format(target_f1))
	# print('AUC: {:.3f}'.format(target_auc))



def main():
	training = not args.test
	kshot = 8
	kquery = 8
	nway = 2
	meta_batchsz = 4
	# meta_batchsz = 16
	K = 5


	# kshot + kquery images per category, nway categories, meta_batchsz tasks.
	db = DataGenerator(nway, kshot, kquery, meta_batchsz, 100)

	if  training:  # only construct training model if needed
		# get the tensor
		# image_tensor: [4, 80, 84*84*3]
		# label_tensor: [4, 80, 5]
		image_tensor, label_tensor = db.make_data_tensor(training=True)

		# NOTICE: the image order in 80 images should like this now:
		# [label2, label1, label3, label0, label4, and then repeat by 15 times, namely one task]
		# support_x : [4, 1*5, 84*84*3]
		# query_x   : [4, 15*5, 84*84*3]
		# support_y : [4, 5, 5]
		# query_y   : [4, 15*5, 5]
		support_x = tf.slice(image_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_x')
		query_x = tf.slice(image_tensor, [0,  nway *  kshot, 0], [-1, -1, -1], name='query_x')
		support_y = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_y')
		query_y = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1], name='query_y')

	# construct test tensors.
	image_tensor, label_tensor = db.make_data_tensor(training=False)
	support_x_test = tf.slice(image_tensor, [0, 0, 0], [-1,  nway *  kshot, -1], name='support_x_test')
	query_x_test = tf.slice(image_tensor, [0,  nway *  kshot, 0], [-1, -1, -1],  name='query_x_test')
	support_y_test = tf.slice(label_tensor, [0, 0, 0], [-1,  nway *  kshot, -1],  name='support_y_test')
	query_y_test = tf.slice(label_tensor, [0,  nway *  kshot, 0], [-1, -1, -1],  name='query_y_test')


	# 1. construct MAML model
	# def __init__(self, d, c, nway, meta_lr=1e-3, train_lr=1e-2):
	model = MAML(db.dim_input, db.dim_output, nway)

	# construct metatrain_ and metaval_
	if  training:
		model.build(support_x, support_y, query_x, query_y, K, meta_batchsz, mode='train')
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K, meta_batchsz, mode='eval')
	else:
		model.build(support_x_test, support_y_test, query_x_test, query_y_test, K + 5, meta_batchsz, mode='test')
	model.summ_op = tf.summary.merge_all()

	all_vars = filter(lambda x: 'meta_optim' not in x.name, tf.trainable_variables())
	for p in all_vars:
		print(p)


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	# tf.global_variables() to save moving_mean and moving variance of batch norm
	# tf.trainable_variables()  NOT include moving_mean and moving_variance.
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

	# initialize, under interative session
	tf.global_variables_initializer().run()
	####################################### Hiyam
	sess.run([tf.initialize_local_variables()])
	#####################################################
	tf.train.start_queue_runners()

	if os.path.exists(os.path.join('ckpt', 'checkpoint')):
		# alway load ckpt both train and test.
		model_file = tf.train.latest_checkpoint('ckpt')
		print("Restoring model weights from ", model_file)
		saver.restore(sess, model_file)


	if training:
		train(model, saver, sess)
		# print('\n DONE TRAINING --------------------- NOW TESTING THE MODEL...')
		# test(model, sess)
	else:
		test(model, sess)


if __name__ == "__main__":
	main()
