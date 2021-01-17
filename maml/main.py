import os
import numpy as np
import pandas as pd
import argparse
import random
import sklearn
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', action='store_true', default=True, help='set for test, otherwise train')
parser.add_argument('-d', '--dirname', default='ckpt', help='directory to save the model in')
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
				saver.save(sess, os.path.join(args.dirname, 'mini.mdl'))
				best_acc = acc
				print('saved into ckpt:', acc)


def evaluate(support_predicted, support_actual, query_predicted, query_actual):
	# for each meta batch, calculate accuracy / meta_batch_size
	support_accuracies = []
	support_precisions, support_recalls, support_f1s = [], [], []
	support_rocs, support_gmeans, support_fbetas, support_bsss, support_pr_aucs = [], [], [], [], []

	query_total_accuracies = []
	query_total_precisions, query_total_recalls, query_total_f1s = [], [], []
	query_total_rocs, query_total_gmeans, query_total_fbetas, query_total_bsss, query_total_pr_aucs = [], [], [], [], []

	for i in range(len(support_predicted)):
		accuracy, precision, recall, f1score, roc, gmean, fscore, bss, pr_auc = compute_metrics(predictions=np.int64(support_predicted[i]),
																							     labels=np.int64(support_actual[i]))

		support_accuracies.append(accuracy)
		support_precisions.append(precision)
		support_recalls.append(recall)
		support_f1s.append(f1score)
		support_rocs.append(roc)
		support_gmeans.append(gmean)
		support_fbetas.append(fscore)
		support_bsss.append(bss)
		support_pr_aucs.append(pr_auc)

	support_accuracy = np.mean(support_accuracies)
	support_precision = np.mean(support_precisions)
	support_recall = np.mean(support_recalls)
	support_f1 = np.mean(support_f1s)
	support_roc = np.mean(support_rocs)
	support_gmean = np.mean(support_gmeans)
	support_fbeta = np.mean(support_fbetas)
	support_bss = np.mean(support_bsss)
	support_pr_auc = np.mean(support_pr_aucs)

	for k in range(len(query_predicted)):
		query_accuracies = []
		query_precisions, query_recalls, query_f1s = [], [], []
		query_rocs, query_gmeans, query_fbetas, query_bsss, query_pr_aucs = [], [], [], [], []
		mini_batch = query_predicted[k]
		for i in range(len(mini_batch)):
			# query_accuracies.append(accuracy_score(query_actual[k][i], query_predicted[k][i]))
			accuracy, precision, recall, f1score, roc, gmean, fscore, bss, pr_auc = compute_metrics(
				predictions=np.int64(query_predicted[k][i]),
				labels=np.int64(query_actual[k][i]))
			query_accuracies.append(accuracy)
			query_precisions.append(precision)
			query_recalls.append(recall)
			query_f1s.append(f1score)
			query_rocs.append(roc)
			query_gmeans.append(gmean)
			query_fbetas.append(fscore)
			query_bsss.append(bss)
			query_pr_aucs.append(pr_auc)

		query_total_accuracies.append(np.mean(query_accuracies))
		query_total_precisions.append(np.mean(query_precisions))
		query_total_recalls.append(np.mean(query_recalls))
		query_total_f1s.append(np.mean(query_f1s))
		query_total_rocs.append(np.mean(query_rocs))
		query_total_gmeans.append(np.mean(query_gmeans))
		query_total_fbetas.append(np.mean(query_fbetas))
		query_total_pr_aucs.append(np.mean(query_pr_aucs))

	results = {
		'accuracy': [support_accuracy] + query_total_accuracies,
		'precision': [support_precision] + query_total_precisions,
		'recall': [support_recall] + query_total_recalls,
		'f1': [support_f1] + query_total_f1s,
		'roc': [support_roc] + query_total_rocs,
		'gmean': [support_gmean] + query_total_gmeans,
		'fbeta': [support_fbeta] + query_total_fbetas,
		'bss': [support_bss] + query_total_bsss,
		'pr_auc': [support_pr_auc] + query_total_pr_aucs
	}
	return results


	# sklearn.metrics.accuracy_score(query_actual[1].flatten(), query_predicted[1].flatten())

	# target_auc, target_ap, target_ar, target_f1 = compute_metrics(target_preds, target_vals)
	# print('precision: {:.3f}'.format(target_ap))
	# print('recall: {:.3f}'.format(target_ar))
	# print('F1: {:.3f}'.format(target_f1))
	# print('AUC: {:.3f}'.format(target_auc))


def brier_skill_score(y, yhat):
	probabilities = [0.01 for _ in range(len(y))]
	brier_ref = brier_score_loss(y, probabilities)
	bs = brier_score_loss(y, yhat)
	return 1.0 - (bs / brier_ref)


def compute_metrics(predictions, labels):
	'''compute metrics - regular and cos sensitive  '''
	accuracy = accuracy_score(labels, predictions)
	precision = precision_score(labels, predictions)
	recall = recall_score(labels, predictions)
	f1score = f1_score(labels, predictions)

	roc = roc_auc_score(labels, predictions)
	gmean = geometric_mean_score(labels, predictions, average='weighted')
	fscore = fbeta_score(labels, predictions, beta=2)
	bss = brier_skill_score(labels, predictions)
	pr_auc = average_precision_score(labels, predictions)
	# tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

	return accuracy, precision, recall, f1score, roc, gmean, fscore, bss, pr_auc


def test(model, sess):

	K = 5
	np.random.seed(1)
	random.seed(1)

	# repeat test accuracy for 600 times
	test_accs = []
	test_precisions = []
	test_recalls = []

	# accuracy, precision, recall, f1score, roc, gmean, fscore, bss, pr_auc
	test_accuracies_hiyam, test_precicions_hiyam, test_recalls_hiyam, test_f1score_hiyam = [], [], [], []
	test_rocs_hiyam = []
	test_gmeans_hiyam, test_fbetas_hiyam, test_bsss_hiyam, test_pr_aucs_hiyam = [], [], [], []



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

		ops = [model.support_pred_hiyam, model.support_actual_hiyam,
			   model.query_preds_hiyam, model.query_actuals_hiyam]

		support_predicted, support_actual, query_predicted, query_actual = sess.run(ops)

		result = evaluate(support_predicted, support_actual, query_predicted, query_actual)
		test_accuracies_hiyam.append(result['accuracy'])
		test_precicions_hiyam.append(result['precision'])
		test_recalls_hiyam.append(result['recall'])
		test_f1score_hiyam.append(result['f1'])
		test_rocs_hiyam.append(result['roc'])
		test_gmeans_hiyam.append(result['gmean'])
		test_fbetas_hiyam.append(result['fbeta'])
		test_bsss_hiyam.append(result['bss'])
		test_pr_aucs_hiyam.append(result['pr_auc'])

		if i == 599:
			ops = [model.test_query_tps, model.test_query_tns, model.test_query_fps, model.test_query_fns]
			tp, fp, tn, fn = sess.run(ops)

	results_hiyam = {
		'accuracy': test_accuracies_hiyam,
		'precision': test_precicions_hiyam,
		'recall': test_recalls_hiyam,
		'f1': test_f1score_hiyam,
		'roc': test_rocs_hiyam,
		'gmean': test_gmeans_hiyam,
		'bss': test_bsss_hiyam,
		'fbeta': test_fbetas_hiyam,
		'pr_auc': test_pr_aucs_hiyam
	}

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

	print('\n============================ Hiyam Results: ============================ ')
	for metric in results_hiyam:
		means = np.mean(results_hiyam[metric], 0)
		stds = np.std(results_hiyam[metric], 0)
		ci95 = 1.96 * stds / np.sqrt(600)

		print('\nMetric: {}'.format(metric))
		print('[support_t0, query_t0 - \t\t\tK] ')
		print('mean:', means)
		print('stds:', stds)
		print('ci95:', ci95)
		print('mean of all {}: {}'.format(metric, np.mean(means)))

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

	if os.path.exists(os.path.join(args.dirname, 'checkpoint')):
		# alway load ckpt both train and test.
		model_file = tf.train.latest_checkpoint(args.dirname)
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
