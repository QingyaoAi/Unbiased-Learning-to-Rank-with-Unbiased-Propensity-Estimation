"""Training and testing the dual learning algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.
	
	* Qingyao Ai, Keping Bi, Cheng Luo, Jiafeng Guo, W. Bruce Croft. 2018. Unbiased Learning to Rank with Unbiased Propensity Estimation. In Proceedings of SIGIR '18
	
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import numpy as np
from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
from DLA_model import DLA
import click_models as cm

#rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "/tmp/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./tmp/", "Training directory.")
tf.app.flags.DEFINE_string("test_dir", "./tmp/", "Directory for output test results.")
tf.app.flags.DEFINE_string("click_model_json", "", "Josn file for the click model used to generate clicks.")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")

tf.app.flags.DEFINE_integer("batch_size", 256,
							"Batch size to use during training.")
tf.app.flags.DEFINE_integer("train_list_cutoff", 10,
							"The number of documents to consider in each list during training.")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
							"Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
							"How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_non_clicked_data", False,
							"Set to True for estimating propensity weights for non-click data.")
tf.app.flags.DEFINE_boolean("decode", False,
							"Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
							"Set to True for decoding training data.")
# To be discarded.
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Set to True for self-testing.")



FLAGS = tf.app.flags.FLAGS


def create_model(session, data_set, forward_only):
	"""Create model and initialize or load parameters in session."""
	click_model = None
	with open(FLAGS.click_model_json) as fin:
		model_desc = json.load(fin)
		click_model = cm.loadModelFromJson(model_desc)
	
	model = DLA(click_model, data_set.rank_list_size, 
		data_set.embed_size, FLAGS.batch_size, FLAGS.hparams, forward_only, FLAGS.feed_previous)

	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
	if ckpt:
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print("Created model with fresh parameters.")
		session.run(tf.global_variables_initializer())
	return model


def train():
	# Prepare data.
	print("Reading data in %s" % FLAGS.data_dir)
	
	train_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
	valid_set = data_utils.read_data(FLAGS.data_dir, 'valid', FLAGS.train_list_cutoff)
	print("Rank list size %d" % train_set.rank_list_size)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Creating model...")
		model = create_model(sess, train_set, False)
		print("Created %d layers of %d units." % (model.hparams.num_layers, model.embed_size))

		# Create tensorboard summarizations.
		train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
										sess.graph)
		valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')

		#pad data
		train_set.pad(train_set.rank_list_size)
		valid_set.pad(valid_set.rank_list_size)


		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		best_loss = None
		while True:
			# Get a batch and make a step.
			start_time = time.time()
			input_feed, _ = model.get_batch(train_set)
			step_loss, _, summary = model.step(sess, input_feed, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1
			train_writer.add_summary(summary, current_step)

			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				
				# Print statistics for the previous epoch.
				#loss = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f loss "
							 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
												 step_time, loss))
				#train_writer.add_summary({'step-time':step_time, 'loss':loss}, current_step)

				# Decrease learning rate if no improvement was seen over last 3 times.
				#if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				#	sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				# Validate model
				it = 0
				count_batch = 0.0
				valid_loss = 0
				while it < len(valid_set.initial_list) - model.batch_size:
					input_feed, _ = model.get_next_batch(it, valid_set)
					v_loss, results, summary = model.step(sess, input_feed, True)
					it += model.batch_size
					valid_loss += v_loss
					count_batch += 1.0
				valid_writer.add_summary(summary, current_step)
				valid_loss /= count_batch
				#valid_loss = math.exp(valid_loss) if valid_loss < 300 else float('inf')
				print("  eval: loss %.2f" % (valid_loss))

				# Save checkpoint and zero timer and loss. # need to rethink
				#if best_loss == None or best_loss >= eval_ppx:
				#	best_loss = eval_ppx
				checkpoint_path = os.path.join(FLAGS.train_dir, "DLA.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				
				if loss == float('inf'):
					break

				step_time, loss = 0.0, 0.0
				sys.stdout.flush()

				if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
					break



def decode():
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Load test data.
		print("Reading data in %s" % FLAGS.data_dir)
		test_set = None
		if FLAGS.decode_train:
			test_set = data_utils.read_data(FLAGS.data_dir,'train')
		else:
			test_set = data_utils.read_data(FLAGS.data_dir,'test')

		# Create model and load parameters.
		model = create_model(sess, test_set, True)
		model.batch_size = 1	# We decode one sentence at a time.

		test_set.pad(test_set.rank_list_size)

		rerank_scores = []

		# Decode from test data.
		for i in xrange(len(test_set.initial_list)):
			input_feed, _ = model.get_data_by_index(test_set, i)
			test_loss, output_logits, summary = model.step(sess, input_feed, True)

			#The output is a list of rerank index for decoder_inputs (which represents the gold rank list)
			rerank_scores.append(output_logits[0])
			if i % FLAGS.steps_per_checkpoint == 0:
				print("Decoding %.2f \r" % (float(i)/len(test_set.initial_list))),

		#get rerank indexes with new scores
		rerank_lists = []
		for i in xrange(len(rerank_scores)):
			scores = rerank_scores[i]
			rerank_lists.append(sorted(range(len(scores)), key=lambda k: scores[k], reverse=True))

		if FLAGS.decode_train:
			data_utils.output_ranklist(test_set, rerank_scores, FLAGS.test_dir, 'train')
		else:
			data_utils.output_ranklist(test_set, rerank_scores, FLAGS.test_dir, 'test')

	return

#done
def self_test():
	"""Test the DLA model."""
	print('Self Test')
	os.system('rm -r ./tmp/*')
	# Prepare data.
	FLAGS.data_dir += 'sample_500/'
	print("Reading data in %s" % FLAGS.data_dir)
	train_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
	valid_set = data_utils.read_data(FLAGS.data_dir, 'valid', FLAGS.train_list_cutoff)
	print("Rank list size %d" % train_set.rank_list_size)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		# Create model.
		print("Creating model...")
		model = create_model(sess, train_set, False)
		print("Created %d layers of %d units." % (model.hparams.num_layers, model.embed_size))
		# Create tensorboard summarizations.
		train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
										sess.graph)
		valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')
		#pad data
		train_set.pad(train_set.rank_list_size)
		valid_set.pad(valid_set.rank_list_size)

		# This is the training loop.
		step_time, loss = 0.0, 0.0
		current_step = 0
		previous_losses = []
		best_loss = None
		print('Start Training')
		while True:
			# Get a batch and make a step.
			start_time = time.time()
			input_feed, others = model.get_batch(train_set)
			step_loss, _, summary = model.step(sess, input_feed, False)
			step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
			loss += step_loss / FLAGS.steps_per_checkpoint
			current_step += 1
			train_writer.add_summary(summary, current_step)


			# Once in a while, we save checkpoint, print statistics, and run evals.
			if current_step % FLAGS.steps_per_checkpoint == 0:
				'''
				for k in xrange(3):
					q_idx = others['rank_list_idxs'][k]
					print('Initial list')
					print(train_set.initial_list[q_idx])
					print('Features')
					print(train_set.features[train_set.initial_list[q_idx][5]][4:10])
					print('Embeddings')
					print(others['embeddings'][others['input_list'][k][5]][4:10])
					print('Encoder Inputs')
					print(others['input_list'][k])
					print('Gold labels')
					print(train_set.gold_weights[q_idx])
					#print('Outputs')
					#print(results[k])
					print('Decoder Clicks')
					print(others['click_list'][k])
					print('Propensity Weights')
					print(others['propensity_weights'][k])
					print('\n')
				'''

				# Print statistics for the previous epoch.
				#loss = math.exp(loss) if loss < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f loss "
							 "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
												 step_time, loss))
				

				# Decrease learning rate if no improvement was seen over last 3 times.
				#if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				#	sess.run(model.learning_rate_decay_op)
				previous_losses.append(loss)

				if loss == float('inf'):
					break

				# Validate model
				it = 0
				count_batch = 0.0
				valid_loss = 0
				while it < len(valid_set.initial_list) - model.batch_size:
					input_feed, others = model.get_batch(valid_set)
					v_loss, results, summary = model.step(sess, input_feed, True)
					it += model.batch_size
					valid_loss += v_loss
					count_batch += 1.0
				valid_writer.add_summary(summary, current_step)
				valid_loss /= count_batch
				#valid_loss = math.exp(valid_loss) if valid_loss < 300 else float('inf')
				print("  eval: loss %.2f" % (valid_loss))

				# Debug print
				q_idx = others['rank_list_idxs'][0]
				print('Gold labels')
				print(valid_set.gold_weights[q_idx])
				print('Outputs')
				print(results[0])
				print('Decoder Clicks')
				print(others['click_list'][0])
				print('\n')


				checkpoint_path = os.path.join(FLAGS.train_dir, "DLA.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)
				
				step_time, loss = 0.0, 0.0
				sys.stdout.flush()

				# Save checkpoint and zero timer and loss.
				#if best_loss == None or best_loss >= eval_ppx:
				#	best_loss = eval_ppx
				checkpoint_path = os.path.join(FLAGS.train_dir, "DLA.ckpt")
				model.saver.save(sess, checkpoint_path, global_step=model.global_step)

				if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
					break


def main(_):
	if FLAGS.self_test:
		self_test()
	elif FLAGS.decode:
		decode()
	else:
		train()

if __name__ == "__main__":
	tf.app.run()
