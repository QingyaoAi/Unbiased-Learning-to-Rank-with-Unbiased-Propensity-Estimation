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
import click_models as cm
import propensity_estimator as pe
from six.moves import xrange# pylint: disable=redefined-builtin
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import xrange# pylint: disable=redefined-builtin
from six.moves import zip	 # pylint: disable=redefined-builtin


import copy
import itertools
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

def imerge(a, b):
    for i, j in itertools.izip(a,b):
        yield i
        yield j

def sigmoid_prob(logits):
	return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))

class DLA(object):
	"""Sequence-to-sequence model with attention and for multiple buckets.

	This class implements a multi-layer recurrent neural network as encoder,
	and an attention-based decoder. This is the same as the model described in
	this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
	or into the seq2seq library for complete model implementation.
	This class also allows to use GRU cells in addition to LSTM cells, and
	sampled softmax to handle large output vocabulary size. A single-layer
	version of this model, but with bi-directional encoder, was presented in
	http://arxiv.org/abs/1409.0473
	and sampled softmax is described in Section 3 of the following paper.
	http://arxiv.org/abs/1412.2007
	"""

	def __init__(self, click_model, rank_list_size,
		embed_size, batch_size, hparam_str, forward_only=False, feed_previous = False):
		"""Create the model.
	
		Args:
			rank_list_size: size of the ranklist.
			batch_size: the size of the batches used during training;
						the model construction is not independent of batch_size, so it cannot be
						changed after initialization.
			embed_size: the size of the input feature vectors.
			forward_only: if set, we do not construct the backward pass in the model.
		"""
		print('Noise LSTM')
		self.click_model = click_model

		self.hparams = tf.contrib.training.HParams(
			learning_rate=0.05, 				# Learning rate.
			#learning_rate_decay_factor=0.8, # Learning rate decays by this much.
			max_gradient_norm=5.0,			# Clip gradients to this norm.
			#reverse_input=True,				# Set to True for reverse input sequences.
			hidden_layer_sizes=[512, 256, 128],		# Number of neurons in each layer of a RankNet. 
			loss_func='click_weighted_softmax_cross_entropy',			# Select Loss function
			logits_to_prob='softmax',		# the function used to convert logits to probability distributions
			ranker_learning_rate=-1.0, 		# The learning rate for ranker (-1 means same with learning_rate).
			ranker_loss_weight=1.0,			# Set the weight of unbiased ranking loss
			l2_loss=0.0,					# Set strength for L2 regularization.
			grad_strategy='ada',			# Select gradient strategy
			relevance_category_num=5,		# Select the number of relevance category
			use_previous_rel_prob=False,  # Set to True for using ranking features in denoise model.
			use_previous_clicks=False,  # Set to True for using ranking features in denoise model.
			split_gradients_for_denoise=True, # Set to True for splitting gradient computation in denoise model.
		)
		print(hparam_str)
		self.hparams.parse(hparam_str)

		self.start_index = 0
		self.count = 1
		self.rank_list_size = rank_list_size
		self.embed_size = embed_size
		self.batch_size = batch_size
		if self.hparams.ranker_learning_rate < 0:
			self.ranker_learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
		else:
			self.ranker_learning_rate = tf.Variable(float(self.hparams.ranker_learning_rate), trainable=False)
		self.learning_rate = self.ranker_learning_rate
		#self.ranker_learning_rate_decay_op = self.ranker_learning_rate.assign(
		#	self.ranker_learning_rate * self.hparams.learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)
		#self.PAD_embed = tf.zeros([1,self.embed_size],dtype=tf.float32)
		
		# Feeds for inputs.
		self.encoder_inputs = []
		self.embeddings = tf.placeholder(tf.float32, shape=[None, embed_size], name="embeddings")
		self.target_inputs = []
		self.target_clicks = []
		for i in xrange(self.rank_list_size):
			self.encoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
											name="encoder{0}".format(i)))
			self.target_inputs.append(tf.placeholder(tf.int64, shape=[None],
											name="target{0}".format(i)))
			self.target_clicks.append(tf.placeholder(tf.float32, shape=[None],
											name="click{0}".format(i)))

		# Select logits to prob function
		self.logits_to_prob = tf.nn.softmax
		if self.hparams.logits_to_prob == 'sigmoid':
			self.logits_to_prob = sigmoid_prob

		# Build model
		self.output = self.RankNet(forward_only)
		self.propensity = self.DenoisingNet(forward_only)

		print('Loss Function is ' + self.hparams.loss_func)
		# Select loss function
		self.loss_func = None
		if self.hparams.loss_func == 'click_weighted_softmax_cross_entropy':
			self.loss_func = self.click_weighted_softmax_cross_entropy_loss
		elif self.hparams.loss_func == 'click_weighted_log_loss':
			self.loss_func = self.click_weighted_log_loss
		else: # softmax loss without weighting
			self.loss_func = self.softmax_loss

		# Compute rank loss
		self.rank_loss, self.propensity_weights = self.loss_func(self.output, self.target_inputs, self.target_clicks, self.propensity)
		pw_list = tf.split(self.propensity_weights, self.rank_list_size, 1) # Compute propensity weights
		for i in xrange(self.rank_list_size):
			tf.summary.scalar('Avg Propensity weights %d' % i, tf.reduce_mean(pw_list[i]))
		tf.summary.scalar('Rank Loss', tf.reduce_mean(self.rank_loss))

		# Compute examination loss
		self.exam_loss, self.relevance_weights = self.loss_func(self.propensity, self.target_inputs, self.target_clicks, self.output)
		rw_list = tf.split(self.relevance_weights, self.rank_list_size, 1) # Compute propensity weights
		for i in xrange(self.rank_list_size):
			tf.summary.scalar('Avg Relevance weights %d' % i, tf.reduce_mean(rw_list[i]))
		tf.summary.scalar('Exam Loss', tf.reduce_mean(self.exam_loss))
		
		# Gradients and SGD update operation for training the model.
		self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss
		if not forward_only:
			# Select optimizer
			self.optimizer_func = tf.train.AdagradOptimizer
			if self.hparams.grad_strategy == 'sgd':
				self.optimizer_func = tf.train.GradientDescentOptimizer

			print('Split gradients computation %r' % self.hparams.split_gradients_for_denoise)
			if self.hparams.split_gradients_for_denoise:
				self.separate_gradient_update()
			else:
				self.global_gradient_update()
			tf.summary.scalar('Gradient Norm', self.norm)
			tf.summary.scalar('Learning Rate', self.ranker_learning_rate)
			tf.summary.scalar('Final Loss', tf.reduce_mean(self.loss))

		self.summary = tf.summary.merge_all()
		self.saver = tf.train.Saver(tf.global_variables())

	def separate_gradient_update(self):
		denoise_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "denoising_model")
		ranknet_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "RankNet")

		if self.hparams.l2_loss > 0:
			for p in denoise_params:
				self.click_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
			for p in ranknet_params:
				self.rank_loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
		self.loss = self.exam_loss + self.hparams.ranker_loss_weight * self.rank_loss

		denoise_gradients = tf.gradients(self.exam_loss, denoise_params)
		ranknet_gradients = tf.gradients(self.rank_loss, ranknet_params)
		if self.hparams.max_gradient_norm > 0:
			denoise_gradients, denoise_norm = tf.clip_by_global_norm(denoise_gradients,
																	 self.hparams.max_gradient_norm)
			ranknet_gradients, ranknet_norm = tf.clip_by_global_norm(ranknet_gradients,
																	 self.hparams.max_gradient_norm * self.hparams.ranker_loss_weight)
		self.norm = tf.global_norm(denoise_gradients + ranknet_gradients)
			#ranknet_norm = tf.global_norm()

		opt_denoise = self.optimizer_func(self.hparams.learning_rate)
		opt_ranker = self.optimizer_func(self.ranker_learning_rate)
		#opt_denoise = tf.train.GradientDescentOptimizer(self.hparams.learning_rate)
		#opt_ranker = tf.train.GradientDescentOptimizer(self.ranker_learning_rate)
		#tf.train.GradientDescentOptimizer(self.learning_rate)
		#tf.train.GradientDescentOptimizer(self.learning_rate)
		denoise_updates = opt_denoise.apply_gradients(zip(denoise_gradients, denoise_params),
											global_step=self.global_step)
		ranker_updates = opt_ranker.apply_gradients(zip(ranknet_gradients, ranknet_params))

		self.updates = tf.group(denoise_updates, ranker_updates)
		#self.updates = opt.apply_gradients(imerge(zip(denoise_gradients, denoise_params), 
		#									zip(ranknet_gradients, ranknet_params)),
		#									global_step=self.global_step)

	def global_gradient_update(self):
		params = tf.trainable_variables()
		if self.hparams.l2_loss > 0:
			for p in params:
				self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
		opt = self.optimizer_func(self.hparams.learning_rate)
		#opt = tf.train.GradientDescentOptimizer(self.ranker_learning_rate)
		self.gradients = tf.gradients(self.loss, params)	
		if self.hparams.max_gradient_norm > 0:
			self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																 self.hparams.max_gradient_norm)
			self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
										 global_step=self.global_step)
		else:
			self.norm = tf.global_norm(self.gradients)
			self.updates = opt.apply_gradients(zip(self.gradients, params),
											 global_step=self.global_step)

	def RankNet(self, forward_only=False, scope=None):
		with variable_scope.variable_scope(scope or "RankNet"):
			#PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)
			PAD_embed = tf.zeros([1,self.embed_size],dtype=tf.float32)
			embeddings = tf.concat(axis=0,values=[self.embeddings, PAD_embed])
			encoder_embed = []
			output_scores = []
			def network(input_data, index):
				reuse = None if index < 1 else True
				with variable_scope.variable_scope(variable_scope.get_variable_scope(),
												 reuse=reuse):
					output_data = input_data
					output_sizes = self.hparams.hidden_layer_sizes + [1]
					current_size = self.embed_size
					for i in xrange(len(output_sizes)):
						expand_W = variable_scope.get_variable("ranknet_W_%d" % i, [current_size, output_sizes[i]]) 
						expand_b = variable_scope.get_variable("ranknet_b_%d" % i, [output_sizes[i]])
						output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
						#output_data = tf.layers.batch_normalization(output_data, training=forward_only)
						output_data = tf.nn.elu(output_data)
						current_size = output_sizes[i]
					#if True:
					#	expand_W = variable_scope.get_variable("ranknet_W_final", [1,1]) 
					#	expand_b = variable_scope.get_variable("ranknet_b_final", [1])
					#	output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
					return output_data
			for i in xrange(self.rank_list_size):
				encoder_embed.append(embedding_ops.embedding_lookup(embeddings, self.encoder_inputs[i]))
				output_scores.append(network(encoder_embed[-1], i))

			return tf.concat(output_scores,1)

	def DenoisingNet(self, forward_only=False, scope=None):
		with variable_scope.variable_scope(scope or "denoising_model"):
			# If we are in testing, do not compute propensity
			if forward_only:
				return tf.ones_like(self.output)#, tf.ones_like(self.output)
			input_vec_size = self.rank_list_size
			print('Use previous relevance probability for denoising %r' % self.hparams.use_previous_rel_prob)
			rel_prob_list = []
			previous_rel_prob_list = []
			if self.hparams.use_previous_rel_prob:
				rel_prob_list = tf.split(self.logits_to_prob(self.output), self.rank_list_size, 1)
				previous_rel_prob_list = [tf.zeros_like(rel_prob_list[0]) for _ in xrange(self.rank_list_size)]
				input_vec_size += len(previous_rel_prob_list)

			print('Use previous clicks for denoising %r' % self.hparams.use_previous_clicks)
			previous_click_list = []
			if self.hparams.use_previous_clicks:
				previous_click_list = [tf.expand_dims(tf.zeros_like(self.target_clicks[0]) , -1) for _ in xrange(self.rank_list_size)]
				input_vec_size += len(previous_click_list)

			def propensity_network(input_data, index):
				reuse = None if index < 1 else True
				with variable_scope.variable_scope("propensity_network",
												 reuse=reuse):
					output_data = input_data
					current_size = input_vec_size
					output_sizes = [
						int((self.hparams.relevance_category_num+self.rank_list_size+1)/2), 
						int((self.hparams.relevance_category_num+self.rank_list_size+1)/4),
					]
					for i in xrange(len(output_sizes)):
						expand_W = variable_scope.get_variable("W_%d" % i, [current_size, output_sizes[i]])
						expand_b = variable_scope.get_variable("b_%d" % i, [output_sizes[i]])
						output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
						output_data = tf.nn.elu(output_data)
						current_size = output_sizes[i]
					expand_W = variable_scope.get_variable("final_W", [current_size, 1])
					expand_b = variable_scope.get_variable("final_b" , [1])
					output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
					return output_data

			output_propensity_list = []
			for i in xrange(self.rank_list_size):
				# Add position information
				click_feature = [tf.expand_dims(tf.zeros_like(self.target_clicks[i]) , -1) for _ in xrange(self.rank_list_size)]
				click_feature[i] = tf.expand_dims(tf.ones_like(self.target_clicks[i]) , -1)
				if self.hparams.use_previous_clicks:
					click_feature.extend(previous_click_list)
					previous_click_list[i] = tf.expand_dims(self.target_clicks[i] , -1)
				if self.hparams.use_previous_rel_prob:
					click_feature.extend(previous_rel_prob_list)
					previous_rel_prob_list[i] = rel_prob_list[i]

				output_propensity_list.append(propensity_network(tf.concat(click_feature, 1), i))
			
		return tf.concat(output_propensity_list,1)

	def step(self, session, input_feed, forward_only):
		"""Run a step of the model feeding the given inputs.

		Args:
			session: tensorflow session to use.
			encoder_inputs: list of numpy [None, embedd_size] float vectors to feed as encoder inputs.
			decoder_inputs: list of numpy [None, embedd_size] float vectors to feed as decoder inputs.
			target_labels: list of numpy int vectors to feed as target re-ranking labels.
			target_weights: list of numpy float vectors to feed as target weights.
			bucket_id: which bucket of the model to use.
			forward_only: whether to do the backward step or only forward.

		Returns:
			A triple consisting of gradient norm (or None if we did not do backward),
			average perplexity, and the outputs.

		Raises:
			ValueError: if length of encoder_inputs, decoder_inputs, or
			target_weights disagrees with bucket size for the specified bucket_id.
		"""
		
		# Output feed: depends on whether we do a backward step or not.
		if not forward_only:
			output_feed = [self.updates,	# Update Op that does SGD.
							self.loss,	# Loss for this batch.
							self.summary # Summarize statistics.
							]	
		else:
			output_feed = [self.loss, # Loss for this batch.
						self.summary, # Summarize statistics.
						self.output   # Model outputs
			]	

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], None, outputs[-1]	# loss, no outputs, summary.
		else:
			return outputs[0], outputs[2], outputs[1]	# loss, outputs, summary.

	#def prepare_data_with_index(self, input_seq, output_seq, output_weights, output_initial_score, features, index, encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights):
	def prepare_data_with_index(self, data_set, index, encoder_inputs, decoder_targets, embeddings, decoder_clicks):
		i = index
		base = len(embeddings)
		for x in data_set.initial_list[i]:
			if x >= 0:
				embeddings.append(data_set.features[x])
		decoder_targets.append([x if data_set.gold_list[i][x] < 0 else data_set.gold_list[i][x] for x in xrange(len(data_set.gold_list[i]))])
		#if self.hparams.reverse_input:
		encoder_inputs.append(list([-1 if data_set.initial_list[i][x] < 0 else base+x for x in xrange(len(data_set.initial_list[i]))]))
			
		# Generate clicks with click models.
		gold_label_list = [0 if data_set.initial_list[i][x] < 0 else data_set.gold_weights[i][x] for x in xrange(len(data_set.initial_list[i]))]
		click_list, _, _ = self.click_model.sampleClicksForOneList(list(gold_label_list))
		while sum(click_list) == 0:
			click_list, _, _ = self.click_model.sampleClicksForOneList(list(gold_label_list))
		#click_list = list(gold_label_list) # debug
		decoder_clicks.append(click_list)
			
	#def get_batch(self, input_seq, output_seq, output_weights, output_initial_score, features):
	def get_batch(self, data_set):
		"""Get a random batch of data from the specified bucket, prepare for step.

		To feed data in step(..) it must be a list of batch-major vectors, while
		data here contains single length-major cases. So the main logic of this
		function is to re-index data cases to be in the proper format for feeding.

		Args:
			input_seq: a list of initial ranking ([0,1,2,3,4...])
			output_seq: the target ranking list ([2,3,1,0,4,...])
			output_weights: the weight list of each inputs
			features: a list of feature vectors for initial ranking list

		Returns:
			The triple (encoder_inputs, decoder_inputs, target_weights) for
			the constructed batch that has the proper format to call step(...) later.
		"""

		if len(data_set.initial_list[0]) != self.rank_list_size:
			raise ValueError("Input ranklist length must be equal to the one in bucket,"
							 " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
		length = len(data_set.initial_list)
		encoder_inputs, decoder_targets, embeddings, decoder_clicks = [], [], [], []
		
		rank_list_idxs = []
		for _ in xrange(self.batch_size):
			i = int(random.random() * length)
			rank_list_idxs.append(i)
			self.prepare_data_with_index(data_set, i,
								encoder_inputs, decoder_targets, embeddings, decoder_clicks)

		#self.start_index += self.batch_size

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		for length_idx in xrange(self.rank_list_size):
			# Batch encoder inputs are just re-indexed encoder_inputs.
			batch_encoder_inputs.append(
				np.array([encoder_inputs[batch_idx][length_idx]
					for batch_idx in xrange(self.batch_size)], dtype=np.float32))
			# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
			batch_targets.append(
				np.array([decoder_targets[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.int32))
			batch_clicks.append(
				np.array([decoder_clicks[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
		# Create others_map to store other information
		others_map = {
			'rank_list_idxs' : rank_list_idxs,
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
			'embeddings' : embeddings
		}

		return input_feed, others_map

	def get_next_batch(self, index, data_set):
		"""Get a random batch of data from the specified bucket, prepare for step.

		To feed data in step(..) it must be a list of batch-major vectors, while
		data here contains single length-major cases. So the main logic of this
		function is to re-index data cases to be in the proper format for feeding.

		Args:
			input_seq: a list of initial ranking ([0,1,2,3,4...])
			output_seq: the target ranking list ([2,3,1,0,4,...])
			output_weights: the weight list of each inputs
			features: a list of feature vectors for initial ranking list

		Returns:
			The triple (encoder_inputs, decoder_inputs, target_weights) for
			the constructed batch that has the proper format to call step(...) later.
		"""
		if len(data_set.initial_list[0]) != self.rank_list_size:
			raise ValueError("Input ranklist length must be equal to the one in bucket,"
							 " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
		length = len(data_set.initial_list)
		encoder_inputs, decoder_targets, embeddings, decoder_clicks = [], [], [], []
		
		for offset in xrange(self.batch_size):
			i = index + offset
			self.prepare_data_with_index(data_set, i, encoder_inputs, decoder_targets, 
				embeddings, decoder_clicks)

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		for length_idx in xrange(self.rank_list_size):
			# Batch encoder inputs are just re-indexed encoder_inputs.
			batch_encoder_inputs.append(
				np.array([encoder_inputs[batch_idx][length_idx]
					for batch_idx in xrange(self.batch_size)], dtype=np.float32))
			# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
			batch_targets.append(
				np.array([decoder_targets[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.int32))
			batch_clicks.append(
				np.array([decoder_clicks[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
		# Create others_map to store other information
		others_map = {
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
		}

		return input_feed, others_map

	def get_data_by_index(self, data_set, index): #not fixed
		"""Get one data from the specified index, prepare for step.

				Args:
					input_seq: a list of initial ranking ([0,1,2,3,4...])
					output_seq: the target ranking list ([2,3,1,0,4,...])
					output_weights: the weight list of each output
					features: a list of feature vectors for initial ranking list
					index: the index of the data

				Returns:
					The triple (encoder_inputs, decoder_inputs, target_weights) for
					the constructed batch that has the proper format to call step(...) later.
				"""
		if len(data_set.initial_list[0]) != self.rank_list_size:
			raise ValueError("Input ranklist length must be equal to the one in bucket,"
							 " %d != %d." % (len(data_set.initial_list[0]), self.rank_list_size))
		length = len(data_set.initial_list)
		encoder_inputs, decoder_targets, embeddings, decoder_clicks = [], [], [], []
		
		i = index
		self.prepare_data_with_index(data_set, i, encoder_inputs, decoder_targets, 
				embeddings, decoder_clicks)

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		for length_idx in xrange(self.rank_list_size):
			# Batch encoder inputs are just re-indexed encoder_inputs.
			batch_encoder_inputs.append(
				np.array([encoder_inputs[batch_idx][length_idx]
					for batch_idx in xrange(self.batch_size)], dtype=np.float32))
			# Batch decoder inputs are re-indexed decoder_inputs, we create weights.
			batch_targets.append(
				np.array([decoder_targets[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.int32))
			batch_clicks.append(
				np.array([decoder_clicks[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
		# Create others_map to store other information
		others_map = {
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
		}

		return input_feed, others_map

	def softmax_loss(self, output, target_inputs, target_clicks, propensity, name=None):
		loss = None
		with ops.name_scope(name, "softmax_loss",[output] + target_inputs + target_clicks):
			propensity_weights = tf.ones_like(propensity)
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			label_dis = clicks / tf.reduce_sum(clicks, 1, keep_dims=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(clicks, 1)
		return math_ops.reduce_sum(loss) / tf.reduce_sum(clicks), propensity_weights

	def click_weighted_softmax_cross_entropy_loss(self, output, target_inputs, target_clicks, propensity, name=None):
		loss = None
		with ops.name_scope(name, "click_softmax_cross_entropy",[output] + target_inputs + target_clicks):
			propensity_list = tf.split(self.logits_to_prob(propensity), self.rank_list_size, 1) # Compute propensity weights
			pw_list = []
			for i in xrange(self.rank_list_size):
				pw_i = propensity_list[0] / propensity_list[i]
				pw_list.append(pw_i)
			propensity_weights = tf.concat(pw_list, 1)
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			label_dis = clicks*propensity_weights / tf.reduce_sum(clicks*propensity_weights, 1, keep_dims=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(clicks*propensity_weights, 1)
		return math_ops.reduce_sum(loss) / tf.reduce_sum(clicks*propensity_weights), propensity_weights

	def click_weighted_log_loss(self, output, target_inputs, target_clicks, propensity, name=None):
		loss = None
		with ops.name_scope(name, "click_weighted_log_loss",[output] + target_inputs + target_clicks):
			propensity_list = tf.split(self.logits_to_prob(propensity), self.rank_list_size, 1) # Compute propensity weights
			pw_list = []
			for i in xrange(self.rank_list_size):
				pw_i = propensity_list[0] / propensity_list[i]
				pw_list.append(pw_i)
			propensity_weights = tf.concat(pw_list, 1)
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			click_prob = tf.sigmoid(output)
			loss = tf.losses.log_loss(clicks, click_prob, propensity_weights)
		return loss, propensity_weights

	def clip_by_each_value(self, t_list, clip_max_value = None, clip_min_value = None, name=None):
		if (not isinstance(t_list, collections.Sequence)
			or isinstance(t_list, six.string_types)):
			raise TypeError("t_list should be a sequence")
		t_list = list(t_list)

		with ops.name_scope(name, "clip_by_each_value",t_list + [clip_norm]) as name:
			values = [
					ops.convert_to_tensor(
							t.values if isinstance(t, ops.IndexedSlices) else t,
							name="t_%d" % i)
					if t is not None else t
					for i, t in enumerate(t_list)]

			values_clipped = []
			for i, v in enumerate(values):
				if v is None:
					values_clipped.append(None)
				else:
					t = None
					if clip_value_max != None:
						t = math_ops.minimum(v, clip_value_max)
					if clip_value_min != None:
						t = math_ops.maximum(t, clip_value_min, name=name)
					with ops.colocate_with(t):
						values_clipped.append(
								tf.identity(t, name="%s_%d" % (name, i)))

			list_clipped = [
					ops.IndexedSlices(c_v, t.indices, t.dense_shape)
					if isinstance(t, ops.IndexedSlices)
					else c_v
					for (c_v, t) in zip(values_clipped, t_list)]

		return list_clipped



