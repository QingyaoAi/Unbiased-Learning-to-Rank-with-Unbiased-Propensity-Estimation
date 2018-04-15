"""Training and testing the inverse propensity weighting algorithm for unbiased learning to rank.

See the following paper for more information on the dual learning algorithm.
	
	* Xuanhui Wang, Michael Bendersky, Donald Metzler, Marc Najork. 2016. Learning to Rank with Selection Bias in Personal Search. In Proceedings of SIGIR '16
	* Thorsten Joachims, Adith Swaminathan, Tobias Schnahel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of WSDM '17
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

def selu(x):
    with ops.name_scope('selu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

class IPWrank(object):
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

	def __init__(self, click_model, propensity_estimator, use_non_clicked_data, rank_list_size,
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
		self.click_model = click_model
		self.propensity_estimator = propensity_estimator
		self.use_non_clicked_data = use_non_clicked_data

		self.hparams = tf.contrib.training.HParams(
			learning_rate=0.5, 				# Learning rate.
			#learning_rate_decay_factor=0.8, # Learning rate decays by this much.
			max_gradient_norm=5.0,			# Clip gradients to this norm.
			#reverse_input=True,				# Set to True for reverse input sequences.
			hidden_layer_sizes=[512, 256, 128],		# Number of neurons in each layer of a RankNet. 
			loss_func='click_weighted_softmax_cross_entropy',			# Select Loss function
			l2_loss=0.0,					# Set strength for L2 regularization.
		)
		print(hparam_str)
		self.hparams.parse(hparam_str)

		self.start_index = 0
		self.count = 1
		self.rank_list_size = rank_list_size
		self.embed_size = embed_size
		self.batch_size = batch_size
		self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
		#self.learning_rate_decay_op = self.learning_rate.assign(
		#	self.learning_rate * self.hparams.learning_rate_decay_factor)
		self.global_step = tf.Variable(0, trainable=False)
		
		
		# Feeds for inputs.
		self.encoder_inputs = []
		self.embeddings = tf.placeholder(tf.float32, shape=[None, embed_size], name="embeddings")
		self.target_inputs = []
		self.target_clicks = []
		self.target_propensity_weights = []
		for i in xrange(self.rank_list_size):
			self.encoder_inputs.append(tf.placeholder(tf.int64, shape=[None],
											name="encoder{0}".format(i)))
			self.target_inputs.append(tf.placeholder(tf.int64, shape=[None],
											name="target{0}".format(i)))
			self.target_clicks.append(tf.placeholder(tf.float32, shape=[None],
											name="click{0}".format(i)))
			self.target_propensity_weights.append(tf.placeholder(tf.float32, shape=[None],
											name="propensity_weights{0}".format(i)))
		#self.PAD_embed = tf.get_variable("PAD_embed", [1,self.embed_size],dtype=tf.float32)
		self.PAD_embed = tf.zeros([1,self.embed_size],dtype=tf.float32)

		# Build model
		self.output = self.RankNet(forward_only)

		#self.output = self.output - tf.reduce_min(self.output,1,keep_dims=True)
		# Training outputs and losses.
		print('Loss Function is ' + self.hparams.loss_func)
		self.loss = None
		if self.hparams.loss_func == 'softmax':
			self.loss = self.softmax_loss(self.output, self.target_inputs, self.target_clicks, self.target_propensity_weights)
		elif self.hparams.loss_func == 'click_weighted_softmax_cross_entropy':
			self.loss = self.click_weighted_softmax_loss(self.output, self.target_inputs, self.target_clicks, self.target_propensity_weights)			
		else:
			self.loss = self.sigmoid_loss(self.output, self.target_inputs, self.target_clicks, self.target_propensity_weights)


		# Gradients and SGD update operation for training the model.
		params = tf.trainable_variables()
		if self.hparams.l2_loss > 0:
			for p in params:
				self.loss += self.hparams.l2_loss * tf.nn.l2_loss(p)
		if not forward_only:
			opt = tf.train.AdagradOptimizer(self.hparams.learning_rate)
			#opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.gradients = tf.gradients(self.loss, params)
			if self.hparams.max_gradient_norm > 0:
				self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
																	 self.hparams.max_gradient_norm)
				self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
											 global_step=self.global_step)
			else:
				self.norm = None #tf.norm(self.gradients)
				self.updates = opt.apply_gradients(zip(self.gradients, params),
											 global_step=self.global_step)
			tf.summary.scalar('Learning Rate', self.learning_rate)
		tf.summary.scalar('Loss', tf.reduce_mean(self.loss))

		self.summary = tf.summary.merge_all()
		self.saver = tf.train.Saver(tf.global_variables())


	def RankNet(self, forward_only=False, scope=None):
		with variable_scope.variable_scope(scope or "RankNet"):
			embeddings = tf.concat(axis=0,values=[self.embeddings,self.PAD_embed])
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
						expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]]) 
						expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
						output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
						#output_data = tf.layers.batch_normalization(output_data, training=forward_only)
						output_data = tf.nn.elu(output_data)
						current_size = output_sizes[i]
					return output_data
			for i in xrange(self.rank_list_size):
				encoder_embed.append(embedding_ops.embedding_lookup(embeddings, self.encoder_inputs[i]))
				output_scores.append(network(encoder_embed[-1], i))
			for i in xrange(self.rank_list_size):
				tf.summary.scalar('Avg Output Scores %d' % i, tf.reduce_mean(output_scores[i]))
			return tf.concat(output_scores,1)


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
	def prepare_data_with_index(self, data_set, index, encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights):
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
		# Estimate propensity with click models.
		propensity_weight = self.propensity_estimator.getPropensityForOneList(click_list, self.use_non_clicked_data)
		#propensity_weight = [1.0 for _ in xrange(len(click_list))] # debug

		decoder_propensity_weights.append(propensity_weight)
			

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
		encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights = [], [], [], [], []
		
		rank_list_idxs = []
		for _ in xrange(self.batch_size):
			i = int(random.random() * length)
			rank_list_idxs.append(i)
			self.prepare_data_with_index(data_set, i,
								encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights)

		#self.start_index += self.batch_size

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		batch_propensity_weights = []
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
			batch_propensity_weights.append(
				np.array([decoder_propensity_weights[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
			input_feed[self.target_propensity_weights[l].name] = batch_propensity_weights[l]
		# Create others_map to store other information
		others_map = {
			'rank_list_idxs' : rank_list_idxs,
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
			'propensity_weights' : decoder_propensity_weights,
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
		encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights = [], [], [], [], []
		
		for offset in xrange(self.batch_size):
			i = index + offset
			self.prepare_data_with_index(data_set, i, encoder_inputs, decoder_targets, 
				embeddings, decoder_clicks, decoder_propensity_weights)

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		batch_propensity_weights = []
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
			batch_propensity_weights.append(
				np.array([decoder_propensity_weights[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
			input_feed[self.target_propensity_weights[l].name] = batch_propensity_weights[l]
		# Create others_map to store other information
		others_map = {
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
			'propensity_weights' : decoder_propensity_weights
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
		encoder_inputs, decoder_targets, embeddings, decoder_clicks, decoder_propensity_weights = [], [], [], [], []
		
		i = index
		self.prepare_data_with_index(data_set, i, encoder_inputs, decoder_targets, 
				embeddings, decoder_clicks, decoder_propensity_weights)

		embedings_length = len(embeddings)
		for i in xrange(self.batch_size):
			for j in xrange(self.rank_list_size):
				if encoder_inputs[i][j] < 0:
					encoder_inputs[i][j] = embedings_length


		batch_encoder_inputs = []
		batch_clicks = []
		batch_targets = []
		batch_propensity_weights = []
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
			batch_propensity_weights.append(
				np.array([decoder_propensity_weights[batch_idx][length_idx]
						for batch_idx in xrange(self.batch_size)], dtype=np.float32))
		# Create input feed map
		input_feed = {}
		input_feed[self.embeddings.name] = np.array(embeddings)
		for l in xrange(self.rank_list_size):
			input_feed[self.encoder_inputs[l].name] = batch_encoder_inputs[l]
			input_feed[self.target_inputs[l].name] = batch_targets[l]
			input_feed[self.target_clicks[l].name] = batch_clicks[l]
			input_feed[self.target_propensity_weights[l].name] = batch_propensity_weights[l]
		# Create others_map to store other information
		others_map = {
			'input_list' : encoder_inputs,
			'click_list' : decoder_clicks,
			'propensity_weights' : decoder_propensity_weights
		}

		return input_feed, others_map

	
	def sigmoid_loss(self, output, target_inputs, target_clicks, target_propensity_weights, name=None):
		loss = None
		print(output.get_shape())
		with ops.name_scope(name, "sigmoid_loss",[output] + target_inputs + target_clicks + target_propensity_weights):
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			propensity_weights = tf.transpose(ops.convert_to_tensor(target_propensity_weights)) # (?, rank_list_size)
			original_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=clicks, logits=output)
			loss = original_loss * propensity_weights
		batch_size = tf.shape(target_clicks[0])[0]
		return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32) #/ (tf.reduce_sum(propensity_weights)+1)

	def softmax_loss(self, output, target_inputs, target_clicks, target_propensity_weights, name=None):
		loss = None
		with ops.name_scope(name, "softmax_loss",[output] + target_inputs + target_clicks + target_propensity_weights):
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			propensity_weights = tf.transpose(ops.convert_to_tensor(target_propensity_weights)) # (?, rank_list_size)
			#tmp = tf.reduce_sum(propensity_weights)
			#tmp = tf.Print(tmp, [tmp], 'this is tmp', summarize=5)
			#clicks = tf.nn.softmax(clicks)# debug
			#output = tf.Print(output, [output], 'this is output', summarize=10)
			#clicks = tf.Print(clicks, [clicks], 'this is clicks', summarize=10)
			#propensity_weights = tf.Print(propensity_weights, [propensity_weights], 'this is propensity_weights', summarize=10)
			#label_dis = clicks*propensity_weights / tf.reduce_sum(clicks*propensity_weights, 1, keep_dims=True)
			#label_dis = clicks / tf.reduce_sum(clicks, 1, keep_dims=True)
			#label_dis = tf.Print(label_dis, [label_dis], 'this is label_dis', summarize=10)
			
			label_dis = clicks / tf.reduce_sum(clicks, 1, keep_dims=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(clicks, 1)
			#list_weights = tf.reduce_prod(tf.maximum(propensity_weights, 1), 1, keep_dims=True)
			#list_weights = tf.Print(list_weights, [list_weights], 'this is list_weights', summarize=1)
			#loss = tf.Print(loss, [loss], 'this is loss', summarize=1)
			#loss = loss * list_weights
			#loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=clicks)
		#batch_size = tf.shape(target_clicks[0])[0]
		#return math_ops.reduce_sum(loss) / math_ops.cast(batch_size, dtypes.float32) / tf.reduce_sum(list_weights)
		return math_ops.reduce_sum(loss) / tf.reduce_sum(clicks)

	
	def click_weighted_softmax_loss(self, output, target_inputs, target_clicks, target_propensity_weights, name=None):
		loss = None
		with ops.name_scope(name, "softmax_loss",[output] + target_inputs + target_clicks + target_propensity_weights):
			clicks = tf.transpose(ops.convert_to_tensor(target_clicks)) # (?, rank_list_size)
			propensity_weights = tf.transpose(ops.convert_to_tensor(target_propensity_weights)) # (?, rank_list_size)
			label_dis = clicks*propensity_weights / tf.reduce_sum(clicks*propensity_weights, 1, keep_dims=True)
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=label_dis) * tf.reduce_sum(clicks*propensity_weights, 1)
		return math_ops.reduce_sum(loss) / tf.reduce_sum(clicks*propensity_weights)

	
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



