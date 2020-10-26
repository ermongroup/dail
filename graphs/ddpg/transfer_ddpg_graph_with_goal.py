import tensorflow as tf
import numpy as np
from pdb import set_trace
import time

from dail.model import *
from dail.sample import *

# Number of expert joints
LEA_NJOINTS = 2
EXP_NJOINTS = 2

#GOAL_SCALE = EXP_NJOINTS/LEA_NJOINTS
GOAL_SCALE = 1.

def transfer_ddpg_graph_with_goal(env, ph, params):
	'''
	Builds computation graph for learner policy including the expert policy
	Args:
		env : environments for expert and learner : dict
		ph : placeholders : dict of tf.placeholders
		params : parameters dictionary for constructing networks : dict
	Returns:d
		graph : dictionary of computation graph nodes : dict
	'''
	graph = {}

	# Make sure that expert graph is set up  first
	for d_ in sorted(env.keys()):
		graph[d_] = {}
		trans_d_ = 'learner' if d_ == 'expert' else 'expert'

		# ========= SET SIZE MAP ==========
		if d_ == 'learner':
			# Set size map
			with tf.variable_scope('setsizemap', reuse=tf.AUTO_REUSE):
				# Q function for set sizes with an output size of max_set_size
				set_q = feedforward(in_node=ph[d_]['state'], is_training=ph[d_]['is_training'],
									params=params[d_]['setsizemap'], scope=d_)

				next_set_q = feedforward(in_node=ph[d_]['next_state'], is_training=ph[d_]['is_training'],
										 params=params[d_]['setsizemap'], scope=d_)

				# Log probabilities over set sizes
				graph[d_]['set_probs'] = tf.nn.softmax(set_q)
				set_logits = tf.log(tf.nn.softmax(set_q))
				next_set_logits = tf.log(tf.nn.softmax(next_set_q))

				# Sampled set size for actor: [batch, 1]
				graph[d_]['set_size_sample'] = eps_greedy_sample(set_logits, ph[d_]['epsilon'])
				graph[d_]['next_set_size_sample'] = eps_greedy_sample(next_set_logits, ph[d_]['epsilon'])

				# Setsizemap qvalues: [batch, 1]
				mask = tf.one_hot(ph[d_]['set_size'], depth=params['train']['max_set_size'])
				graph[d_]['qvalue_setsize'] = tf.reduce_sum(set_q * mask, axis=1, keepdims=True)


			# Target network for setsizemap
			with tf.variable_scope('slow_setsizemap', reuse=False):
				# Q function for set sizes with an output size of max_set_size
				slow_set_q = feedforward(in_node=ph[d_]['next_state'], is_training=ph[d_]['is_training'],
										 params=params[d_]['setsizemap'], scope=d_)
				slow_set_logits = tf.log(tf.nn.softmax(slow_set_q))

				# Target sampled set size for slow actor: [batch, 1]
				graph[d_]['slow_set_size_sample'] = eps_greedy_sample(slow_set_logits, ph[d_]['epsilon']) ## TODO: Replace sampling function

				# Target setsize qvalue for training setsizemap: [batch, 1]
				graph[d_]['slow_qvalue_setsize'] = tf.reduce_max(slow_set_q, axis=1, keepdims=True)


		# ========= ACTOR ==========
		# Expert policy
		if d_ == 'expert':
			graph[d_]['action'] = feedforward(in_node=ph[d_]['state'],
											  is_training=ph[d_]['is_training'],
											  params=params[d_]['actor'],
											  scope='actor/'+d_, scale=True,
											  scale_fn=scale_action,
											  scale_params=env[d_]['env'])

		# Learner policy
		else:
			with tf.variable_scope('', reuse=tf.AUTO_REUSE):

				# Statemapping from learner to expert space
				agent_state = tf.concat([ph[d_]['state'][:, :2*LEA_NJOINTS],
										 ph[d_]['state'][:, 2*LEA_NJOINTS+2:]], axis=1)
				goal_state = ph[d_]['state'][:, 2*LEA_NJOINTS:2*LEA_NJOINTS+2]*GOAL_SCALE

				graph[d_]['mapped_agent_state'] = feedforward(in_node=agent_state,
															  is_training=ph[d_]['is_training'],
															  params=params[d_]['statemap'],
															  scope='actor/'+d_+'/statemap',
															  scale=params['train']['scale_state'],
															  scale_fn=scale_state, scale_params=env[trans_d_]['env'])

				graph[d_]['mapped_state'] = tf.concat([graph[d_]['mapped_agent_state'][:, :2*EXP_NJOINTS],
													   goal_state,
													   graph[d_]['mapped_agent_state'][:, 2*EXP_NJOINTS:]],
													   axis=1)


				# Should the set size be obtained from target Q or real Q? Should it be a ph or real time sampled value
				# Split masks for each timestep
				mask = tf.one_hot(tf.squeeze(graph[d_]['set_size_sample'], axis=1),
								  depth=params['train']['max_set_size']) # [batch, max_set_size]
				split_mask = tf.split(mask, params['train']['max_set_size'], axis=1)
				assert len(split_mask) == params['train']['max_set_size']

				# Rollout expert trajectory until max_set_size
				mapped_set = [graph[d_]['mapped_state']]
				graph[d_]['mapped_state_end'] = split_mask[0]*graph[d_]['mapped_state']

				for t in range(1, params['train']['max_set_size']):
					a = feedforward(in_node=mapped_set[-1], is_training=ph[d_]['is_training'],
									params=params[trans_d_]['actor'], scope='actor/'+trans_d_,
									scale=True, scale_fn=scale_action, scale_params=env[trans_d_]['env'])

					sa = tf.concat([mapped_set[-1], a], axis=1)

					# Feed through expert dynamics model
					next_state = feedforward(in_node=sa, is_training=ph[d_]['is_training'],
											 params=params[trans_d_]['model'], scope='model/'+trans_d_)

					# Whether to add the timestep state
					graph[d_]['mapped_state_end'] += split_mask[t] * next_state
					mapped_set.append(next_state)


				# Feed last state set thorugh expert policy
				graph[d_]['premap_action'] = feedforward(in_node=graph[d_]['mapped_state_end'],
														 is_training=ph[d_]['is_training'],
														 params=params[trans_d_]['actor'],
														 scope='actor/'+d_+'/expert_pi',
														 scale=True, scale_fn=scale_action,
														 scale_params=env[trans_d_]['env'])


				# Map expert action to learner action via actionmap
				sa_action = tf.concat([graph[d_]['premap_action'], ph[d_]['state']], axis=1)

				graph[d_]['action'] = feedforward(in_node=graph[d_]['premap_action'],
												  is_training=ph[d_]['is_training'],
												  params=params[d_]['actionmap'],
												  scope='actor/'+d_+'/actionmap',
												  scale=True, scale_fn=scale_action,
												  scale_params=env[d_]['env'])


		# ========= SLOW TARGET ACTOR ==========
		# Expert policy
		if d_ == 'expert':
			graph[d_]['slow_target_action'] = feedforward(in_node=ph[d_]['next_state'],
														  is_training=ph[d_]['is_training'],
														  params=params[d_]['actor'],
														  scope='slow_target_actor/'+d_, scale=True,
														  scale_fn=scale_action,
														  scale_params=env[d_]['env'])

		# Learner policy
		else:
			with tf.variable_scope('', reuse=tf.AUTO_REUSE):

				next_agent_state = tf.concat([ph[d_]['next_state'][:, :2*LEA_NJOINTS],
											  ph[d_]['next_state'][:, 2*LEA_NJOINTS+2:]], axis=1)
				next_goal_state = ph[d_]['next_state'][:, 2*LEA_NJOINTS:2*LEA_NJOINTS+2]*GOAL_SCALE

				# Statemapping from learner to expert space
				mapped_agent_state = feedforward(in_node=next_agent_state,
												 is_training=ph[d_]['is_training'],
												 params=params[d_]['statemap'],
												 scope='slow_target_actor/'+d_+'/statemap',
												 scale=params['train']['scale_state'],
												 scale_fn=scale_state, scale_params=env[trans_d_]['env'])

				mapped_state = tf.concat([mapped_agent_state[:, :2*EXP_NJOINTS],
										  next_goal_state,
										  mapped_agent_state[:, 2*EXP_NJOINTS:]],
										  axis=1)

				# Should the set size be obtained from target Q or real Q?
				# Should it be a ph or real time sampled value?
				# Split masks for each timestep ### REMOVE four +1
				mask = tf.one_hot(tf.squeeze(graph[d_]['next_set_size_sample'], axis=1),
								  depth=params['train']['max_set_size']) # [batch, max_set_size]
				split_mask = tf.split(mask, params['train']['max_set_size'], axis=1)
				assert len(split_mask) == params['train']['max_set_size']

				# Rollout expert trajectory until max_set_size
				mapped_set = [mapped_state]
				mapped_state_end = split_mask[0]*mapped_state

				for t in range(1, params['train']['max_set_size']):
					a = feedforward(in_node=mapped_set[-1], is_training=ph[d_]['is_training'],
									params=params[trans_d_]['actor'], scope='actor/'+trans_d_,
									scale=True, scale_fn=scale_action, scale_params=env[trans_d_]['env'])

					sa = tf.concat([mapped_set[-1], a], axis=1)

					# Feed through expert dynamics model
					next_state = feedforward(in_node=sa, is_training=ph[d_]['is_training'],
											 params=params[trans_d_]['model'], scope='model/'+trans_d_)

					# Whether to add the timestep state
					mapped_state_end += split_mask[t] * next_state
					mapped_set.append(next_state)

				premap_action = feedforward(in_node=mapped_state_end,
											is_training=ph[d_]['is_training'],
											params=params[trans_d_]['actor'],
											scope='slow_target_actor/'+d_+'/expert_pi',
											scale=True, scale_fn=scale_action,
											scale_params=env[trans_d_]['env'])

				# Map expert action to learner action via actionmap
				next_sa_action = tf.concat([premap_action, ph[d_]['next_state']], axis=1)
				graph[d_]['slow_target_action'] = feedforward(in_node=premap_action,
															  is_training=ph[d_]['is_training'],
															  params=params[d_]['actionmap'],
															  scope='slow_target_actor/'+d_+'/actionmap',
															  scale=True, scale_fn=scale_action,
															  scale_params=env[d_]['env'])


		# ========= DYNAMICS MODEL ==========
		# Dynamics model for temporal backprop
		with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
			# Statemapping from learner to expert space
			njoints = EXP_NJOINTS if d_ == 'expert' else LEA_NJOINTS

			model_agent_state = tf.concat([ph[d_]['state'][:, :2*njoints],
										   ph[d_]['state'][:, 2*njoints+2:]], axis=1)
			model_next_goal_state = ph[d_]['state'][:, 2*njoints:2*njoints+2]

			sa_model = tf.concat([model_agent_state, ph[d_]['action']], axis=1)
			sa_raw = tf.concat([model_agent_state, ph[d_]['raw_action']], axis=1)

			# (for training the dynamics model)
			graph[d_]['model_next_state'] = feedforward(in_node=sa_model, is_training=ph[d_]['is_training'],
														params=params[d_]['model'], scope=d_)

			model_next_agent_state = feedforward(in_node=sa_raw, is_training=ph[d_]['is_training'],
												 params=params[d_]['model'], scope=d_)

			graph[d_]['model_raw_next_state'] = tf.concat([model_next_agent_state[:, :2*njoints],
														   model_next_goal_state,
														   model_next_agent_state[:, 2*njoints:]], axis=1)


		# ========= CRITIC ==========
		# Critic
		with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
			# Inputs to the Q function
			sa_critic = tf.concat([ph[d_]['state'], ph[d_]['action']], axis=1)
			sa_actor = tf.concat([ph[d_]['state'], graph[d_]['action']], axis=1)

			# Critic applied to state_ph and a given action (for training critic)
			graph[d_]['qvalue_critic'] = feedforward(in_node=sa_critic, is_training=ph[d_]['is_training'],
													 params=params[d_]['critic'], scope=d_)

			# Critic applied to state_ph and the current policy's outputted actions for state_ph
			# (for training actor via deterministic policy gradient)
			graph[d_]['qvalue_actor'] = feedforward(in_node=sa_actor, is_training=ph[d_]['is_training'],
													params=params[d_]['critic'], scope=d_)


	   # ========= SLOW TARGET CRITIC ==========
		# Slow target critic
		with tf.variable_scope('slow_target_critic', reuse=False):
			# Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
			sa_target = tf.concat([ph[d_]['next_state'], graph[d_]['slow_target_action']], axis=1)
			graph[d_]['qvalue_target'] = tf.stop_gradient(feedforward(in_node=sa_target, is_training=ph[d_]['is_training'],
																	  params=params[d_]['critic'], scope=d_))


	# N-step state prediction
	### TODO: add a is_done function when the done criterion is not
	### Just the timelimit
	d_ = 'learner'
	trans_d_ = 'expert'
	t_horizon = params['train']['tloss_horizon']
	graph[d_]['multi_mapped_next_state'] = []
	graph[d_]['multi_trans_next_state'] = []
	graph[d_]['multi_next_state'] = []
	graph[d_]['trans_action'] = []

	next_goal_state = ph[d_]['state'][:, 2*LEA_NJOINTS:2*LEA_NJOINTS+2]*GOAL_SCALE
	agent_state = tf.concat([ph[d_]['state'][:, :2*LEA_NJOINTS],
							 ph[d_]['state'][:, 2*LEA_NJOINTS+2:]], axis=1)
	action = graph[d_]['action']

	agent_trans_state = tf.concat([graph[d_]['mapped_state_end'][:, :2*EXP_NJOINTS],
								   graph[d_]['mapped_state_end'][:, 2*EXP_NJOINTS+2:]], axis=1)
	trans_action = graph[d_]['premap_action']

	# Get next state via dynamics model
	with tf.variable_scope('model', reuse=True):
		sa = tf.concat([agent_state, action], axis=1)
		sa_trans = tf.concat([agent_trans_state, trans_action], axis=1)

		# Next learner state
		next_state = feedforward(in_node=sa, is_training=ph[d_]['is_training'],
								 params=params[d_]['model'], scope=d_)


		# Next expert state
		trans_agent_next_state = feedforward(in_node=sa_trans, is_training=ph[d_]['is_training'],
											 params=params[trans_d_]['model'], scope=trans_d_)

		trans_next_state = tf.concat([trans_agent_next_state[:, :2*EXP_NJOINTS],
									  next_goal_state,
									  trans_agent_next_state[:, 2*EXP_NJOINTS:]],
									  axis=1)


	# Map learner next state to expert space
	with tf.variable_scope('actor', reuse=True):
		mapped_agent_next_state = feedforward(in_node=next_state, is_training=ph[d_]['is_training'],
											  params=params[d_]['statemap'], scope=d_+'/statemap',
											  scale=params['train']['scale_state'],
											  scale_fn=scale_state, scale_params=env[d_]['env'])


		mapped_next_state = tf.concat([mapped_agent_next_state[:, :2*EXP_NJOINTS],
									   next_goal_state,
									   mapped_agent_next_state[:, 2*EXP_NJOINTS:]],
									   axis=1)


	# Store relevant graph nodes at each time step
	graph[d_]['multi_mapped_next_state'].append(mapped_next_state)
	graph[d_]['multi_trans_next_state'].append(trans_next_state)
	graph[d_]['multi_next_state'].append(next_state)


	# Discriminator to force statemapping into distribution of expert
	d_ = 'learner'
	trans_d_ = 'expert'
	future_sa = graph[d_]['multi_mapped_next_state'][-1:]

	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		'''
		fake_agent_state = tf.concat([graph[d_]['mapped_state_end'][:, :2*EXP_NJOINTS],
									  graph[d_]['mapped_state_end'][:, 2*EXP_NJOINTS+2:]], axis=1)
		real_agent_state = tf.concat([ph[trans_d_]['state'][:, :2*EXP_NJOINTS],
									  ph[trans_d_]['state'][:, 2*EXP_NJOINTS+2:]], axis=1)
		'''
		sas_fake = tf.concat([graph[d_]['mapped_state_end'], graph[d_]['premap_action'], mapped_next_state], axis=1)
		sas_real = tf.concat([ph[trans_d_]['state'], ph[trans_d_]['raw_action'], graph[trans_d_]['model_raw_next_state']], axis=1)

		#sas_real = tf.concat([ph[trans_d_]['state'], ph[trans_d_]['action'], ph[trans_d_]['next_state']], axis=1)

		'''
		# Sampling based generator loss
		noisy_premap_action = graph[d_]['premap_action'] + noise
		sas_fake = tf.concat([graph[d_]['mapped_state_end'], graph[d_]['premap_action']] + future_sa, axis=1)
		sas_real = tf.concat([ph[trans_d_]['state'], ph[trans_d_]['action'], ph[trans_d_]['next_state']], axis=1)
		'''

		# Create mixed sample for WGAN gradient penalty
		eps = tf.random_uniform([], 0.0, 1.0)
		graph[d_]['mixed_sas'] = sas_mixed = eps * sas_real + (1 - eps) * sas_fake

		graph[d_]['fake_prob'] = feedforward(in_node=sas_fake, is_training=ph[d_]['is_training'],
											 params=params[d_]['discriminator'], scope=d_)
		graph[d_]['disc_reward'] = tf.sigmoid(graph[d_]['fake_prob'])
		graph[d_]['real_prob'] = feedforward(in_node=sas_real, is_training=ph[d_]['is_training'],
											 params=params[d_]['discriminator'], scope=d_)
		graph[d_]['mixed_prob'] = feedforward(in_node=sas_mixed, is_training=ph[d_]['is_training'],
											  params=params[d_]['discriminator'], scope=d_)
	return graph


def get_transfer_ddpg_with_goal_vars(env):
	'''
	Get variables pertinent to target definitions in the inclusive graph
	Args:
		env : environments for learner and actor : dict
	Returns:
		graph_vars : graph variables : dict
	'''

	graph_vars = {}
	for d_ in env.keys():
		graph_vars[d_] = {}
		trans_d_ = 'learner' if d_ == 'expert' else 'expert'

		# Actor, setsizemap grad vars
		if d_ == 'expert':
			graph_vars[d_]['actor_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_)

			# Variables for target network updates
			graph_vars[d_]['all_actor_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_)
			graph_vars[d_]['all_slow_actor_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor/'+d_)

		else:
			#graph_vars[d_]['actor_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/statemap')
			graph_vars[d_]['actor_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/actionmap')
			#graph_vars[d_]['actor_grad_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/expert_pi')
			print("only modify actionmap")
			#print("only modify statemap")

			# For transfer
			# graph_vars[d_]['actor_grad_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+trans_d_)


			# Variables for target network updates
			graph_vars[d_]['all_actor_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/statemap')
			graph_vars[d_]['all_actor_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/actionmap')
			graph_vars[d_]['all_actor_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/expert_pi')


			graph_vars[d_]['all_slow_actor_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor/'+d_+'/statemap')
			graph_vars[d_]['all_slow_actor_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor/'+d_+'/actionmap')
			graph_vars[d_]['all_slow_actor_vars'] += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor/'+d_+'/expert_pi')


			# Setsizemap grad vars
			graph_vars[d_]['setsizemap_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='setsizemap/'+d_)
			graph_vars[d_]['all_setsizemap_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='setsizemap/'+d_)
			graph_vars[d_]['all_slow_setsizemap_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_setsizemap/'+d_)


		# Critic, statemap, autoencoder grad vars
		graph_vars[d_]['critic_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/'+d_)
		graph_vars[d_]['all_critic_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/'+d_)
		graph_vars[d_]['all_slow_critic_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic/'+d_)

		# Statemap, autoencoder grad_vars
		graph_vars[d_]['statemap_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/statemap')
		graph_vars[d_]['auto_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor/'+d_+'/statemap') ### TODO: consider changing this bothways

		# Dynamics model vars
		graph_vars[d_]['model_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/'+d_)

		# Discriminator vars
		graph_vars[d_]['disc_grad_vars'] = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator/'+d_)

		# Variables to save
		all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		expert_save_vars = [var for var in all_vars if 'expert/' in var.name]
		learner_save_vars = [var for var in all_vars if ('learner' in var.name and 'expert_pi' not in var.name)]

	return graph_vars, expert_save_vars, learner_save_vars


def get_transfer_ddpg_with_goal_targets(env, ph, graph, var_dict, params):
	'''
	Get variables pertinent to target definitions in the exclusive graph
	Args:
		env : environments for learner and actor : dict
		graph : computation graph nodes : dict
		vars : variables relevant to target computation : dict
	Returns:
		targets : dictionary of target nodes : dict
	'''
	targets = {}
	gamma = params['train']['gamma']

	# Episode inc_op
	episodes = tf.Variable(0.0, trainable=False, name='episodes')
	episode_inc_op = episodes.assign_add(1)

	for d_ in env.keys():
		targets[d_] = {}

		# Setsizemap loss
		if d_ == 'learner':
			q_target = tf.expand_dims(ph[d_]['disc_reward'], 1) + \
					   tf.expand_dims(ph[d_]['is_not_terminal'], 1) * gamma * graph[d_]['slow_qvalue_setsize']
			q_error = q_target - graph[d_]['qvalue_setsize']
			setsizemap_loss = tf.reduce_mean(tf.square(q_error))

			lr_setsizemap = params[d_]['setsizemap']['lr']
			lr_decay_setsizemap = params[d_]['setsizemap']['lr_decay']
			setsizemap_op = tf.train.AdamOptimizer(lr_setsizemap*lr_decay_setsizemap**episodes)
			setsizemap_grads_and_vars = setsizemap_op.compute_gradients(loss=setsizemap_loss, var_list=var_dict[d_]['setsizemap_grad_vars'])
			setsizemap_train_op = setsizemap_op.apply_gradients(grads_and_vars=setsizemap_grads_and_vars)

		else:
			setsizemap_loss = tf.constant(0.)
			setsizemap_train_op = tf.constant(0.)

		# Critic loss
		# 1-step temporal difference errors
		td_target = tf.expand_dims(ph[d_]['reward'], 1) + tf.expand_dims(ph[d_]['is_not_terminal'], 1) * gamma * graph[d_]['qvalue_target']
		td_errors = td_target - graph[d_]['qvalue_critic']
		critic_loss = tf.reduce_mean(tf.square(td_errors))

		# Critic train op
		lr_critic = params[d_]['critic']['lr']
		lr_decay_critic = params[d_]['critic']['lr_decay']
		critic_op = tf.train.AdamOptimizer(lr_critic*lr_decay_critic**episodes)
		critic_grads_and_vars = critic_op.compute_gradients(loss=critic_loss, var_list=var_dict[d_]['critic_grad_vars'])
		critic_train_op = critic_op.apply_gradients(grads_and_vars=critic_grads_and_vars)

		# Actor loss (mean Q-values under current policy with regularization)
		if d_ == 'learner' and params[d_]['use_bc']:
			action_loss = tf.reduce_mean(tf.square(graph[d_]['action'] - graph['expert']['action']))
			#action_loss += -1*tf.reduce_mean(graph[d_]['qvalue_actor'])
		else:
			action_loss = -1*tf.reduce_mean(graph[d_]['qvalue_actor'])

		# Add temporal loss based on learned dynamics model
		if d_ == 'learner':
			### TODO: get rid of unncessary model map code in inclusive_graph
			### Deal with end of the episode predictions (add an is_done model)
			t_horizon = params['train']['tloss_horizon']
			temporal_loss = []
			for t in range(t_horizon):
				temp_diff = tf.square(graph[d_]['multi_mapped_next_state'][t] - graph[d_]['multi_trans_next_state'][t]) * \
							tf.expand_dims(ph[d_]['is_not_terminal'], 1)

				temporal_loss.append(tf.reduce_mean(temp_diff))

			if len(temporal_loss) == 1:
				temporal_loss = temporal_loss[0]

			# generator loss
			if params['train']['use_wgan'] or params['train']['use_grad_wgan']:
				gen_loss = -tf.reduce_mean(graph[d_]['fake_prob'])
			else:
				#gen_loss = -tf.reduce_mean(tf.log(tf.sigmoid(graph[d_]['fake_prob']) + 1e-8))
				gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['fake_prob'], labels=tf.ones_like(graph[d_]['fake_prob']))
				gen_loss = tf.reduce_mean(gen_loss)

			# Identity loss
			# eyeloss_state = tf.reduce_mean(tf.square(graph[d_]['mapped_state'] - ph[d_]['state']))

			# Aggregate actor loss
			actor_loss = gen_loss + 0.*action_loss #+ 0*eyeloss_state


		else:
			gen_loss = tf.constant(0)
			temporal_loss = tf.constant(0)
			actor_loss = action_loss

		'''
		# Orthogonal regularization
		for var in var_dict[d_]['actor_grad_vars']:
			if 'statemap' in var.name and 'bias' not in var.name:
				var_dim = tf.shape(var)[1]
				actor_loss += tf.square(tf.norm(tf.matmul(tf.transpose(var), var) - tf.eye(var_dim), ord='fro', axis=(0, 1)))
		'''

		# Actor train op
		# Gradient of mean Q-values w.r.t actor params is the deterministic policy gradient (keeping critic params fixed)
		lr_actor = params[d_]['actor']['lr']
		lr_decay_actor = params[d_]['actor']['lr_decay']
		if params['train']['use_wgan']:
			actor_op = tf.train.RMSPropOptimizer(0.1*lr_actor*lr_decay_actor**episodes)
		elif params['train']['use_grad_wgan']:
			actor_op = tf.train.AdamOptimizer(lr_actor*lr_decay_actor**episodes)
		else:
			actor_op = tf.train.AdamOptimizer(lr_actor*lr_decay_actor**episodes)

		actor_grads_and_vars = actor_op.compute_gradients(loss=actor_loss, var_list=var_dict[d_]['actor_grad_vars'])
		actor_train_op = actor_op.apply_gradients(grads_and_vars=actor_grads_and_vars)

		# Gradient clipping
		#gradients, variables = zip(*actor_op.compute_gradients(loss=actor_loss, var_list=var_dict[d_]['actor_grad_vars']))
		#gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
		#actor_train_op = actor_op.apply_gradients(zip(gradients, variables))

		'''
		# Statemap loss
		lr_statemap = params[d_]['statemap']['lr']
		lr_decay_statemap = params[d_]['statemap']['lr_decay']
		statemap_loss = tf.reduce_mean(tf.square(graph[d_]['remapped_state'] - graph[d_]['mapped_state']))

		# Identity loss for evaluation purposes only (total deviation)
		eyeloss_state = tf.reduce_mean(tf.square(graph[d_]['mapped_state'] - ph[d_]['state']))
		if d_ == 'learner':
			eyeloss_action = tf.reduce_mean(tf.square(graph[d_]['premap_action'] - graph[d_]['action']))
		else:
			eyeloss_action = tf.constant(0)

		statemap_op = tf.train.AdamOptimizer(lr_statemap*lr_decay_statemap**episodes)
		statemap_grads_and_vars = statemap_op.compute_gradients(loss=statemap_loss, var_list=var_dict[d_]['statemap_grad_vars'])
		statemap_train_op = statemap_op.apply_gradients(grads_and_vars=statemap_grads_and_vars)

		# Autoencoding loss
		lr_auto = params[d_]['auto']['lr']
		lr_decay_auto = params[d_]['auto']['lr_decay']
		auto_loss = tf.reduce_mean(tf.square(graph[d_]['inv_mapped_state'] - ph[d_]['state']))
		auto_op = tf.train.AdamOptimizer(lr_auto*lr_decay_auto**episodes)
		auto_grads_and_vars = auto_op.compute_gradients(loss=auto_loss, var_list=var_dict[d_]['auto_grad_vars'])
		auto_train_op = auto_op.apply_gradients(grads_and_vars=auto_grads_and_vars)
		'''

		# Dynamics model loss
		lr_model = params[d_]['model']['lr']
		lr_decay_model = params[d_]['model']['lr_decay']
		njoints = EXP_NJOINTS if d_ == 'expert' else LEA_NJOINTS
		model_target_state = tf.concat([ph[d_]['next_state'][:, :2*njoints],
										ph[d_]['next_state'][:, 2*njoints+2:]], axis=1)
		model_diff = tf.square(graph[d_]['model_next_state'] - model_target_state) \
					 * tf.expand_dims(ph[d_]['is_not_terminal'], 1)
		model_loss = tf.reduce_mean(model_diff)
		model_op = tf.train.AdamOptimizer(lr_model*lr_decay_model**episodes)
		model_grads_and_vars = model_op.compute_gradients(loss=model_loss, var_list=var_dict[d_]['model_grad_vars'])
		model_train_op = model_op.apply_gradients(grads_and_vars=model_grads_and_vars)

		# Discriminator loss
		if d_ == 'learner':
			lr_disc = params[d_]['discriminator']['lr']
			lr_decay_disc = params[d_]['discriminator']['lr_decay']

			if params['train']['use_wgan']:
				disc_loss = -tf.reduce_mean(graph[d_]['real_prob']) + tf.reduce_mean(graph[d_]['fake_prob'])
				disc_op = tf.train.RMSPropOptimizer(0.1*lr_disc*lr_decay_disc**episodes)
				disc_grads_and_vars = disc_op.compute_gradients(loss=disc_loss*ph[d_]['train_disc'], var_list=var_dict[d_]['disc_grad_vars'])
				disc_train_op = [disc_op.apply_gradients(grads_and_vars=disc_grads_and_vars)]
				disc_weight_clip = [var.assign(tf.clip_by_value(var, -0.1, 0.1)) for var in var_dict[d_]['disc_grad_vars']]
				disc_train_op = disc_train_op + disc_weight_clip

			elif params['train']['use_grad_wgan']:
				# Gradient penalty
				grad = tf.gradients(graph[d_]['mixed_prob'], graph[d_]['mixed_sas'])
				grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1))
				grad_penalty = tf.reduce_mean(tf.square(grad_norm - 1.0))

				# WGAN loss
				disc_loss = -tf.reduce_mean(graph[d_]['real_prob']) + tf.reduce_mean(graph[d_]['fake_prob'])
				disc_loss += grad_penalty
				disc_op = tf.train.AdamOptimizer(lr_disc*lr_decay_disc**episodes)
				disc_grads_and_vars = disc_op.compute_gradients(loss=disc_loss*ph[d_]['train_disc'], var_list=var_dict[d_]['disc_grad_vars'])
				disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads_and_vars)

			else:
				# Add a entropy regularizer to the discriminator
				logits = tf.concat([graph[d_]['fake_prob'], graph[d_]['real_prob']], 0)
				entropy_loss = -tf.reduce_mean(logit_bernoulli_entropy(logits))

				'''
				disc_loss = -tf.reduce_mean(tf.log(tf.sigmoid(graph[d_]['real_prob']) + 1e-8) + \
											tf.log(1. - tf.sigmoid(graph[d_]['fake_prob']) + 1e-8))
				'''

				fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['fake_prob'], labels=tf.zeros_like(graph[d_]['fake_prob']))
				fake_loss = tf.reduce_mean(fake_loss)
				real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['real_prob'], labels=tf.ones_like(graph[d_]['real_prob']))
				real_loss = tf.reduce_mean(real_loss)
				disc_loss = fake_loss + real_loss + 0.0005*entropy_loss

				#disc_loss += entropy_loss

				disc_op = tf.train.AdamOptimizer(lr_disc*lr_decay_disc**episodes)
				disc_grads_and_vars = disc_op.compute_gradients(loss=disc_loss*ph[d_]['train_disc'], var_list=var_dict[d_]['disc_grad_vars'])
				disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads_and_vars)

		else:
			disc_train_op = tf.constant(0)
			disc_loss = tf.constant(0)

		# Aggregate training targets
		targets[d_]['train'] = {'actor_train_op': actor_train_op,
								#'setsizemap_train_op': setsizemap_train_op,
								'critic_train_op': critic_train_op,
								#'model_train_op': model_train_op,
								'disc_train_op': disc_train_op,
								'model_train_op': model_train_op,
								'model_loss': model_loss,
								#'auto_train_op': auto_train_op,
								#'statemap_train_op': statemap_train_op,
								'action_loss': action_loss,
								#'setsizemap_loss': setsizemap_loss,
								#'model_loss': model_loss,
								'disc_loss': disc_loss,
								'gen_loss': gen_loss,
								#'smap_loss': statemap_loss,
								'temp_loss': temporal_loss}
								#'eyeloss_state': eyeloss_state,
								#'eyeloss_action': eyeloss_action,
								#'statemap_weights': var_dict[d_]['all_actor_vars']}
								#'auto_loss': auto_loss}

		'''
		targets[d_]['model'] = {'model_train_op': model_train_op,
								'model_loss': model_loss}
		'''


		#targets[d_]['train'].pop('auto_loss')
		#targets[d_]['train'].pop('auto_train_op')

		'''
		if d_ == 'learner':
			targets[d_]['train'].pop('actor_train_op')
			targets[d_]['train'].pop('critic_train_op')
		'''

		# Update the target
		targets[d_]['update'] = []
		tau = params['train']['tau']

		assert len(var_dict[d_]['all_slow_actor_vars']) == len(var_dict[d_]['all_actor_vars'])
		assert len(var_dict[d_]['all_slow_critic_vars']) == len(var_dict[d_]['all_critic_vars'])

		for i, slow_target_actor_var in enumerate(var_dict[d_]['all_slow_actor_vars']):
			actor_var = var_dict[d_]['all_actor_vars'][i]
			update_slow_target_actor_op = slow_target_actor_var.assign(tau*actor_var+(1-tau)*slow_target_actor_var)
			targets[d_]['update'].append(update_slow_target_actor_op)

			'''
			print(slow_target_actor_var)
			print(actor_var)
			print('\n')
			'''

		for i, slow_target_var in enumerate(var_dict[d_]['all_slow_critic_vars']):
			critic_var = var_dict[d_]['all_critic_vars'][i]
			update_slow_target_critic_op = slow_target_var.assign(tau*critic_var+(1-tau)*slow_target_var)
			targets[d_]['update'].append(update_slow_target_critic_op)

			'''
			print(slow_target_var)
			print(critic_var)
			print('\n')
			'''

		if d_ == 'learner':
			assert len(var_dict[d_]['all_slow_setsizemap_vars']) == len(var_dict[d_]['all_setsizemap_vars'])
			for i, slow_target_setsize_var in enumerate(var_dict[d_]['all_slow_setsizemap_vars']):
				setsize_var = var_dict[d_]['all_setsizemap_vars'][i]
				update_slow_target_setsize_op = slow_target_setsize_var.assign(tau*setsize_var+(1-tau)*slow_target_setsize_var)
				targets[d_]['update'].append(update_slow_target_setsize_op)

		targets[d_]['update'] = tf.group(*targets[d_]['update'], name='update_slow_targets')


	# Episode count increment op
	targets['episode_inc_op'] = episode_inc_op

	return targets
