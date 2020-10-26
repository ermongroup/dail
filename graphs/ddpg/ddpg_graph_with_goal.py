import tensorflow as tf
import numpy as np
from pdb import set_trace
import time

from dail.model import *


#GOAL_SCALE = EXP_NJOINTS/LEA_NJOINTS
GOAL_SCALE = 1.


def ddpg_graph_with_goal(env, ph, params):
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

    # Set the number of joints parameter
    if 'reacher2' in env['expert']['name']:
        EXP_NJOINTS = 2
    elif 'reacher3' in env['expert']['name']:
        EXP_NJOINTS = 3
    else:
        print('[ddpg_graph_with_goal.py] ERROR: unrecognized expert env name {}'.format(env['expert']['name']))

    if 'reacher2' in env['learner']['name']:
        LEA_NJOINTS = 2
    elif 'reacher3' in env['learner']['name']:
        LEA_NJOINTS = 3
    else:
        print('[ddpg_graph_with_goal.py] ERROR: unrecognized learner env name {}'.format(env['learner']['name']))


    # Make sure that expert graph is set up  first
    for d_ in sorted(env.keys()):
        graph[d_] = {}
        trans_d_ = 'learner' if d_ == 'expert' else 'expert'

        # ========= ACTOR ==========
        # Expert policy
        if d_ == 'expert':
            graph[d_]['action'] = feedforward(in_node=ph[d_]['state'],
                                              is_training=ph[d_]['is_training'],
                                              params=params[d_]['actor'],
                                              scope='actor/'+d_, scale=True,
                                              scale_fn=scale_action,
                                              scale_params=env[d_]['env'])

        # Self policy
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

                graph[d_]['mapped_state_end'] = graph[d_]['mapped_state']


                # Inverse statemap from expert agent state to learner agent state
                graph[d_]['inv_agent_state'] = feedforward(in_node=graph[d_]['mapped_agent_state'],
                                                           is_training=ph[d_]['is_training'],
                                                           params=params[d_]['invmap'],
                                                           scope='actor/'+d_+'/invmap',
                                                           scale=params['train']['scale_state'],
                                                           scale_fn=scale_state, scale_params=env[d_]['env'])


                # Feed last state set thorugh expert policy
                graph[d_]['premap_action'] = feedforward(in_node=graph[d_]['mapped_state_end'],
                                                         is_training=ph[d_]['is_training'],
                                                         params=params[trans_d_]['actor'],
                                                         scope='actor/'+trans_d_,
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


                # Feed expert states through inverse map for GAMA-DA
                expert_agent_state =tf.concat([ph[trans_d_]['state'][:, :2*EXP_NJOINTS],
                                               ph[trans_d_]['state'][:, 2*EXP_NJOINTS+2:]], axis=1)
                expert_goal = ph[trans_d_]['state'][:, 2*EXP_NJOINTS:2*EXP_NJOINTS+2]*GOAL_SCALE

                mapped_lea_agent_state = feedforward(in_node=expert_agent_state,
                                                     is_training=ph[d_]['is_training'],
                                                     params=params[d_]['invmap'],
                                                     scope='actor/'+d_+'/invmap',
                                                     scale=params['train']['scale_state'],
                                                     scale_fn=scale_state, scale_params=env[d_]['env'])

                graph[trans_d_]['mapped_state'] = tf.concat([mapped_lea_agent_state[:, :2*LEA_NJOINTS],
                                                             expert_goal,
                                                             mapped_lea_agent_state[:, 2*LEA_NJOINTS:]],
                                                             axis=1)

                graph[trans_d_]['mapped_action'] = feedforward(in_node=ph[trans_d_]['action'],
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
                mapped_state_end = mapped_state


                premap_action = feedforward(in_node=mapped_state_end,
                                            is_training=ph[d_]['is_training'],
                                            params=params[trans_d_]['actor'],
                                            scope='actor/'+trans_d_,
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
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            # Statemapping from learner to expert space
            njoints = EXP_NJOINTS if d_ == 'expert' else LEA_NJOINTS

            model_agent_state = tf.concat([ph[d_]['state'][:, :2*njoints],
                                           ph[d_]['state'][:, 2*njoints+2:]], axis=1)
            model_next_goal_state = ph[d_]['state'][:, 2*njoints:2*njoints+2]

            sa_model = tf.concat([model_agent_state, ph[d_]['action']], axis=1)
            sa_raw = tf.concat([model_agent_state, ph[d_]['raw_action']], axis=1)

            # (for training the dynamics model)
            graph[d_]['model_next_state'] = feedforward(in_node=sa_model,
                                                        is_training=ph[d_]['is_training'],
                                                        params=params[d_]['model'],
                                                        scope=d_)

            model_next_agent_state = feedforward(in_node=sa_raw,
                                                 is_training=ph[d_]['is_training'],
                                                 params=params[d_]['model'],
                                                 scope=d_)

            graph[d_]['model_raw_next_state'] = tf.concat([model_next_agent_state[:, :2*njoints],
                                                           model_next_goal_state,
                                                           model_next_agent_state[:, 2*njoints:]],
                                                          axis=1)


        # ========= CRITIC ==========
        with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
            # Inputs to the Q function
            sa_critic = tf.concat([ph[d_]['state'], ph[d_]['action']], axis=1)
            sa_actor = tf.concat([ph[d_]['state'], graph[d_]['action']], axis=1)

            # Critic applied to state_ph and a given action (for training critic)
            graph[d_]['qvalue_critic'] = feedforward(in_node=sa_critic,
                                                     is_training=ph[d_]['is_training'],
                                                     params=params[d_]['critic'],
                                                     scope=d_)

            # Critic applied to state_ph and the current policy's outputted actions for state_ph
            # (for training actor via deterministic policy gradient)
            graph[d_]['qvalue_actor'] = feedforward(in_node=sa_actor,
                                                    is_training=ph[d_]['is_training'],
                                                    params=params[d_]['critic'],
                                                    scope=d_)


        # ========= SLOW TARGET CRITIC ==========
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
            sa_target = tf.concat([ph[d_]['next_state'], graph[d_]['slow_target_action']], axis=1)
            graph[d_]['qvalue_target'] = tf.stop_gradient(feedforward(in_node=sa_target,
                                                                      is_training=ph[d_]['is_training'],
                                                                      params=params[d_]['critic'],
                                                                      scope=d_))


    #============== COMPUTE MAPPED TRIPLETS ==============
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
                                   graph[d_]['mapped_state_end'][:, 2*EXP_NJOINTS+2:]],
                                  axis=1)
    trans_action = graph[d_]['premap_action']

    # Get next state via dynamics model
    with tf.variable_scope('model', reuse=True):
        sa = tf.concat([agent_state, action], axis=1)
        sa_trans = tf.concat([agent_trans_state, trans_action], axis=1)

        # Next learner state
        next_state = feedforward(in_node=sa,
                                 is_training=ph[d_]['is_training'],
                                 params=params[d_]['model'],
                                 scope=d_)


        # Next expert state
        trans_agent_next_state = feedforward(in_node=sa_trans,
                                             is_training=ph[d_]['is_training'],
                                             params=params[trans_d_]['model'],
                                             scope=trans_d_)

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
        # real and fake samples
        sas_fake = tf.concat([graph[d_]['mapped_state_end'],
                              graph[d_]['premap_action'],
                              mapped_next_state], axis=1)
        sas_real = tf.concat([ph[trans_d_]['state'],
                              ph[trans_d_]['action'],
                              ph[trans_d_]['next_state']],
                             axis=1)

        graph[d_]['fake_prob'] = feedforward(in_node=sas_fake,
                                             is_training=ph[d_]['is_training'],
                                             params=params[d_]['discriminator'],
                                             scope=d_)
        graph[d_]['disc_reward'] = tf.sigmoid(graph[d_]['fake_prob'])
        graph[d_]['real_prob'] = feedforward(in_node=sas_real,
                                             is_training=ph[d_]['is_training'],
                                             params=params[d_]['discriminator'],
                                             scope=d_)
    return graph



def get_ddpg_with_goal_vars(env):
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

        # Actor grad vars
        glob_vars = tf.GraphKeys.GLOBAL_VARIABLES
        if d_ == 'expert':
            graph_vars[d_]['actor_grad_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_)

            # Variables for target network updates
            graph_vars[d_]['all_actor_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_)
            graph_vars[d_]['all_slow_actor_vars'] = tf.get_collection(glob_vars, scope='slow_target_actor/'+d_)

        else:
            graph_vars[d_]['actor_grad_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_+'/statemap')
            graph_vars[d_]['actor_grad_vars'] += tf.get_collection(glob_vars, scope='actor/'+d_+'/actionmap')

            # For inverse statemap
            graph_vars[d_]['auto_grad_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_+'/invmap')

            # Variables for target network updates
            graph_vars[d_]['all_actor_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_+'/statemap')
            graph_vars[d_]['all_actor_vars'] += tf.get_collection(glob_vars, scope='actor/'+d_+'/actionmap')
            graph_vars[d_]['all_slow_actor_vars'] = tf.get_collection(glob_vars, scope='slow_target_actor/'+d_+'/statemap')
            graph_vars[d_]['all_slow_actor_vars'] += tf.get_collection(glob_vars, scope='slow_target_actor/'+d_+'/actionmap')



        # Critic, statemap, autoencoder grad vars
        graph_vars[d_]['critic_grad_vars'] = tf.get_collection(glob_vars, scope='critic/'+d_)
        graph_vars[d_]['all_critic_vars'] = tf.get_collection(glob_vars, scope='critic/'+d_)
        graph_vars[d_]['all_slow_critic_vars'] = tf.get_collection(glob_vars, scope='slow_target_critic/'+d_)

        # Statemap, autoencoder grad_vars
        graph_vars[d_]['statemap_grad_vars'] = tf.get_collection(glob_vars, scope='actor/'+d_+'/statemap')

        # Dynamics model vars
        graph_vars[d_]['model_grad_vars'] = tf.get_collection(glob_vars, scope='model/'+d_)

        # Discriminator vars
        graph_vars[d_]['disc_grad_vars'] = tf.get_collection(glob_vars, scope='discriminator/'+d_)

        # Variables to save
        all_vars = tf.get_collection(glob_vars)
        expert_save_vars = [var for var in all_vars if 'expert' in var.name]
        learner_save_vars = [var for var in all_vars if 'learner' in var.name]


    return graph_vars, expert_save_vars, learner_save_vars



def get_ddpg_with_goal_targets(env, ph, graph, var_dict, params):
        '''
        Get variables pertinent to target definitions in the exclusive graph
        Args:
                env : environments for learner and actor : dict
                graph : computation graph nodes : dict
                vars : variables relevant to target computation : dict
        Returns:
                targets : dictionary of target nodes : dict
        '''

        # Set the number of joints parameter
        if 'reacher2' in env['expert']['name']:
            EXP_NJOINTS = 2
        elif 'reacher3' in env['expert']['name']:
            EXP_NJOINTS = 3
        else:
            print('[ddpg_graph_with_goal.py] ERROR: unrecognized expert env name {}'.format(env['expert']['name']))

        if 'reacher2' in env['learner']['name']:
            LEA_NJOINTS = 2
        elif 'reacher3' in env['learner']['name']:
            LEA_NJOINTS = 3
        else:
            print('[ddpg_graph_with_goal.py] ERROR: unrecognized learner env name {}'.format(env['learner']['name']))


        targets = {}
        gamma = params['train']['gamma']

        # Episode inc_op
        episodes = tf.Variable(0.0, trainable=False, name='episodes')
        episode_inc_op = episodes.assign_add(1)

        for d_ in env.keys():
            targets[d_] = {}

            #============= RL Loss for Expert / BC Loss for Learner =============
            # Critic loss
            # 1-step temporal difference errors
            td_target = tf.expand_dims(ph[d_]['reward'], 1) \
                        + tf.expand_dims(ph[d_]['is_not_terminal'], 1) * gamma * graph[d_]['qvalue_target']
            td_errors = td_target - graph[d_]['qvalue_critic']
            critic_loss = tf.reduce_mean(tf.square(td_errors))

            # Critic train op
            lr_critic = params[d_]['critic']['lr']
            lr_decay_critic = params[d_]['critic']['lr_decay']
            critic_op = tf.train.AdamOptimizer(lr_critic*lr_decay_critic**episodes)
            critic_grads_and_vars = critic_op.compute_gradients(loss=critic_loss,
                                                                var_list=var_dict[d_]['critic_grad_vars'])
            critic_train_op = critic_op.apply_gradients(grads_and_vars=critic_grads_and_vars)

            # Actor loss (mean Q-values under current policy with regularization)
            if d_ == 'learner' and params[d_]['use_bc']:
                action_loss = tf.reduce_mean(tf.square(graph[d_]['action'] - ph[d_]['raw_action']))
            else:
                action_loss = -1*tf.reduce_mean(graph[d_]['qvalue_actor'])



            #============= Distribution Matching Loss =============
            if d_ == 'learner':
                # Temporal loss is good to monitor to see how well distributions are matched
                t_horizon = params['train']['tloss_horizon']
                temporal_loss = []
                for t in range(t_horizon):
                    temp_diff = tf.square(graph[d_]['multi_mapped_next_state'][t] - graph[d_]['multi_trans_next_state'][t]) * \
                                            tf.expand_dims(ph[d_]['is_not_terminal'], 1)

                    temporal_loss.append(tf.reduce_mean(temp_diff))

                if len(temporal_loss) == 1:
                    temporal_loss = temporal_loss[0]

                # Generator loss
                gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['fake_prob'],
                                                                   labels=tf.ones_like(graph[d_]['fake_prob']))
                gen_loss = tf.reduce_mean(gen_loss)

                # Identity loss
                # eyeloss_state = tf.reduce_mean(tf.square(graph[d_]['mapped_state'] - ph[d_]['state']))

                #========================== Aggregate into GAMA Loss ========================
                if 'reacher2_act' in env['learner']['name']:
                    gen_weight = 0.001 * ph[d_]['train_disc']
                    actor_loss = gen_weight*gen_loss + 100*action_loss #+ 0*eyeloss_state
                else:
                    gen_weight = 0.001 * ph[d_]['train_disc']
                    actor_loss = gen_weight*gen_loss + action_loss #+ 0*eyeloss_state

            else:
                gen_loss = tf.constant(0)
                temporal_loss = tf.constant(0)
                actor_loss = action_loss

                # Behavioral cloning loss
                bc_loss = tf.reduce_mean(tf.square(graph[d_]['action'] - ph[d_]['action_tv']))


            #============ Expert DDPG Train Operation / GAMA Train Operation ============
            lr_actor = params[d_]['actor']['lr']
            lr_decay_actor = params[d_]['actor']['lr_decay']
            actor_op = tf.train.AdamOptimizer(lr_actor*lr_decay_actor**episodes)
            actor_grads_and_vars = actor_op.compute_gradients(loss=actor_loss, var_list=var_dict[d_]['actor_grad_vars'])
            actor_train_op = actor_op.apply_gradients(grads_and_vars=actor_grads_and_vars)


            #============= Behavioral cloning for target expert ===============
            if d_ == 'expert':
                lr_bc = params['bc']['lr']
                bc_op = tf.train.AdamOptimizer(lr_bc)
                bc_grads_and_vars = bc_op.compute_gradients(loss=bc_loss, var_list=var_dict[d_]['actor_grad_vars'])
                bc_train_op = bc_op.apply_gradients(grads_and_vars=bc_grads_and_vars)


            #============= Autoencoding loss ==============
            if d_ == 'learner':
                lr_auto = params[d_]['invmap']['lr']
                lr_decay_auto = params[d_]['invmap']['lr_decay']
                auto_target_state = tf.concat([ph[d_]['state'][:, :2*LEA_NJOINTS],
                                               ph[d_]['state'][:, 2*LEA_NJOINTS+2:]], axis=1)
                auto_loss = tf.reduce_mean(tf.square(graph[d_]['inv_agent_state'] - auto_target_state))
                auto_op = tf.train.AdamOptimizer(lr_auto*lr_decay_auto**episodes)
                auto_grads_and_vars = auto_op.compute_gradients(loss=auto_loss, var_list=var_dict[d_]['auto_grad_vars'])
                auto_train_op = auto_op.apply_gradients(grads_and_vars=auto_grads_and_vars)

            else:
                auto_train_op = tf.constant(0)
                auto_loss = tf.constant(0)


            #============== Dynamics model loss ==============
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


            #============== Discriminator loss ===============
            if d_ == 'learner':
                lr_disc = params[d_]['discriminator']['lr']
                lr_decay_disc = params[d_]['discriminator']['lr_decay']

                # Add a entropy regularizer to the discriminator
                logits = tf.concat([graph[d_]['fake_prob'], graph[d_]['real_prob']], 0)
                entropy_loss = -tf.reduce_mean(logit_bernoulli_entropy(logits))

                fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['fake_prob'], labels=tf.zeros_like(graph[d_]['fake_prob']))
                fake_loss = tf.reduce_mean(fake_loss)
                real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=graph[d_]['real_prob'], labels=tf.ones_like(graph[d_]['real_prob']))
                real_loss = tf.reduce_mean(real_loss)
                disc_loss = fake_loss + real_loss + 0.0005*entropy_loss

                disc_op = tf.train.AdamOptimizer(lr_disc*lr_decay_disc**episodes)
                disc_grads_and_vars = disc_op.compute_gradients(loss=disc_loss*ph[d_]['train_disc'], var_list=var_dict[d_]['disc_grad_vars'])
                disc_train_op = disc_op.apply_gradients(grads_and_vars=disc_grads_and_vars)

            else:
                disc_train_op = tf.constant(0)
                disc_loss = tf.constant(0)

            # Aggregate training targets
            targets[d_]['train_rl'] = {'actor_train_op': actor_train_op,
                                       'critic_train_op': critic_train_op,
                                       'disc_train_op': disc_train_op,
                                       'model_train_op': model_train_op,
                                       'model_loss': model_loss,
                                       'auto_train_op': auto_train_op,
                                       'action_loss': action_loss,
                                       'disc_loss': disc_loss,
                                       'gen_loss': gen_loss,
                                       'temp_loss': temporal_loss,
                                       'auto_loss': auto_loss}

            # Aggregate training targets
            targets[d_]['train_gama'] = {'actor_train_op': actor_train_op,
                                         'disc_train_op': disc_train_op,
                                         'gama_loss': actor_loss,
                                         'bc_loss': action_loss,
                                         'gen_loss': gen_loss,
                                         'disc_loss': disc_loss,
                                         'temp_loss': temporal_loss}

            # Aggregate model targets
            targets[d_]['train_model'] = {'model_train_op': model_train_op,
                                          'model_loss': model_loss}

            if d_ == 'expert':
                    targets[d_]['bc'] = {'bc_train_op': bc_train_op,
                                         'bc_loss': bc_loss}

            # Update the target
            targets[d_]['update'] = []
            tau = params['train']['tau']

            assert len(var_dict[d_]['all_slow_actor_vars']) == len(var_dict[d_]['all_actor_vars'])
            assert len(var_dict[d_]['all_slow_critic_vars']) == len(var_dict[d_]['all_critic_vars'])

            for i, slow_target_actor_var in enumerate(var_dict[d_]['all_slow_actor_vars']):
                actor_var = var_dict[d_]['all_actor_vars'][i]
                update_slow_target_actor_op = slow_target_actor_var.assign(tau*actor_var+(1-tau)*slow_target_actor_var)
                targets[d_]['update'].append(update_slow_target_actor_op)


            for i, slow_target_var in enumerate(var_dict[d_]['all_slow_critic_vars']):
                critic_var = var_dict[d_]['all_critic_vars'][i]
                update_slow_target_critic_op = slow_target_var.assign(tau*critic_var+(1-tau)*slow_target_var)
                targets[d_]['update'].append(update_slow_target_critic_op)

            targets[d_]['update'] = tf.group(*targets[d_]['update'], name='update_slow_targets')


        # Episode count increment op
        targets['episode_inc_op'] = episode_inc_op

        return targets
