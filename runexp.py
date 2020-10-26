import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import json, sys, os
from os import path
import time
import shutil
from matplotlib import pyplot as plt
import importlib
import argparse
import time
import sys

# Experiment modules
from environment import create_env
from compgraph import build_compgraph
from replaymemory import create_replay_memory
from utils import *

# os settings
sys.path.append(os.path.dirname(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.set_printoptions(threshold=np.nan)
np.random.seed(0)

#####################################################################################################
# Algorithm

# Deep Deterministic Policy Gradient (DDPG)
# An off-policy actor-critic algorithm that uses additive exploration noise (e.g. an Ornstein-Uhlenbeck process) on top
# of a deterministic policy to generate experiences (s, a, r, s'). It uses minibatches of these experiences from replay
# memory to update the actor (policy) and critic (Q function) parameters.
# Neural networks are used for function approximation.
# Slowly-changing "target" networks are used to improve stability and encourage convergence.
# Parameter updates are made via Adam.
# Assumes continuous action spaces!

#####################################################################################################

# Parse cmdline args
parser = argparse.ArgumentParser(description='CycleGAIL')
parser.add_argument('--logdir', default='./logs', type=str)
parser.add_argument('--exp_id', default='4', type=str)
parser.add_argument('--render', default=False, type=bool)
args = parser.parse_args()

# Clear out the save logs directory (for tensorboard)
if os.path.isdir(args.logdir):
    shutil.rmtree(args.logdir)

# Create the environment
env_params = {'expert': 'cartpole', 'learner': 'cartpole'}
#env_params = {'expert': 'pendulum'}
env = create_env(env_params)

# Print experiment details
print('Booting exp id: {}...'.format(args.exp_id))
time.sleep(2)

# Experiment parameters
mod = importlib.import_module('saved_params.exp'+args.exp_id)
params = mod.generate_params(env=env)

# Replay memory for each domain
replay_memory = create_replay_memory(env=env, params=params)

# Build the computation graph
ph, graph, targets, save_vars = build_compgraph(params=params, env=env)

# Readouts from train loop
expert_readouts = {'total_steps': 0,
                   'total_reward': 0,
                   'steps_in_ep': 0,
                  }
learner_readouts = {'total_steps': 0,
                    'total_reward': 0,
                    'steps_in_ep': 0,
                   }

readouts = {'expert': expert_readouts, 'learner': learner_readouts}
fig_dict = {d_: i for i, d_ in enumerate(env.keys())}
stop_train = {'expert': False, 'learner': False}
num_good_runs = 0

# Train loop
saver = tf.train.Saver(save_vars)
sess = tf.Session()
writer = tf.summary.FileWriter(args.logdir, sess.graph)

# Graph variable initialization
sess.run(tf.global_variables_initializer())

for ep in range(params['train']['num_episodes']):
    for d_ in env.keys():
        # Reset readouts every episode
        for k_ in readouts[d_].keys():
            if k_ != 'total_steps':
                if 'loss' in k_:
                    readouts[d_][k_] = []
                else:
                    readouts[d_][k_] = 0

        # Initialize exploration noise process
        noise_process = np.zeros(env[d_]['action_dim'])
        action_range = env[d_]['env'].action_space.high - env[d_]['env'].action_space.low
        noise_scale = params['train']['initial_noise_scale']*(params['train']['noise_decay']**ep) * action_range


        if not stop_train[d_]:

            # Reset the environment
            obs = env[d_]['env'].reset()

            for t in range(params['train']['max_steps_ep']):

                # choose action based on deterministic policy
                raw_action, = sess.run(graph[d_]['action'], feed_dict={ph[d_]['state']: obs[None], ph[d_]['is_training']: False})

                # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                noise_process = params['train']['exploration_theta']*(params['train']['exploration_mu'] - noise_process) \
                              + params['train']['exploration_sigma']*np.random.randn(env[d_]['action_dim'])
                action_for_state = raw_action + noise_scale*noise_process

                # Take step
                next_obs, reward, done, _info = env[d_]['env'].step(action_for_state)
                readouts[d_]['total_reward'] += reward

                # is next_obs a terminal state?
                # 0.0 if done and not env.env._past_limit() else 1.0))
                trans_tuple = (obs, action_for_state, reward, next_obs, 0.0 if done else 1.0, raw_action)
                replay_memory[d_].add_to_memory(trans_tuple)

                # update network weights to fit a minibatch of experience
                if readouts[d_]['total_steps'] % params['train']['train_every'] == 0 and \
                   replay_memory['learner'].len() >= params['train']['batchsize'] and \
                   replay_memory['expert'].len() >= params['train']['batchsize'] and not stop_train[d_]:

                    #print('got here: {}'.format(d_))

                    #print('learner buffer: {}'.format(replay_memory['learner'].len()))
                    #print('expert buffer: {}'.format(replay_memory['expert'].len()))

                    # grab N (s,a,r,s') tuples from replay memory
                    minibatch = replay_memory[d_].sample_from_memory(batchsize=params['train']['batchsize'])

                    # update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
                    feed_dict = {ph[d_]['state']: np.asarray([elem[0] for elem in minibatch]),
                                 ph[d_]['action']: np.asarray([elem[1] for elem in minibatch]),
                                 ph[d_]['reward']: np.asarray([elem[2] for elem in minibatch]),
                                 ph[d_]['next_state']: np.asarray([elem[3] for elem in minibatch]),
                                 ph[d_]['is_not_terminal']: np.asarray([elem[4] for elem in minibatch]),
                                 ph[d_]['is_training']: True,
                                 ph[d_]['train_disc']: 0}

                    # If learner is cloning expert actions
                    if params['learner']['use_bc'] and d_ == 'learner':
                        feed_dict.update({ph['expert']['state']: np.asarray([elem[0] for elem in minibatch])})
                        feed_dict.update({ph['expert']['is_training']: True})

                    if d_ == 'learner':
                        minibatch_exp = replay_memory['expert'].sample_from_memory(batchsize=params['train']['batchsize'])
                        feed_dict.update({ph['expert']['state']: np.asarray([elem[0] for elem in minibatch_exp])})
                        feed_dict.update({ph['expert']['action']: np.asarray([elem[1] for elem in minibatch_exp])})
                        feed_dict.update({ph['expert']['next_state']: np.asarray([elem[3] for elem in minibatch_exp])})
                        feed_dict.update({ph['expert']['raw_action']: np.asarray([elem[5] for elem in minibatch_exp])})

                        if ep % 3 == 0:
                            feed_dict.update({ph[d_]['train_disc']: 1.})

                    # Train operation
                    # Train the dynamics model only for the first few episodes
                    '''
                    if (ep < 100 and d_ == 'learner') or (d_ == 'expert'):
                        fetches = sess.run({'model_train_op': targets[d_]['train']['model_train_op'],
                                            'model_loss': targets[d_]['train']['model_loss']}, feed_dict=feed_dict)
                    else:
                    '''
                    fetches = sess.run(targets[d_]['train'], feed_dict=feed_dict)

                    # For wgan, train the discriminator more
                    if d_ == 'learner' and params['train']['use_wgan']:
                        for train_num in range(20):
                            learner_batch = replay_memory['learner'].sample_from_memory(batchsize=params['train']['batchsize'])
                            expert_batch = replay_memory['expert'].sample_from_memory(batchsize=params['train']['batchsize'])

                            feed_dict = {ph['learner']['state']: np.asarray([elem[0] for elem in learner_batch]),
                                         ph['learner']['is_training']: True,
                                         ph['learner']['train_disc']: 1,
                                         ph['expert']['state']: np.asarray([elem[0] for elem in expert_batch]),
                                         ph['expert']['raw_action']: np.asarray([elem[5] for elem in expert_batch])}

                            sess.run(targets['learner']['train']['disc_train_op'], feed_dict=feed_dict)

                    # Loss readouts
                    for k_, v_ in fetches.items():
                        if 'loss' in k_:
                            readouts[d_][k_] = readouts[d_].get(k_, []) + [fetches[k_]]


                    # Update slow actor and critic targets towards current actor and critic
                    sess.run(targets[d_]['update'])

                obs = next_obs
                readouts[d_]['total_steps'] += 1
                readouts[d_]['steps_in_ep'] += 1

                if done:
                    break

    # Stop updating expert parameters after it achieves good performance
    if readouts['expert']['total_reward'] > 180:
        num_good_runs += 1
        if num_good_runs > 6:
            #saver.save(sess, savedir)
            stop_train['expert'] = True
            stop_train['learner'] = False
    else:
        num_good_runs = 0


    '''
    if ep == 100:
        stop_train['expert'] = True
        render_policy(sess, graph, ph, env, 'expert')
    '''

    # Evaluate the learned state mapping
    if ep % 100 == 0 and ep != 0:
        render_policy(sess, graph, ph, env, 'expert')
        done = False
        init_flag = True
        obs = env['learner']['env'].reset()
        env['expert']['env'].reset()
        while not done:
            mapped_state_raw, trans_act = sess.run([graph['learner']['mapped_state'], graph['learner']['premap_action']],
                                                    feed_dict={ph['learner']['state']: obs[None]})
            mapped_state_raw = mapped_state_raw[0]

            trans_future_states = sess.run(graph['learner']['multi_trans_next_state'],
                                           feed_dict={ph['learner']['state']: obs[None]})

            mapped_future_states = sess.run(graph['learner']['multi_mapped_next_state'],
                                           feed_dict={ph['learner']['state']: obs[None]})

            multi_next_states = sess.run(graph['learner']['multi_next_state'],
                                           feed_dict={ph['learner']['state']: obs[None]})

            mapped_state = env['expert']['env'].env._obs_to_state(mapped_state_raw)

            env['expert']['env'].env.state = mapped_state
            #env['expert']['env'].env.env.state = mapped_state

            print("-------------------------------------------")
            #print("gt: {}".format(env['learner']['env'].env.state))
            #print("predicted: {}".format(mapped_state))

            print("state: {}".format(obs))
            print("mapped_state: {}".format(mapped_state_raw))

            # Render
            env['learner']['env'].render()
            if init_flag:
                time.sleep(3)
                init_flag = False
            env['expert']['env'].render()

            # Choose action
            act, = sess.run(graph['learner']['action'], feed_dict={ph['learner']['state']: obs[None],
                                                                  ph['learner']['is_training']: False})

            print("chosen_action: {}".format(act))
            print("trans_action: {}".format(trans_act))
            print("mapped_future_states: {}".format(mapped_future_states))
            print("trans_future_states: {}".format(trans_future_states))
            print("model_next_state: {}".format(multi_next_states))


            # Step
            next_obs, reward, done, info = env['learner']['env'].step(act)

            #time.sleep(1)

            obs = next_obs


    # When done executing one episode in all domains
    #sess.run(targets['episode_inc_op'])
    #sess.run(targets['learner']['train']['statemap_weights'])

    #print(sess.run(targets['learner']['train']['statemap_weights']))
    #print()
    #print("Learner reward: {}".format(readouts['learner']['total_reward']))


    # Print out episode evaluation metrics
    print("Episode: {}".format(ep))
    print_metrics('expert', readouts['expert'])
    print_metrics('learner', readouts['learner'])
    print('______________________________')

    '''
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Episode Reward/{}'.format(d_), simple_value=total_reward[d_])]), global_step=ep)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Identity Loss/{}'.format(d_), simple_value=np.mean(identity_loss[d_]))]), global_step=ep)
    writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='Statemap Loss/{}'.format(d_), simple_value=np.mean(smap_loss[d_]))]), global_step=ep)
    '''

# Close environments
sess.close()
writer.close()
for d_ in env.keys():
    env[d_]['env'].close()


