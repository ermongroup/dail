import numpy as np
import tensorflow as tf
import time
from time import sleep
import signal
import pdb
import matplotlib
from matplotlib import pyplot as plt
import pylab as pl
import imageio
import random
from tqdm import tqdm

from threading import Lock
import shelve

from dail.compgraph import build_compgraph
from dail.utils import *

class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
            self.kill_now = False
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
            self.kill_now = True



class DDPGAgent():

    def __init__(self,
                 params,
                 cmd_args,
                 env,
                 replay_memory,
                 save_expert_dir,
                 save_learner_dir,
                 save_dataset_dir,
                 load_expert_dir,
                 load_learner_dir,
                 load_dataset_dir,
                 logdir,
                 render,
                 gpu,
                 is_transfer):

            # Agent attributes
            self.params = params
            self.args = cmd_args
            self.env = env
            self.replay_memory = replay_memory

            # Graph attributes
            self.render = render
            self.save_expert_dir = save_expert_dir
            self.save_learner_dir = save_learner_dir
            self.save_dataset_dir = save_dataset_dir
            self.load_expert_dir = load_expert_dir
            self.load_learner_dir = load_learner_dir
            self.load_dataset_dir = load_dataset_dir
            self.logdir = logdir
            self.render = render
            self.gpu = gpu

            # Build computation graph for the DDPG agent
            self.ph, self.graph, self.targets, \
            self.expert_save_vars, \
            self.learner_save_vars = build_compgraph(params=self.params,
                                                     env=self.env,
                                                     algo='ddpg',
                                                     is_transfer=is_transfer)

            # Session and saver
            if self.gpu > -1:
                gpu_options = tf.GPUOptions(allow_growth=True)
                gpu_config = tf.ConfigProto(gpu_options=gpu_options,
                                            log_device_placement=True)
                self.sess = tf.Session(config=gpu_config)
            else:
                self.sess = tf.Session()

            self.saver_expert = tf.train.Saver(var_list=self.expert_save_vars)
            self.saver_learner = tf.train.Saver(var_list=self.learner_save_vars)


            self.expert_basis = np.array([[1.0, 0.0]])
            self.novice_basis = np.array([[0.0, 1.0]])
            self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
            self.killer = GracefulKiller()



    def act(self):
            raise NotImplementedError



    def invoke_killer(self, render_option):
        epsilon = {'expert': 1., 'learner': 1.}

        # Manually save checkpoint or kill training
        if self.killer.kill_now:
            visual_option = input('Visualize (y/[n])?')
            if visual_option == 'y':
                if render_option == 'expert_policy':
                    render_policy(self.sess,
                                  self.graph,
                                  self.ph,
                                  self.env,
                                  'expert',
                                  num_rollout=1,
                                  save_video=True,
                                  save_dir='expert_policy')

                elif render_option == 'expert_dynamics':
                    self.render_dynamics({'epsilon': epsilon},
                                         domain='expert',
                                         num_rollout=3,
                                         save_dir='expert_dynamics')

                elif render_option == 'learner_policy':
                    self.render_statemap({'epsilon': epsilon},
                                         num_rollout=10,
                                         save_dir='learner_policy')

                elif render_option == 'learner_dynamics':
                    self.render_dynamics({'epsilon': epsilon},
                                         domain='learner',
                                         num_rollout=10,
                                         save_dir='learner_dynamics')
                else:
                    print("ERROR: unrecognized render option '{}'".format(render_option))

            save_option = input('Save current model (y/[n])?')
            if save_option == 'y':
                if 'expert' in render_option:
                    self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')
                elif 'learner' in render_option:
                    self.saver_learner.save(self.sess, self.save_learner_dir+'/learner.ckpt')
                else:
                    print("ERROR: unrecognized render option '{}'".format(render_option))

            kill_option = input('Kill session (y/[n])?')
            if kill_option == 'y':
                self.sess.close()
                self.writer.close()
                for d_ in self.env.keys():
                    self.env[d_]['env'].close()

                exit(1)
            else:
                self.killer.kill_now = False



    def train_expert(self, from_ckpt):
        # Readouts from train loop
        expert_readouts = {'total_steps': 0,
                           'total_reward': 0,
                           'steps_in_ep': 0}

        learner_readouts = {'total_steps': 0,
                            'total_reward': 0,
                            'steps_in_ep': 0}

        readouts = {'expert': expert_readouts, 'learner': learner_readouts}
        stop_train = {'expert': False, 'learner': False}
        num_good_runs = 0
        reward_history = []
        best_reward = -np.inf

        # Train loop
        self.sess.run(tf.global_variables_initializer())

        if from_ckpt:
            self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')

        for ep in range(self.params['train']['num_episodes']):
            d_ = 'expert'

            # Reset readouts every episode
            for k_ in readouts[d_].keys():
                    if k_ != 'total_steps':
                            if 'loss' in k_:
                                    readouts[d_][k_] = []
                            else:
                                    readouts[d_][k_] = 0

            # Initialize exploration noise process
            noise_process = np.zeros(self.env[d_]['action_dim'])
            action_range = self.env[d_]['env'].action_space.high \
                            - self.env[d_]['env'].action_space.low
            noise_scale = self.params['train']['initial_noise_scale'] \
                            *(self.params['train']['noise_decay']**ep) * action_range

            # Reset environment
            obs = self.env[d_]['env'].reset()

            for t in range(self.params['train']['max_steps_ep']):

                # choose action based on deterministic policy
                raw_action, = self.sess.run(self.graph[d_]['action'],
                                            feed_dict={self.ph[d_]['state']: obs[None],
                                                       self.ph[d_]['is_training']: False})

                # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                noise_process = self.params['train']['exploration_theta'] \
                                * (self.params['train']['exploration_mu'] - noise_process) \
                                + self.params['train']['exploration_sigma'] * np.random.randn(self.env[d_]['action_dim'])
                action_for_state = raw_action + noise_scale*noise_process

                # Take step
                next_obs, reward, done, _info = self.env[d_]['env'].step(action_for_state)
                readouts[d_]['total_reward'] += reward

                # is next_obs a terminal state?
                # 0.0 if done and not env.env._past_limit() else 1.0))
                trans_tuple = (obs, action_for_state, reward, next_obs, 0.0 if done else 1.0, raw_action)
                self.replay_memory[d_].add_to_memory(trans_tuple)

                # update network weights to fit a minibatch of experience
                if readouts[d_]['total_steps'] % self.params['train']['train_every'] == 0 and \
                   self.replay_memory['expert'].len() >= self.params['train']['batchsize'] and \
                   not stop_train[d_]:

                    # grab N (s,a,r,s') tuples from replay memory
                    minibatch = self.replay_memory[d_].sample_from_memory(batchsize=self.params['train']['batchsize'])

                    # update the critic and actor params using mean-square value error
                    # and deterministic policy gradient, respectively
                    feed_dict = {self.ph[d_]['state']: np.asarray([elem[0] for elem in minibatch]),
                                 self.ph[d_]['action']: np.asarray([elem[1] for elem in minibatch]),
                                 self.ph[d_]['reward']: np.asarray([elem[2] for elem in minibatch]),
                                 self.ph[d_]['next_state']: np.asarray([elem[3] for elem in minibatch]),
                                 self.ph[d_]['is_not_terminal']: np.asarray([elem[4] for elem in minibatch]),
                                 self.ph[d_]['is_training']: True,
                                 self.ph[d_]['train_disc']: 0}


                    fetches = self.sess.run(self.targets[d_]['train_rl'], feed_dict=feed_dict)

                    # Loss readouts
                    for k_, v_ in fetches.items():
                        if 'loss' in k_:
                            readouts[d_][k_] = readouts[d_].get(k_, []) + [fetches[k_]]


                    # Update slow actor and critic targets towards current actor and critic
                    self.sess.run(self.targets[d_]['update'])

                obs = next_obs
                readouts[d_]['total_steps'] += 1
                readouts[d_]['steps_in_ep'] += 1

                if done:
                    break

            # Manually save checkpoint or kill training
            self.invoke_killer(render_option='expert_policy')


            # save the best model
            reward_history.append(readouts['expert']['total_reward'])
            mean_reward = np.mean(reward_history[-10:])
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')


            # Print out episode evaluation metrics
            print("Episode: {}".format(ep))
            print_metrics('expert', readouts['expert'])
            print("Best reward: {:.2f}".format(best_reward))
            print('______________________________')


            # Save to for tensorboard readout
            save_reward = tf.Summary.Value(tag='Reward', simple_value=readouts['expert']['total_reward'])
            self.writer.add_summary(tf.Summary(value=[save_reward]), readouts['expert']['total_steps'])



    def create_alignment_taskset(self):
            # Readouts from train loop
            expert_readouts = {'total_steps': 0,
                               'total_reward': 0,
                               'steps_in_ep': 0}

            learner_readouts = {'total_steps': 0,
                                'total_reward': 0,
                                'steps_in_ep': 0}

            readouts = {'expert': expert_readouts, 'learner': learner_readouts}
            stop_train = {'expert': True, 'learner': False}
            num_good_runs = 0

            # Graph variable initialization
            self.sess.run(tf.global_variables_initializer())
            self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')


            # Don't restore the inverse map when transfering
            self.learner_save_vars = [var for var in self.learner_save_vars if 'invmap' not in var.name]
            self.saver_learner = tf.train.Saver(var_list=self.learner_save_vars)
            self.saver_learner.restore(self.sess, self.load_learner_dir+'/learner.ckpt')

            # Copy the expert policy into new variable
            expert_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope='actor/expert')
            learner_actor_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      scope='actor/learner/expert_pi')
            learner_slow_actor_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                           scope='slow_target_actor/learner/expert_pi')

            copy_expert_pi_op = []
            for var_idx, var in enumerate(learner_slow_actor_pi_vars):
                copy_expert_pi_op.append(var.assign(expert_pi_vars[var_idx]))

            for var_idx, var in enumerate(learner_actor_pi_vars):
                copy_expert_pi_op.append(var.assign(expert_pi_vars[var_idx]))

            self.sess.run(copy_expert_pi_op)

            print("Creating alignment task set")
            create_hybrid_dataset(self.sess,
                                  self.graph,
                                  self.ph,
                                  self.env,
                                  self.save_dataset_dir,
                                  num_transitions=int(1e5),
                                  save_video=False)

            # Close environments
            self.sess.close()
            self.writer.close()
            for d_ in self.env.keys():
                self.env[d_]['env'].close()

            return



    def gama(self, from_ckpt):

        # Readouts from train loop
        expert_readouts = {'total_steps': 0,
                           'total_reward': 0,
                           'steps_in_ep': 0}

        learner_readouts = {'total_steps': 0,
                            'total_reward': 0,
                            'steps_in_ep': 0}

        readouts = {'expert': expert_readouts, 'learner': learner_readouts}

        # Graph variable initialization
        self.sess.run(tf.global_variables_initializer())
        self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')


        # Training learner from ckpt
        if from_ckpt:
            self.saver_learner.restore(self.sess, self.load_learner_dir+'/learner.ckpt')
            self.params['train']['initial_noise_scale'] = 0.


        # Populate the replay memory with data (TODO)
        mutex = Lock()
        mutex.acquire()
        hybrid_dataset = shelve.open(self.load_dataset_dir, writeback=True)
        self.replay_memory['expert'].set_memory(hybrid_dataset['expert'])
        self.replay_memory['learner'].set_memory(hybrid_dataset['learner'])
        hybrid_dataset.close()
        mutex.release()

        print("Loading data from: {}".format(self.load_dataset_dir))
        print("Num expert transitions: {}".format(self.replay_memory['expert'].len()))
        print("Num learner transitions: {}".format(self.replay_memory['learner'].len()))


        epsilon = {'expert': 1., 'learner': 1.}
        eps_decay_rate = self.params['train']['eps_decay_rate']
        min_eps = self.params['train']['min_eps']
        d_ = 'learner'

        # Train the expert policy and dynamics model
#                self.render_dynamics({'epsilon': epsilon}, domain='expert', num_rollout=5, save_dir='expert_dynamics')


        #=========== Train the learner dynamics model ===========
        step = 0
        dynamics_loss = []
#        for idx in range(10):
        for idx in range(100000):

            # grab N (s,a,r,s') tuples from replay memory
            minibatch_l = self.replay_memory['learner'].sample_from_memory(batchsize=self.params['train']['batchsize'])

            # update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
            feed_dict = {self.ph[d_]['state']: np.asarray([elem[0] for elem in minibatch_l]),
                         self.ph[d_]['action']: np.asarray([elem[1] for elem in minibatch_l]),
                         self.ph[d_]['next_state']: np.asarray([elem[3] for elem in minibatch_l]),
                         self.ph[d_]['is_not_terminal']: np.asarray([elem[4] for elem in minibatch_l]),
                         self.ph[d_]['is_training']: True,
                         self.ph[d_]['train_disc']: 0}


            # Train
            fetches = self.sess.run(self.targets[d_]['train_model'], feed_dict=feed_dict)

            # When done executing one episode in all domains
            dynamics_loss.append(fetches['model_loss'])
            step += 1


            # Print metrics
            if idx % 100 == 0:
                print("Step {} | Learner dynamics loss: {:.2e}".format(idx, np.mean(dynamics_loss)), end="\r")
                dynamics_loss = []

            self.invoke_killer(render_option='learner_dynamics')


#                self.render_dynamics({'epsilon': epsilon}, domain='learner', num_rollout=5, save_dir='learner_dynamics')


        #=========== Train MDP alignment ============
        step = 0
        best_reward = -np.inf
        best_gama_loss = np.inf

        while True:

            for idx in tqdm(range(500)):
                # grab N (s,a,r,s') tuples from replay memory
                minibatch_l = self.replay_memory['learner'].sample_from_memory(batchsize=self.params['train']['batchsize'])

                # update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
                feed_dict = {self.ph[d_]['state']: np.asarray([elem[0] for elem in minibatch_l]),
                             self.ph[d_]['action']: np.asarray([elem[1] for elem in minibatch_l]),
                             self.ph[d_]['raw_action']: np.asarray([elem[5] for elem in minibatch_l]),
                             self.ph[d_]['reward']: np.asarray([elem[2] for elem in minibatch_l]),
                             self.ph[d_]['next_state']: np.asarray([elem[3] for elem in minibatch_l]),
                             self.ph[d_]['is_not_terminal']: np.asarray([elem[4] for elem in minibatch_l]),
                             self.ph[d_]['is_training']: True,
                             self.ph[d_]['train_disc']: 0.,
                             self.ph[d_]['disc_reward']: np.asarray([elem[8] for elem in minibatch_l])}



                if step % 1 == 0:
                    minibatch_e = self.replay_memory['expert'].sample_from_memory(batchsize=self.params['train']['batchsize'])
                    feed_dict_expert = {self.ph['expert']['state']: np.asarray([elem[0] for elem in minibatch_e]),
                                        self.ph['expert']['action']: np.asarray([elem[1] for elem in minibatch_e]),
                                        self.ph['expert']['next_state']: np.asarray([elem[3] for elem in minibatch_e]),
                                        self.ph['expert']['raw_action']: np.asarray([elem[5] for elem in minibatch_e]),
                                        self.ph['expert']['is_training']: True}

                    feed_dict.update({self.ph[d_]['train_disc']: 1.})
                    feed_dict.update(feed_dict_expert)
                    targets = self.targets[d_]['train_gama']

                else:
                    targets = {k: v for k, v in self.targets[d_]['train_gama'].items() if 'disc' not in k}


                # Train
                fetches = self.sess.run(targets, feed_dict=feed_dict)

                # When done executing one episode in all domains
                self.sess.run(self.targets['episode_inc_op'])
                step += 1

                # Loss readouts
                for k_, v_ in fetches.items():
                        if 'loss' in k_:
                                readouts[d_][k_] = readouts[d_].get(k_, []) + [fetches[k_]]

                readouts[d_]['total_steps'] += 1

                # Killer
                self.invoke_killer(render_option='learner_policy')


            # Evaluate episode reward of learner (TODO)
            readouts[d_]['total_reward'] = self.render_statemap({'epsilon': epsilon},
                                                                num_rollout=25,
                                                                save_video=False,
                                                                save_dir='learner_policy')

            # Monitor best true env reward
            if readouts[d_]['total_reward'] > best_reward:
                best_reward = readouts[d_]['total_reward']

            # Save model that achieves best GAMA loss
            epoch_gama_loss = np.mean(readouts[d_]['gama_loss'][-50:])
            if epoch_gama_loss < best_gama_loss:
                best_gama_loss = epoch_gama_loss
                best_gama_reward = readouts[d_]['total_reward']
                self.saver_learner.save(self.sess, self.save_learner_dir+'/learner.ckpt')

            # Print out metrics
            print("Step: {}".format(step))
            print_metrics('learner', readouts['learner'])
            print("Best reward: {:.2f}".format(best_reward))
            print("Best gama reward: {:.2f}".format(best_gama_reward))
            print("Best gama loss: {:.6f}".format(best_gama_loss))
            print('______________________________')


            # Save to tensorboard
            save_reward = tf.Summary.Value(tag='Best Reward', simple_value=readouts['learner']['total_reward'])
            save_gama_reward = tf.Summary.Value(tag='Best GAMA Reward', simple_value=best_gama_reward)
            gama_loss = tf.Summary.Value(tag='GAMA Loss', simple_value=np.mean(readouts['learner']['gama_loss']))
            gen_loss = tf.Summary.Value(tag='Generator Loss', simple_value=np.mean(readouts['learner']['gen_loss']))
            self.writer.add_summary(tf.Summary(value=[save_reward]), step)
            self.writer.add_summary(tf.Summary(value=[save_gama_reward]), step)
            self.writer.add_summary(tf.Summary(value=[gama_loss]), step)
            self.writer.add_summary(tf.Summary(value=[gen_loss]), step)

            # Reset readouts
            for k_ in readouts[d_].keys():
                if k_ != 'total_steps':
                    if 'loss' in k_:
                        readouts[d_][k_] = []
                    else:
                        readouts[d_][k_] = 0



    def zeroshot(self):
        # Readouts from train loop
        expert_readouts = {'total_steps': 0,
                           'total_reward': 0,
                           'steps_in_ep': 0}

        learner_readouts = {'total_steps': 0,
                            'total_reward': 0,
                            'steps_in_ep': 0}

        readouts = {'expert': expert_readouts, 'learner': learner_readouts}
        stop_train = {'expert': True, 'learner': False}
        num_good_runs = 0

        # Graph variable initialization
        self.sess.run(tf.global_variables_initializer())
        self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')

        # Don't restore the inverse map when transfering
        self.learner_save_vars = [var for var in self.learner_save_vars if 'invmap' not in var.name]
        self.saver_learner = tf.train.Saver(var_list=self.learner_save_vars)
        self.saver_learner.restore(self.sess, self.load_learner_dir+'/learner.ckpt')

        # Copy the expert policy into new variable
        expert_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='actor/expert')
        learner_actor_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                  scope='actor/learner/expert_pi')
        learner_slow_actor_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                       scope='slow_target_actor/learner/expert_pi')

        copy_expert_pi_op = []
        for var_idx, var in enumerate(learner_slow_actor_pi_vars):
            copy_expert_pi_op.append(var.assign(expert_pi_vars[var_idx]))

        for var_idx, var in enumerate(learner_actor_pi_vars):
            copy_expert_pi_op.append(var.assign(expert_pi_vars[var_idx]))

        self.sess.run(copy_expert_pi_op)


        # Train the learner
        epsilon = {'expert': 1., 'learner': 1.}
        eps_decay_rate = self.params['train']['eps_decay_rate']
        min_eps = self.params['train']['min_eps']


        # Visualize the statemap
        print("Zeroshot performance...")
        vid_name = self.args.doc + '_{}'.format(self.load_learner_dir[-1]) \
                                                if self.args.doc \
                                                else self.load_learner_dir[-6:]
        self.render_statemap({'epsilon': epsilon}, num_rollout=5, save_dir=vid_name, save_video=True)
        print("Done!")

        # Close environments
        self.sess.close()
        self.writer.close()
        for d_ in self.env.keys():
            self.env[d_]['env'].close()

        return



    def bc(self, from_ckpt=False, num_demo=-1):
            self.sess.run(tf.global_variables_initializer())
            if from_ckpt:
                self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')

            # Load dataset and randomly sample [num_demo] demonstrations
            print("Loading dataset")
            self.traj_data = np.load(self.load_dataset_dir+'.npz', allow_pickle=True)
            num_total_demo = self.traj_data['obs'].shape[0]
            assert self.traj_data['acs'].shape[0] == num_total_demo

            num_demo = num_total_demo if num_demo == -1 else num_demo
            rand_indices = np.random.choice(num_total_demo, num_demo, replace=False)
            exp_o = np.concatenate(self.traj_data['obs'][rand_indices].tolist(), axis=0)
            exp_a = np.concatenate(self.traj_data['acs'][rand_indices].tolist(), axis=0)
            print("state data shape: {}".format(exp_o.shape))
            print("action data shape: {}".format(exp_a.shape))

            num_sa = exp_o.shape[0]
            num_epochs = self.params['bc']['num_epochs']
            batches_per_epoch = self.params['bc']['batches_per_epoch']
            batch = self.params['bc']['batchsize']
            best_reward = -np.inf

            for epoch in range(num_epochs):
                epoch_loss = []
                for i in tqdm(range(batches_per_epoch)):
                    batch_idx = np.random.choice(num_sa, batch, replace=False)
                    batch_o = exp_o[batch_idx]
                    batch_a = exp_a[batch_idx]
                    fetches = self.sess.run(self.targets['expert']['bc'],
                                            feed_dict={self.ph['expert']['state']: batch_o,
                                                       self.ph['expert']['action_tv']: batch_a})
                    epoch_loss.append(fetches['bc_loss'])

                # SL loss
                epoch_loss = np.mean(epoch_loss)
                save_bc_loss = tf.Summary.Value(tag='BC loss', simple_value=epoch_loss)
                self.writer.add_summary(tf.Summary(value=[save_bc_loss]), epoch)

                # Evaluate RL performance after each epoch
                avg_reward = render_policy(self.sess, self.graph, self.ph, self.env, 'expert', num_rollout=50, save_video=False)
                save_reward = tf.Summary.Value(tag='Reward', simple_value=avg_reward)
                self.writer.add_summary(tf.Summary(value=[save_reward]), epoch)
                if self.killer.kill_now:
                    visual_option = input('Visualize (y/[n])?')
                    if visual_option == 'y':
                        render_policy(self.sess, self.graph, self.ph, self.env, 'expert', num_rollout=20, save_video=True)

                    save_option = input('Save current model (y/[n])?')
                    if save_option == 'y':
                        self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')

                    kill_option = input('Kill session (y/[n])?')
                    if kill_option == 'y':
                        self.sess.close()
                        self.writer.close()
                        for d_ in self.env.keys():
                                self.env[d_]['env'].close()

                        break
                    else:
                        self.killer.kill_now = False
                if self.killer.kill_now:
                    visual_option = input('Visualize (y/[n])?')
                    if visual_option == 'y':
                        render_policy(self.sess, self.graph, self.ph, self.env, 'expert', num_rollout=20, save_video=True)

                    save_option = input('Save current model (y/[n])?')
                    if save_option == 'y':
                        self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')

                    kill_option = input('Kill session (y/[n])?')
                    if kill_option == 'y':
                        self.sess.close()
                        self.writer.close()
                        for d_ in self.env.keys():
                                self.env[d_]['env'].close()

                        break
                    else:
                        self.killer.kill_now = False

                print("Epoch {} | bc loss: {}, rl loss: {}".format(epoch, epoch_loss, avg_reward))

                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')

                # Manually save checkpoint or kill training
                self.invoke_killer(render_option='expert_policy')

            # Save the bc expert after all epochs
            self.saver_expert.save(self.sess, self.save_expert_dir+'/expert.ckpt')



    def create_demonstrations(self, num_demo):
        self.sess.run(tf.global_variables_initializer())
        self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')

        # Create dataset
        vid_name = self.args.doc if self.args.doc else 'target_expert_demo'
        create_dataset(self.sess,
                       self.graph,
                       self.ph,
                       self.env,
                       save_dir=self.save_dataset_dir,
                       num_rollout=num_demo,
                       save_video=False,
                       vid_name=vid_name)

        # Close environments
        self.sess.close()
        self.writer.close()
        for d_ in self.env.keys():
            self.env[d_]['env'].close()

        return



    def rollout_expert(self):
        # Graph variable initialization
        self.sess.run(tf.global_variables_initializer())
        self.saver_expert.restore(self.sess, self.load_expert_dir+'/expert.ckpt')

        # Visualize loaded expert policy
        vid_name = self.args.doc + '_{}'.format(self.load_expert_dir[-1]) if self.args.doc else self.load_expert_dir[-6:]
        render_policy(self.sess,
                      self.graph,
                      self.ph,
                      self.env,
                      'expert',
                      num_rollout=50,
                      save_video=False,
                      save_dir=vid_name)

        # Close environments
        self.sess.close()
        self.writer.close()
        for d_ in self.env.keys():
            self.env[d_]['env'].close()
        return



    def render_statemap(self, loop_vars, save_video=True, num_rollout=20, save_dir='learner_policy'):


        # Render the expert policy
        #render_policy(self.sess, self.graph, self.ph, self.env, 'expert')
        frames = []
        tot_reward = []


        for idx in range(num_rollout):
            done = False
            obs = self.env['learner']['env'].reset()
            self.env['expert']['env'].reset()
            ep_reward = 0
            while not done:
                mapped_state_raw, \
                trans_act, \
                act = self.sess.run([self.graph['learner']['mapped_state'],
                                     self.graph['learner']['premap_action'],
                                     self.graph['learner']['action']],
                                     feed_dict={self.ph['learner']['state']: obs[None],
                                                self.ph['learner']['is_training']: False})

                act = act[0]
                mapped_state_raw = mapped_state_raw[0]

                self.env['expert']['env'].env.set_state_from_obs(mapped_state_raw)
#                self.env['expert']['env'].set_state_from_obs(mapped_state_raw)

                # Render
                if save_video:
                    # Concatenate learner and expert images
                    limg = self.env['learner']['env'].render(mode='rgb_array')
                    eimg = self.env['expert']['env'].render(mode='rgb_array')
                    img = np.concatenate([limg, eimg], axis=1)
                    frames.append(img)

                # Step
                next_obs, reward, done, info = self.env['learner']['env'].step(act)
#                                print(np.around(trans_act, 4))

                obs = next_obs
                ep_reward += reward

            tot_reward.append(ep_reward)

        # total reward
        avg_reward = np.mean(tot_reward)
        print("Average reward: {}".format(avg_reward))

        if save_video:
            # Save the frames into video
            save_frames_as_video(frames, save_dir)

        return avg_reward



    def render_dynamics(self, loop_vars, domain, save_video=True, num_rollout=20, save_dir='dynamics'):

        # Render the expert policy
        frames = []
        d_ = domain

        # Reacher specific
        if 'reacher2' in self.env[d_]['name']:
            njoints = 2
        elif 'reacher3' in self.env[d_]['name']:
            njoints = 3
        else:
            print('[ddpg.py] ERROR: unrecognized env name {}'.format(self.env[d_]['name']))
            exit(1)


        for idx in range(num_rollout):
            obs = self.env[d_]['env'].reset()
            ep_step = 0
            while ep_step < 60:
                act = self.sess.run(self.graph[d_]['action'],
                                    feed_dict={self.ph[d_]['state']: obs[None],
                                               self.ph[d_]['is_training']: False})

                act = act[0]


                # Render
                self.env[d_]['env'].env.set_state_from_obs(obs)
                if save_video:
                    img = self.env[d_]['env'].render(mode='rgb_array')
                    frames.append(img)

                # Step with dynamics model
                next_obs = self.sess.run(self.graph[d_]['model_next_state'],
                                         feed_dict={self.ph[d_]['state']: obs[None],
                                                    self.ph[d_]['action']: act[None],
                                                    self.ph[d_]['is_training']: False})


                obs = np.concatenate([next_obs[0, :njoints*2],
                                      obs[2*njoints:2*njoints+2],
                                      next_obs[0, 2*njoints:]])
                ep_step += 1



        if save_video:
            # Save the frames into video
            save_frames_as_video(frames, save_dir)











