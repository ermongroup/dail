import numpy as np
import imageio
from pdb import set_trace
import os
import shelve
from collections import deque

# noinspection PyTypeChecker
def open_file_and_save(file_path, data):
    """
    :param file_path: type==string
    :param data:
    """
    if os.path.exists(file_path):
        os.remove(file_path)

    # Create the directory for the file
    dir_path = file_path.rsplit('/', 1)[0]
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(file_path, 'wb') as f_handle:
        np.savetxt(f_handle, data, fmt='%s')

def save_frames_as_video(frames, filename='temp'):
    """
    Displays a list of frames as a gif, with controls
    """
    # down sample frames to appropriate size
    if frames[0].shape[0] > 500:
        print("downsampling")
        for i, frame in enumerate(frames):
            frames[i] = frame[::3, ::3, :]
    frames = np.asarray(frames)
    imageio.mimwrite('./cached_videos/'+filename+'.mp4', frames, fps=60)


def print_metrics(domain, readouts):
    '''
    Printing the losses from a sess.run() call
    Args:
        readouts: losses and train_ops : dict
        domain: domain : str
    Returns:
    '''
    spacing = 17
    print_str = '\n' + domain.capitalize() + '> \n'
    print_str += 'reward: '.rjust(spacing) + '{:0.1f}'.format(readouts['total_reward']) + '\n'
    if 'total_fake_reward' in readouts:
        print_str += 'fake reward: '.rjust(spacing) + '{:0.1f}'.format(readouts['total_fake_reward']) + '\n'
    for k_, v_ in readouts.items():
        if 'loss' in k_:
            value = np.around(np.mean(v_, axis=0), decimals=6)
            print_str += (k_ + ': ').rjust(spacing) + str(value) + '\n'

    print_str = print_str[:-2]
    print(print_str)



def render_policy(sess, graph, ph, env, domain, num_rollout=20, save_video=True, save_dir='temp'):

    frames = []
    tot_reward = []

    if save_video:
        print("Saving video")
    else:
        print("Evaluating expert performance")

    for idx in range(num_rollout):
        done = False
        obs = env[domain]['env'].reset()
        steps = 0
        ep_reward = 0.

        while not done:
            if save_video:
                frames.append(env[domain]['env'].render(mode='rgb_array'))

            action, = sess.run(graph[domain]['action'], feed_dict={ph[domain]['state']: obs[None],
                                                                   ph[domain]['is_training']: False})
            obs, reward, done, info = env[domain]['env'].step(action)

#            print(np.around(action, 4))

            ep_reward += reward
            steps += 1

        tot_reward.append(ep_reward)

    if save_video:
        save_frames_as_video(frames, save_dir)


    avg_reward = np.mean(tot_reward)
    print("Steps: {}".format(steps))
    print("Avg Reward: {}".format(avg_reward))

    return avg_reward


def create_dataset(sess, graph, ph, env, save_dir, num_rollout=20, save_video=True, vid_name='demonstrations'):

    frames = []
    tot_reward = []
    total_obs = []
    total_acs = []

    if save_video:
        print("Saving video")
    else:
        print("Creating transfer dataset")

    for idx in range(num_rollout):
        done = False
        obs = env['expert']['env'].reset()

        steps = 0
        ep_reward = 0.
        ep_obs = []
        ep_acs = []

        while not done:
            # Get next action
            action, = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
                                                                     ph['expert']['is_training']: False})
            ep_obs.append(np.squeeze(obs))
            ep_acs.append(np.squeeze(action))

            # Save dataset as video
            if save_video:
                eimg = env['expert']['env'].render(mode='rgb_array')
                frames.append(eimg)

            # Step in environment
            ## Add slight noise to the action space
            action += np.random.normal(0, 0.05)
            obs, reward, done, info = env['expert']['env'].step(action)

            ep_reward += reward
            steps += 1

        tot_reward.append(ep_reward)
        total_obs.append(np.array(ep_obs))
        total_acs.append(np.array(ep_acs))

    # Print metrics
    print("Steps: {}".format(steps))
    print("Avg Reward: {}".format(np.mean(tot_reward)))

    # Create a video of the dataset
    if save_video:
        save_frames_as_video(frames, filename=vid_name)

    # Save into dataset
    print("Saved {} demonstrations to {}".format(num_rollout, save_dir))
    # shape [num_demo, ep_len, data_dim]
    total_obs = np.array(total_obs, dtype='object')
    total_acs = np.array(total_acs, dtype='object')
    np.savez(save_dir,
             obs=total_obs,
             acs=total_acs)



def create_hybrid_dataset(sess, graph, ph, env, save_dir, num_transitions=20, save_video=True):

    expert_deque = deque(maxlen=num_transitions)
    learner_deque = deque(maxlen=num_transitions)

    #================ EXPERT DATASET ================
    frames = []
    tot_reward = []
    steps = 0

    while steps < num_transitions:
        done = False
        ep_reward = 0.
        obs = env['expert']['env'].reset()


        while not done:
            # Save dataset as video
            if save_video:
                img = env['expert']['env'].render(mode='rgb_array')
                frames.append(img)

            # Get next action
            raw_action = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
                                                                        ph['expert']['is_training']: False})

            raw_action = raw_action[0]


            # Step in environment
            ## Add slight noise to the action space
            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
#            noisy_action = raw_action
            next_obs, reward, done, info = env['expert']['env'].step(noisy_action)


            ep_reward += reward
            steps += 1

            # Fill up the expert deque
            expert_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
            expert_deque.append(expert_transition)

            obs = next_obs

        # Keep track of reward
        tot_reward.append(ep_reward)


    # Create a video of the dataset
    if save_video:
        print("Saving video")
        save_frames_as_video(frames, filename='expert_dataset')

    print("Expert")
    print("Num Transitions: {}".format(steps))
    print("Avg Reward: {}".format(np.mean(tot_reward)))


#    #================ IDENTITY DATASET ================
#    frames = []
#    tot_reward = []
#    steps = 0
#
#    while steps < num_transitions:
#        done = False
#        ep_reward = 0.
#        obs = env['expert']['env'].reset()
#
#
#        while not done:
#            # Save dataset as video
#            if save_video:
#                img = env['expert']['env'].render(mode='rgb_array')
#                frames.append(img)
#
#            # Get next action
#            raw_action = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
#                                                                        ph['expert']['is_training']: False})
#
#            raw_action = raw_action[0]
#
#
#            # Step in environment
#            ## Add slight noise to the action space
#            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
##            noisy_action = raw_action
#            next_obs, reward, done, info = env['expert']['env'].step(noisy_action)
#
#
#            ep_reward += reward
#            steps += 1
#
#            # Fill up the expert deque
#            learner_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
#            learner_deque.append(learner_transition)
#
#            obs = next_obs
#
#        # Keep track of reward
#        tot_reward.append(ep_reward)
#
#
#    print("-----------------------")
#    print("Learner")
#    print("Num Transitions: {}".format(steps))
#    print("Avg Reward: {}".format(np.mean(tot_reward)))
#
#    # Create a video of the dataset
#    if save_video:
#        print("Saving video")
#        save_frames_as_video(frames, filename='learner_dataset')


#    #================ D_R2R DATASET ================
#    frames = []
#    tot_reward = []
#    steps = 0
#
#    while steps < num_transitions:
#        done = False
#        ep_reward = 0.
#        obs = env['learner']['env'].reset()
#
#
#        while not done:
#            # Save dataset as video
#            if save_video:
#                img = env['learner']['env'].render(mode='rgb_array')
#                frames.append(img)
#
#            # Get next action
#            raw_action = sess.run(graph['expert']['action'], feed_dict={ph['expert']['state']: obs[None],
#                                                                        ph['expert']['is_training']: False})
#
#            raw_action = raw_action[0] / 10.
#
#
#            # Step in environment
#            ## Add slight noise to the action space
#            noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape) / 10.
##            noisy_action = raw_action
#            next_obs, reward, done, info = env['learner']['env'].step(noisy_action)
#
#
#            ep_reward += reward
#            steps += 1
#
#            # Fill up the expert deque
#            learner_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
#            learner_deque.append(learner_transition)
#
#            obs = next_obs
#
#        # Keep track of reward
#        tot_reward.append(ep_reward)
#
#
#    print("-----------------------")
#    print("Learner")
#    print("Num Transitions: {}".format(steps))
#    print("Avg Reward: {}".format(np.mean(tot_reward)))
#
#    # Create a video of the dataset
#    if save_video:
#        print("Saving video")
#        save_frames_as_video(frames, filename='learner_dataset')


    #========================== LEARNER DATASET ==========================
    frames = []
    tot_reward = []
    steps = 0

    while steps < num_transitions:
        done = False
        ep_reward = 0
        env['expert']['env'].reset()
        obs = env['learner']['env'].reset()

        while not done:
                mapped_state_raw, raw_action = sess.run([graph['learner']['mapped_state'], graph['learner']['action']],
                                                         feed_dict={ph['learner']['state']: obs[None],
                                                                    ph['learner']['is_training']: False})

                raw_action = raw_action[0]
                mapped_state_raw = mapped_state_raw[0]

                env['expert']['env'].env.set_state_from_obs(mapped_state_raw)
#		self.env['expert']['env'].set_state_from_obs(mapped_state_raw)

                # Render
                if save_video:
                        # Concatenate learner and expert images
                        limg = env['learner']['env'].render(mode='rgb_array')
                        eimg = env['expert']['env'].render(mode='rgb_array')
                        img = np.concatenate([limg, eimg], axis=1)
                        frames.append(img)

                # Step
                noisy_action = raw_action + np.random.normal(0, 0.05, size=raw_action.shape)
#                noisy_action = raw_action
                next_obs, reward, done, info = env['learner']['env'].step(noisy_action)

                ep_reward += reward
                steps += 1

                # Fill up the expert deque
                learner_transition = (obs, noisy_action, reward, next_obs, 0.0 if done else 1.0, raw_action, 0., 0., 0.)
                learner_deque.append(learner_transition)

                obs = next_obs

        tot_reward.append(ep_reward)



    print("-----------------------")
    print("Learner")
    print("Num Transitions: {}".format(steps))
    print("Avg Reward: {}".format(np.mean(tot_reward)))

    # Create a video of the dataset
    if save_video:
        print("Saving learner video")
        save_frames_as_video(frames, filename='learner_dataset')

    hybrid_dataset = shelve.open(save_dir, writeback=True)
    hybrid_dataset['expert'] = expert_deque
    hybrid_dataset['learner'] = learner_deque
    hybrid_dataset.close()



