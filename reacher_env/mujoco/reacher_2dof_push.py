import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
import random
import pdb

import os

class Reacher2DOFPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.reset_called = False
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.abspath(__file__))+'/assets/reacher_2dof_push.xml', 2)
        self.viewer = None

    def _step(self, a):
        vec = self.get_body_com("target")-self.get_body_com("destination")

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.abs(a).sum()
        a = np.clip(a, -0.002, 0.002)

        proximity_threshold = 0.05
        reward = reward_dist + 100 * int(-reward_dist < proximity_threshold)
        reward = reward * 0.25

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        done = (-reward_dist < proximity_threshold) and self.reset_called

        '''
        if np.linalg.norm(vec) < 0.02:
            done = True
        '''
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = [0, 0, 0] # [-0.1, 0, 0]
        self.viewer.cam.elevation = -90 # -60
        self.viewer.cam.distance = 1.1
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        self.reset_called = True
        n_joints = 2
        max_reachable_len = (n_joints+1) * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        bias_low = 0.8
        bias_high = 0.9
        bias2_low = -2.8
        bias2_high = 2.8
        first_bias = self.np_random.uniform(low=bias_low, high=bias_high, size=1)
        second_bias = self.np_random.uniform(low=bias2_low, high=bias2_high, size=1)
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = first_bias
        #qpos[1:2] = second_bias
        #print("using super diverse starts")
        while True:
            # Diverse goals
            self.goal = self.np_random.uniform(low=-max_reachable_len, high=max_reachable_len, size=2)

            # Corner deterministic
            scale = np.sqrt(2)
            det_corner_options = [[-0.25/scale, -0.25/scale], [-0.25/scale, 0.25/scale], [0.25/scale, -0.25/scale], [0.25/scale, 0.25/scale],
                                  [-0.2/scale, -0.2/scale], [-0.2/scale, 0.2/scale], [0.2/scale, -0.2/scale], [0.2/scale, 0.2/scale],
                                  [-0.15/scale, -0.15/scale], [-0.15/scale, 0.15/scale], [0.15/scale, -0.15/scale], [0.15/scale, 0.15/scale]]


            # Wall deterministic
            det_wall_options = [[0.25, 0], [-0.25, 0], [0, 0.25], [0, -0.25],
                                [0.2, 0], [-0.2, 0], [0, 0.2], [0, -0.2],
                                [0.15, 0], [-0.15, 0], [0, 0.15], [0, -0.15]]

            det_wall_8 = det_wall_options[:8]
            det_wall_4 = det_wall_options[:4]
            det_wall_1 = det_wall_options[:1]

            goals_15 = [[0.15, 0], [0, 0.15], [0, -0.15], [0.15/scale, 0.15/scale], [0.15/scale, -0.15/scale],
                       [0.2, 0], [0, 0.2], [0, -0.2], [0.25, 0], [0, 0.25], [0, -0.25],
                       [0.2/scale, -0.2/scale], [0.2/scale, 0.2/scale],
                       [0.25/scale, -0.25/scale], [0.25/scale, 0.25/scale]]

            goals_5 = goals_15[:5]

            # Single goal option
            single_goal = [0., -0.2]

            # Single transfer goal
            single_transfer = [-0.25, 0.]

            # Reduced wall
            red_wall_options = [[-0.25, 0], [0, 0.25], [0, -0.25],
                                [-0.2, 0], [0, 0.2], [0, -0.2],
                                [-0.15, 0], [0, 0.15], [0, -0.15]]

            # Both
            det_wall_corner_options = det_wall_options + det_corner_options


            ## Goal selection

            # one goal
            chosen_goal = det_wall_options[5]
            print("training with 1 box location")

            # 12 goals
#            chosen_goal = det_wall_options[random.randrange(len(det_wall_options))]
#            print("training with 12 wall goals")

#            chosen_goal = det_corner_options[random.randrange(len(det_corner_options))]
#            print("training with 12 corner goals")
            #chosen_goal = red_wall_options[random.randrange(len(red_wall_options))]
            #chosen_goal = single_goal
            #chosen_goal = single_transfer
            #chosen_goal = det_wall_corner_options[random.randrange(len(det_wall_corner_options))]
            #print("training with wall + corner goals")
            #chosen_goal = goals_15[random.randrange(len(goals_15))]
            #print("training with 15 goals")
            #chosen_goal = goals_5[random.randrange(len(goals_5))]
            #print("training with 5 goals")

            # 8 goals
#            chosen_goal = det_wall_8[random.randrange(len(det_wall_8))]
#            print("training with 8 wall goals")
            #chosen_goal = det_corner_8[random.randrange(len(det_corner_8))]
            #print("training with 8 corner goals")

            # 4 goals
            #chosen_goal = det_wall_4[random.randrange(len(det_wall_4))]
            #print("training with 4 wall goals")

            # 1 goal
            #chosen_goal = det_wall_1[random.randrange(len(det_wall_1))]
            #print("training with 1 wall goal")


            self.goal = np.array(chosen_goal)
#            print("training with div goals")

            if np.linalg.norm(self.goal) < max_reachable_len and np.linalg.norm(self.goal) > min_reachable_len:
                break

        qpos[-2:] = self.goal
        qpos[-4:-2] = self.goal + np.array([0.2, 0.2])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = 2

        theta = self.model.data.qpos.flat[:n_joints]

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            np.array([0., 0.]),
#            self.model.data.qpos.flat[n_joints:n_joints+2], # box position
            self.model.data.qvel.flat[:n_joints] # joint velocities
        ])

        #self.get_body_com("fingertip") - self.get_body_com("target")

    def set_state_from_obs(self, obs):
        n_joints = 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        cos_theta = obs[:n_joints]
        sin_theta = obs[n_joints:2*n_joints]
        theta = np.arctan2(sin_theta, cos_theta) # 3
        target = obs[2*n_joints:2*n_joints+2] # 2

        qpos = np.concatenate([theta, target, self.goal], axis=0)
        qvel[:n_joints] = obs[2*n_joints+2:2*n_joints+2+n_joints] # 5

        self.set_state(qpos, qvel)

    def _get_viewer(self):
        if self.viewer is None:
            size = 128
            self.viewer = mujoco_py.MjViewer(visible=True, init_width=size, init_height=size, go_fast=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer
