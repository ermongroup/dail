import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
import random

import os

class TP_WRITE_Reacher2DOFEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.offset = np.pi ## Offset is 180 degrees
        self.target_pos = [[0.1, 0.2], [0.2, 0.1], [0.1, 0], [0.15, 0.15], [0.15, 0.05], [0.1, -0.15], [0.2, -0.15]]
        self.cur_target_idx = 0

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(os.path.abspath(__file__))+'/assets/reacher_2dof.xml', 2)
        self.viewer = None

    def dist_from_sp(self, cur_pos, l1, l2):
        return np.abs((l2[1] - l1[1])*cur_pos[0] - (l2[0] - l1[0])*cur_pos[1] + l2[0]*l1[1] - l2[1]*l1[0]) / np.linalg.norm(l2 - l1, ord=2)

    def linear_quad(self, d):
        if d <= 1:
            return 10
        else:
            return d**2

    def _step(self, a):

        done = False

        #vec = self.get_body_com("fingertip")-self.get_body_com("target")
        vec = self.get_body_com("fingertip")[:2]-np.array(self.target_pos[self.cur_target_idx])

        # Reached next target reward
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # Time cost
        reward += -1

        # Straight line reward
        if self.cur_target_idx > 0:
            cur_pos = self.get_body_com("fingertip")[:2]
            prev_target = np.array(self.target_pos[self.cur_target_idx-1])
            next_target = np.array(self.target_pos[self.cur_target_idx])

            dev = self.dist_from_sp(cur_pos=cur_pos, l1=prev_target, l2=next_target)
            #sp_reward = -self.linear_quad(20*dev)
            sp_reward = -10*dev
            reward += sp_reward

        # If reached target position, then give big reward
        if np.linalg.norm(vec) < 0.075:
            self.cur_target_idx += 1
            #reward += 15*(self.cur_target_idx)
            reward += 10

            # Reached the end of the target position list
            if self.cur_target_idx == len(self.target_pos):
                done = True
            else:
                obs = self._get_obs()
                obs[4:6] = np.array(self.target_pos[self.cur_target_idx])
                self.set_state_from_obs(obs)

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.lookat[:] = [0, 0, 0] # [-0.1, 0, 0]
        self.viewer.cam.elevation = -90 # -60
        self.viewer.cam.distance = 1.1
        self.viewer.cam.azimuth = 0

    def reset_model(self):
        self.cur_target_idx = 0

        n_joints = self.model.nq - 2
        max_reachable_len = (n_joints+1) * 0.1 # .1 is the length of each link
        min_reachable_len = 0.1 # joint ranges: inf, .9, 2.8

        # Deterministic initial state
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:1] = self.np_random.uniform(low=3.14, high=4, size=1)
        #qpos[0] = -2.6

        qpos[-2:] = np.array(self.target_pos[0])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        n_joints = self.model.nq - 2

        theta = self.model.data.qpos.flat[:n_joints] + self.offset

        #print("true theta: {}, tp theta: {}".format((theta - self.offset)*(180 / np.pi), theta*(180 / np.pi)))

        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[n_joints:], # target position
            self.model.data.qvel.flat[:n_joints] # joint velocities
        ])
        '''
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            np.array([0, 0]), # target position
            self.model.data.qvel.flat[:n_joints] # joint velocities
        ])
        '''
        #self.get_body_com("fingertip") - self.get_body_com("target")

    def set_state_from_obs(self, obs):
        n_joints = self.model.nq - 2
        qvel = np.zeros((self.model.nv, ))

        # Positions
        cos_theta = obs[:n_joints]
        sin_theta = obs[n_joints:2*n_joints]
        theta = np.arctan2(sin_theta, cos_theta) - self.offset# 3
        target = obs[2*n_joints:2*n_joints+2] # 2

        qpos = np.concatenate([theta, target], axis=0)
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
