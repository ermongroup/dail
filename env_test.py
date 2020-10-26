import os, sys
sys.path.append(os.getcwd() + '/..')

import gym
from dail.environments import register
from dail import reacher_env
import time
import pdb
import numpy as np

env = gym.make("Reacher3DOFPush-v0")



while True:
    done = False
    step = 0
    env.reset()
    while not done:
        step += 1
        env.render()
        a = env.action_space.sample()
        obs, r, done, _ = env.step(a)
        print("step: {} | r: {}, a: {}, s: {}".format(step, r, a, obs))
#        print(np.around(obs, 4))
        time.sleep(0.01)
