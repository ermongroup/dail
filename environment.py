import gym
from gym import Wrapper
from matplotlib import pyplot as plt
import numpy as np
import math
import gym
from gym import utils, spaces, logger
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env
from gym.wrappers.time_limit import TimeLimit


# Create a dictionary of environments and return
def create_env(env_params, seed_dict):
    env_dict = {}
    print('_______________')
    for d_, n_ in env_params.items():
        # Unmodified gym environments
        if n_ == 'cartpole':
            curenv = CustomCartPoleEnv()
            curenv = TimeLimit(curenv, max_episode_steps=200)
            env_type = 'nogoal'
        elif n_ == 'mountaincar':
            curenv = gym.make('MountainCarContinuous-v0')
            curenv = CustomMountainCarEnv(curenv)
            env_type = 'nogoal'
        elif n_ == 'pendulum':
            curenv = gym.make('Pendulum-v0')
            curenv = CustomPendulumEnv(curenv, d_)
            env_type = 'nogoal'
        elif n_ == 'trunc_pendulum':
            curenv = gym.make('Pendulum-v0')
            curenv = TruncatedPendulumEnv(curenv)
            env_type = 'nogoal'
        elif n_ == 'permutedpendulum':
            curenv = gym.make('Pendulum-v0')
            curenv = PermutedPendulumEnv(curenv, d_)
            env_type = 'nogoal'
        elif n_ == 'modifiedpendulum':
            curenv = gym.make('Pendulum-v0')
            curenv = ModifiedPendulumEnv(curenv)
            env_type = 'nogoal'
        elif n_ == 'invpendulum':
            curenv = gym.make('InvertedPendulum-v2')
            curenv = InvertedPendulumEnv(curenv)
            env_type = 'nogoal'
        elif n_ == 'swimmer':
            curenv = gym.make('Swimmer-v1')
            env_type = 'nogoal'
        elif n_ == 'snake3':
            curenv = gym.make('SnakeThree-v1')
            env_type = 'nogoal'
        elif n_ == 'snake4':
            curenv = gym.make('SnakeFour-v1')
            env_type = 'nogoal'
        elif n_ == 'snake5':
            curenv = gym.make('SnakeFive-v1')
            env_type = 'nogoal'
        elif n_ == 'snake7':
            curenv = gym.make('SnakeSeven-v1')
            env_type = 'nogoal'

        elif n_ == 'reacher2':
            curenv = gym.make('Reacher2DOF-v0')
            env_type = 'goal'
        elif n_ == 'reacher2_corner':
            curenv = gym.make('Reacher2DOFCorner-v0')
            env_type = 'goal'
        elif n_ == 'reacher2_wall':
            curenv = gym.make('Reacher2DOFWall-v0')
            env_type = 'goal'

        # Dynamics
        elif n_ == 'reacher2_act':
            curenv = gym.make('Reacher2DOFAct-v0')
            env_type = 'goal'
        elif n_ == 'reacher2_act_wall':
            curenv = gym.make('Reacher2DOFActWall-v0')
            env_type = 'goal'
        elif n_ == 'reacher2_act_corner':
            curenv = gym.make('Reacher2DOFActCorner-v0')
            env_type = 'goal'

        # Embodiment
        elif n_ == 'reacher3':
            curenv = gym.make('Reacher3DOF-v0')
            env_type = 'goal'
        elif n_ == 'reacher3_wall':
            curenv = gym.make('Reacher3DOFWall-v0')
            env_type = 'goal'
        elif n_ == 'reacher3_corner':
            curenv = gym.make('Reacher3DOFCorner-v0')
            env_type = 'goal'

        # Push
        elif n_ == 'reacher2_push':
            curenv = gym.make('Reacher2DOFPush-v0')
            env_type = 'goal'
        elif n_ == 'reacher2_act_push':
            curenv = gym.make('Reacher2DOFActPush-v0')
            env_type = 'goal'
        elif n_ == 'reacher3_push':
            curenv = gym.make('Reacher3DOFPush-v0')
            env_type = 'goal'

        # Viewpoint
        elif n_ == 'tp_reacher2':
            curenv = gym.make('TP_Reacher2DOF-v0')
            env_type = 'goal'
        elif n_ == 'tp_write_reacher2':
            curenv = gym.make('TP_WRITE_Reacher2DOF-v0')
            env_type = 'goal'
        elif n_ == 'write_reacher2':
            curenv = gym.make('WRITE_Reacher2DOF-v0')
            env_type = 'goal'

        # Longer reachers
        elif n_ == 'reacher4':
            curenv = gym.make('Reacher4DOF-v0')
            env_type = 'goal'
        elif n_ == 'reacher5':
            curenv = gym.make('Reacher5DOF-v0')
            env_type = 'goal'
        elif n_ == 'reacher6':
            curenv = gym.make('Reacher6DOF-v0')
            env_type = 'goal'
        else:
            print("Unrecognized environment name: {}".format(n_))
            exit(1)

        # Seed the chosen env
        curenv.seed(seed_dict[d_])

        # State action space of the chosen env
        state_dim = np.prod(np.array(curenv.observation_space.shape))
        action_dim = np.prod(np.array(curenv.action_space.shape))

        env_dict.update({d_: {'env': curenv,
                              'state_dim': state_dim,
                              'action_dim': action_dim,
                              'type': env_type,
                              'name': n_}})

        # Print out environment details
        print('{} env name: {}, state_dim: {}, action_dim: {}, type: {}'.format(d_, n_, state_dim, action_dim, env_type))
    print('_______________')
    return env_dict

## Define custom environments below
class CustomMountainCarEnv(Wrapper):

    def __init__(self, env):
        #print(super(PendulumEnv, self).__init__)
        super(CustomMountainCarEnv, self).__init__(env)
        self.env.env.power = 0.0015
        self.env.env.min_position = -1.6
        self.other_goal_position = -1.45

        print("got to init")

    def step(self, action):
        for i in range(1):
            obs, reward, done, info = self.env.step(action)

            # Allow climbing up the hill in both directions
            position = self.env.env.state[0]

            if position < self.other_goal_position:
                done = True
                reward = 100.
                return obs, reward, done, info

            if done:
                return obs, reward, done, info

        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        self.env.env.state = np.array([self.env.env.np_random.uniform(low=-0.7, high=-0.3), 0.])
        return self.env.env.state

class CustomPendulumEnv(Wrapper):

    def __init__(self, env, domain):
        super(CustomPendulumEnv, self).__init__(env)

    def set_state_from_obs(self, obs):
        '''
        Set the environment state manually from observations
        obs has size [state_dim, ]
        Args:
        Returns:
        '''
        state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])
        self.env.env.state = state

## Define custom environments below
class TruncatedPendulumEnv(Wrapper):

    def __init__(self, env):
        #print(super(PendulumEnv, self).__init__)
        super(TruncatedPendulumEnv, self).__init__(env)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)

        angle = np.arctan2(next_obs[1], next_obs[0])

        if np.absolute(angle) < 0.2:
            done = True
            reward = 100.

        return next_obs, reward, done, info

    def reset(self):
        self.env.reset()
        low = np.array([np.pi-0.2, -1.])
        high = np.array([np.pi+0.2, 1.])
        self.env.env.state = self.env.env.np_random.uniform(low=low, high=high)
        return self.env.env._get_obs()

## Define custom environments below
class PermutedPendulumEnv(Wrapper):

    def __init__(self, env, domain):
        #print(super(PendulumEnv, self).__init__)
        super(PendulumEnv, self).__init__(env)

        if domain == 'expert':
            self.p = np.array([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0]])
        else:
            self.p = np.eye(3)

        self.pinv = np.linalg.inv(self.p)

    def reset(self):
        obs = self.env.reset()
        return np.matmul(self.p, obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return np.matmul(self.p, next_obs), reward, done, info

    '''
    def render(self):
        img = super(PendulumEnv, self).render(mode='rgb_array')
        plt.title(self.domain)
        plt.imshow(img)
    '''


class ModifiedPendulumEnv(Wrapper):
    '''
    Pendulum environment with a longer rod.
    '''

    def __init__(self, env):
        #print(super(PendulumEnv, self).__init__)
        super(ModifiedPendulumEnv, self).__init__(env)


    def step(self, action):
        assert self.env._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.inner_step(action)
        self.env._elapsed_steps += 1

        if self.env._past_limit():
            if self.env.metadata.get('semantics.autoreset'):
                _ = self.env.reset() # automatically reset the env
            done = True

        return observation, reward, done, info

    def inner_step(self, u):
        th, thdot = self.env.env.state # th := theta

        g = 10.
        m = 1.
        l = 0.5
        dt = self.env.env.dt

        u = np.clip(u, -self.env.env.max_torque, self.env.env.max_torque)[0]
        self.env.env.last_u = u # for rendering
        costs = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.env.env.max_speed, self.env.env.max_speed) #pylint: disable=E1111

        self.env.env.state = np.array([newth, newthdot])
        return self.env.env._get_obs(), -costs, False, {}

    def angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)



class CustomCartPoleEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        self.masscart = 0.5 # 0.5
        self.masspole = 0.5 # 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.6 # 0.4 actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.max_force = 2*self.force_mag  ## TODO: May have to change maximum allowed force
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            1.,
            1.,
            np.finfo(np.float32).max])

        '''
        high = np.array([
            self.x_threshold * 2,
            100.,
            1.,
            1.,
            100.])
        '''

        # high = np.array([
        #     self.x_threshold * 2,
        #     np.finfo(np.float32).max,
        #     self.theta_threshold_radians * 2,
        #     np.finfo(np.float32).max])

        #self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(-self.max_force, self.max_force, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def angle_normalize(self, x):
        return (((x+np.pi) % (2*np.pi)) - np.pi)

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def _obs_to_state(self, obs):
        x, x_dot, cos_theta, sin_theta, theta_dot = obs
        return np.array([x, x_dot, np.arctan2(sin_theta, cos_theta), theta_dot])

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state

        # Use continuous actions
        force = np.clip(action, -self.max_force, self.max_force)[0]
        #force = self.force_mag if action==1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass

        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Calculate the new state
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

        # Update self.state
        self.state = (x,x_dot,theta,theta_dot)

        # done =  x < -self.x_threshold \
        #         or x > self.x_threshold \
        #         or theta < -self.theta_threshold_radians \
        #         or theta > self.theta_threshold_radians

        # Reward (similar to the pendulum environment)
        #cost = self.angle_normalize(theta)**2 + .1*theta_dot**2 + .001*(force**2)
        #cost = self.angle_normalize(theta)**2 + .1*theta_dot**2 + .001*(force**2)
        #reward = (-cost + 20.) * 0.05

        reward_theta = (np.cos(theta)+1.0)/2.0
        reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))

        reward = reward_theta*reward_x

        # Only finish if cart slides out of view
        done =  bool(x < -self.x_threshold or x > self.x_threshold)
        #done = False

        if done:
            if self.steps_beyond_done is None:
                # Pole just fell!
                self.steps_beyond_done = 0
                #reward = 1.0
            elif self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -100.

        #return np.array(self.state), reward, done, {}
        return self._get_obs(), reward, done, {}

    def reset(self):
        # self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        # self.steps_beyond_done = None
        # return np.array(self.state)
        # start from center, any pendulum angle, and low angular/x velocity
        #high = np.array([0.05, 0., 0, 1.])
        #high = np.array([0.05, 0.05, 0.05, 0.05])
        #self.state = self.np_random.uniform(-high, high)
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        return self._get_obs()

    def render(self, mode='human'):
        screen_width = 450
        screen_height = 300

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2]) ### TODO: might have to change this

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class InvertedPendulumEnv(Wrapper):
    '''
    Inverted pendulum on a sliding cart
    '''

    def __init__(self, env):
        super(InvertedPendulumEnv, self).__init__(env)

    def step(self, action):
        assert self.env._episode_started_at is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.inner_step(action)
        self.env._elapsed_steps += 1

        if self.env._past_limit():
            if self.env.metadata.get('semantics.autoreset'):
                _ = self.env.reset() # automatically reset the env
            done = True

        return observation, reward, done, info

    def inner_step(self, a):
        ## Modify the reward function similar to pendulum
        reward = 1.0
        self.env.env.do_simulation(a, self.env.env.frame_skip)
        ob = self.env.env._get_obs()

        #notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        #done = not notdone

        return ob, reward, False, {}

    def inner_step(self, u):
        th, thdot = self.env.env.state # th := theta

        g = 10.
        m = 1.
        l = 0.5
        dt = self.env.env.dt

        u = np.clip(u, -self.env.env.max_torque, self.env.env.max_torque)[0]
        self.env.env.last_u = u # for rendering

        costs = self.angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.env.env.max_speed, self.env.env.max_speed) #pylint: disable=E1111

        self.env.env.state = np.array([newth, newthdot])
        return self.env.env._get_obs(), -costs, False, {}


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
        self.t = 0

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        #notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        self.t += 1
        notdone = np.isfinite(ob).all() and (self.t < 199)

        print(self.t)

        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent



class MountainCarEnv(Wrapper):

    def __init__(self, env):
        #print(super(PendulumEnv, self).__init__)
        super(MountainCarEnv, self).__init__(env)
