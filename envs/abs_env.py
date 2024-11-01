import gymnasium as gym
import numpy

class AbsEnv(gym.Env):
    def __init__(self, act_dim=2, obs_dim=1, skew=1, A=0, render_mode='rgb_array'):
        super(AbsEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim, ))
        self.action_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(act_dim, ))

        self.A = A
        self.skew = skew

    def reset(self, seed=None, options=None):
        assert options is None
        return numpy.zeros(self.observation_space.shape, dtype=numpy.float32), {}

    def step(self, action):

        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        rew = self.A * self.action_space.shape[0]

        for act_i in range(self.action_space.shape[0]):
            shifted_action = action[act_i] + 1
            if shifted_action > 0:
                rew += shifted_action - self.A * numpy.cos(shifted_action * numpy.pi * 2)
            else:
                rew += (- shifted_action * self.skew) - self.A * numpy.cos(shifted_action * self.skew * numpy.pi * 2)
        
        rew = -rew

        return numpy.zeros(self.observation_space.shape, dtype=numpy.float32), rew, True, False, {}

    def render(self):
        assert False