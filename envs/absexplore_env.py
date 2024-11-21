import gymnasium as gym
import numpy

class AbsExploreEnv(gym.Env):
    def __init__(self, act_dim=2, obs_dim=1, skew=1, A=0, render_mode='rgb_array'):
        super(AbsExploreEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim, ))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim, ))

        self.A = A
        self.skew = skew

    def reset(self, seed=None, options=None):
        assert options is None
        return numpy.zeros(self.observation_space.shape, dtype=numpy.float32), {}

    def step(self, action):

        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        rew = self.A * self.action_space.shape[0]

        for act_i in range(self.action_space.shape[0]):
            a = action[act_i] * 3.0
            if a < 0:
                rew += a + 1
            elif a < 1:
                rew += -a + 1
            elif a < 1.5:
                rew += 4 * a - 4
            else:
                rew += -4 * a + 8
        

        return numpy.zeros(self.observation_space.shape, dtype=numpy.float32), rew, True, False, {}

    def render(self):
        assert False