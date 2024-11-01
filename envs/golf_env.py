import gymnasium as gym
import numpy

class GolfEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        super(GolfEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=numpy.array([-1000.0, -1000.0]), high=numpy.array([1000.0, 1000.0]), shape=(2, )) # angle from goal, distance from goal
        self.action_space = gym.spaces.Box(low=numpy.array([-1.0, -1.0]), high=numpy.array([1.0, 1.0]), shape=(2, )) # angle from current point, distance

        self.step_count = 1
        self.EP_LEN = 10

        self.INIT_POS = (0, 0)
        self.GOAL_POS = (400, 400)
        self.GREEN_RAD = 200

        self.ACTION_RANGE = 200

        self.pos = self.INIT_POS

    def xy2rad(self, xy):
        if xy[0] == 0:
            return (0, (xy[0] ** 2 + xy[1] ** 2) ** (1/2))
        return (numpy.arctan(xy[1]/xy[0]), (xy[0] ** 2 + xy[1] ** 2) ** (1/2))

    def reset(self, seed=None, options=None):
        assert options is None
        self.step_count = 1
        self.pos = self.INIT_POS
        # return self.xy2rad(self.pos), {}
        return self.pos, {}
    
    def check_OB(self, pos):
        if pos[0] ** 2 + pos[1] ** 2 < self.GREEN_RAD ** 2:
            return False
        if ((pos[0] - self.GOAL_POS[0]) ** 2 + (pos[1] - self.GOAL_POS[1]) ** 2) < self.GREEN_RAD ** 2:
            return False
        if (pos[0] - self.GOAL_POS[0]) ** 2 + pos[1] ** 2 < self.GREEN_RAD ** 2:
            return False
        if (pos[0] - self.GOAL_POS[0] / 2) ** 2 + (pos[1] - self.GOAL_POS[1] / 2) ** 2 < self.GREEN_RAD ** 2:
            return False
        
        if 0 < pos[0] < self.GOAL_POS[0] and -self.GREEN_RAD < pos[1] < self.GREEN_RAD:
            return False
        if self.GOAL_POS[0] - self.GREEN_RAD < pos[0] < self.GOAL_POS[0] + self.GREEN_RAD and 0 < pos[1] < self.GOAL_POS[1]:
            return False

        return True

    def step(self, action):
        self.step_count += 1

        scaled_action = numpy.clip(action, [-1, -1], [1.0, 1.0])
        scaled_action = scaled_action * self.ACTION_RANGE

        new_x = self.pos[0] + scaled_action[0]
        new_y = self.pos[1] + scaled_action[1]
        self.pos = (new_x, new_y)


        if (new_x - self.GOAL_POS[0]) ** 2 + (new_y - self.GOAL_POS[1]) ** 2 < self.GREEN_RAD ** 2:
            done = True
            rew = 100
        # elif self.check_OB(self.pos):
        #     done = True
        #     rew = -1
        else:
            done = False
            rew = -1 / 100 * (((new_x - self.GOAL_POS[0]) ** 2 + (new_y - self.GOAL_POS[1]) ** 2) ** (1/2))

        # return rad_pos, -rad_pos[1] / 10000 if not done else 100, done, self.step_count > 100, {}
        return self.pos, rew, done, self.step_count > self.EP_LEN, {}

    def render(self):
        assert False