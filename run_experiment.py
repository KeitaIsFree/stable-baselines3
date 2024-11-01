

from stable_baselines3 import A2C, TD3, SAC, OURS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import numpy

import sys

import envs
from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn

from tqdm import tqdm

import colorsys

import os

from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt

# print("SKEWING ENV")
def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        # print('no skew')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

eval_scores = []


def evaluate(
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    actor,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    skew: float = 1,
    num_timesteps: int = 0
):
    ########
    # ignoring eval_episodes, only doing one
    ########
    # envs = make_env(env_id, 0)
    env = gym.make(env_id)
    # note: qf1 and qf2 are not used in this script

    obs, _ = env.reset()

    action_log = []
    pos_log = [(0, 0)]
    ob_log = [False]
    r_log = [0]
    ep_r = 0
    while True:
        with torch.no_grad():
            # actions = numpy.array([(numpy.random.rand() * 2 - 1.0, numpy.random.rand() * 2 - 1.0)])
            if hasattr(actor, 'actor_e'):
                action = actor.actor_e(torch.tensor([obs], device=device), deterministic=True)[0]
            else:
                action = actor.predict(obs, deterministic=True)[0]
            action_log.append(action)
            # raw_1 = float(input('act_1?: '))
            # raw_2 = float(input('act_2?: '))
            # actions = [(raw_1, raw_2)]

        # print(actor.policy.evaluate_actions(torch.Tensor(obs).to('cuda:1'), torch.Tensor(actions).to('cuda:1')))

        next_obs, r, done, trun, infos = env.step(action.cpu())
        ep_r += r
        obs = next_obs
        pos_log.append(next_obs)
        r_log.append(r)
        ob_log.append(False)

        if done or trun:
            break
    

    # # plt.clf()
    # for i in range(1, len(pos_log)):
    #     marker = 'X' if ob_log[i] else 'o'
    #     c = 'r' if ob_log[i] else 'b'
    #     c = colorsys.hsv_to_rgb(max(0, float(r_log[i] + 300) / 500), 1, 1)
    #     plt.scatter(pos_log[i][0], pos_log[i][1], marker=marker, c=[c])
    # # plt.plot([e[0] for e in pos_log], [e[1] for e in pos_log])
    # plt.xlim(-500, 500)
    # plt.ylim(-500, 500)
    # plt.savefig(f'gaussian_{sys.argv[1]}/checkpoint_{num_timesteps}.png')


    return ep_r

results = []
TOTAL_TIMESTEPS = 50000

ENV_NAME = 'GolfEnv-v0'
EXP_NAME = f'Golf-OURS-1e+0-sde-pibe0.5'
DEVICE = 'cuda:1'

class evaluateCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.num_timesteps % (TOTAL_TIMESTEPS / 100) == 0:
            result = evaluate(make_env, ENV_NAME, 1, "ppp", self.model, num_timesteps=self.num_timesteps, device=DEVICE)
            results.append(result)
            self.model.logger.record('eval/ep_r', result.item())
            # print('noskew')
            # print(results[-1][-1])
        return True

from stable_baselines3.common.noise import NormalActionNoise

if __name__=="__main__":

    seed = sys.argv[1]


    os.makedirs(f'runs/{EXP_NAME}', exist_ok=True)
    os.chdir(f'runs/{EXP_NAME}')
    os.makedirs(f'gaussian_{seed}/scalars/charts', exist_ok=True)


    env = make_vec_env(ENV_NAME, n_envs=1, vec_env_cls=SubprocVecEnv)
    # noskew
    # model = TD3("MlpPolicy", env, device=DEVICE, learning_rate=3e-4, action_noise=NormalActionNoise(numpy.zeros(4), numpy.ones(4) / 1000), tensorboard_log=f'.')
    # model = A2C("MlpPolicy", env, device=DEVICE, learning_rate=3e-4, ent_coef=1e-3, tensorboard_log=f'.')
    # model = SAC("MlpPolicy", env, device=DEVICE, learning_rate=3e-4, ent_coef=1e-5, tensorboard_log=f'.', use_sde=True)
    model = OURS("MlpPolicy", env, device=DEVICE, learning_rate=3e-4, ent_coef=1e+0, tensorboard_log=f'.', use_sde=True)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=evaluateCallback())
    # results.append(evaluate(make_env, "AbsEnv-v0", 1, "ppp", model, skew=10))

    # with open(f'gaussian_{seed}/scalars/charts/eval_return', 'w') as f:
    #     f.write('value' + '\n')
    #     for va in results:
    #         f.write(str(va[0][0]))
    #         f.write('\n')
        

# print(results)
# print(numpy.mean(results))