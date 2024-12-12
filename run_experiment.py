

from stable_baselines3 import A2C, TD3, SAC, OURS, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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

import hydra
from omegaconf import DictConfig, OmegaConf

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
                action = actor.actor_e.predict(obs, deterministic=True)[0]
            else:
                action = actor.predict(obs, deterministic=True)[0]
            
            action_log.append(action)
            # raw_1 = float(input('act_1?: '))
            # raw_2 = float(input('act_2?: '))
            # actions = [(raw_1, raw_2)]

        # print(actor.policy.evaluate_actions(torch.Tensor(obs).to('cuda:1'), torch.Tensor(actions).to('cuda:1')))

        # if ALGO == 'OURS':
        #     action = action.cpu()[0]


        next_obs, r, done, trun, infos = env.step(action)
        ep_r += r
        obs = next_obs
        pos_log.append(next_obs)
        r_log.append(r)
        ob_log.append(False)

        if done or trun:
            break
    

    # plt.clf()
    # for i in range(1, len(pos_log)):
    #     marker = 'X' if ob_log[i] else 'o'
    #     c = 'r' if ob_log[i] else 'b'
    #     c = colorsys.hsv_to_rgb(max(0, float(r_log[i] + 300) / 500), 1, 1)
    #     plt.scatter(pos_log[i][0], pos_log[i][1], marker=marker, c=[c], alpha=0.5)
    # # plt.plot([e[0] for e in pos_log], [e[1] for e in pos_log])
    # plt.xlim(-1000, 1000)
    # plt.ylim(-1000, 1000)
    # plt.savefig(f'gaussian_{sys.argv[1]}/checkpoint_{num_timesteps}.png')
    # plt.savefig(f'gaussian_{sys.argv[1]}/checkpoint_{num_timesteps}.eps')

    if True:
        return ep_r
    q_vals = [[], []]
    # action_scale = env.action_space.high[0]
    # ACT_LIM = (-1, )
    # action_scale = 3.0
    if ALGO == 'A2C' or ALGO == 'PPO':
        return ep_r
    maxes = [-100, -100]
    argmaxes = [None, None]
    pi_p = []
    for y in numpy.arange(1, -1.00, -0.02):
        second_values = torch.arange(-1, 1.00, 0.02, dtype=torch.float32)
        first_values = torch.full((100, ), y, dtype=torch.float32)
        dummy = torch.stack([first_values, second_values], dim=1).to(device)
        if hasattr(actor, 'critic'):
            temp = actor.critic(torch.full((100, 1), 0.0, dtype=torch.float32).to(device), dummy)
        else:
            temp = [actor.policy.evaluate_actions(torch.full((100, 1), 0.0, dtype=torch.float32).to(device), dummy)[0]]
        # probs = actor.actor_b.get_log_prob_from_act(torch.full((100, 1), 0.0, dtype=torch.float32).to(device), dummy).tolist()
        # pi_p.append(probs)
        if len(temp) > 1:
            q_0_vals = temp[0].squeeze(1).tolist()
            q_vals[0].append(q_0_vals)
            q_1_vals = temp[1].squeeze(1).tolist()
            q_vals[1].append(q_1_vals)
        else:
            q_0_vals = temp[0].squeeze().tolist()
            q_vals[0].append(q_0_vals)

        max_in_row = max(q_0_vals)
        if max_in_row > maxes[0]:
            argmaxes[0] = (numpy.argmax(q_0_vals) /50 - 1, y)
            maxes[0] = max_in_row

        if len(temp) > 1:
            max_in_row = max(q_1_vals)
            if max_in_row > maxes[1]:
                argmaxes[1] = (numpy.argmax(q_1_vals) /50 - 1, y)
                maxes[1] = max_in_row

    x = numpy.linspace(-1, 1, 100)  # 100 points from -5 to 5
    y = numpy.linspace(-1, 1, 100)
    X, Y = numpy.meshgrid(x,y)

    for i in range(len(temp)):
        plt.clf()
        # plt.contour(X, Y, pi_p)
        plt.imshow(q_vals[i], cmap='viridis', aspect='auto', extent=[-1, 1, -1, 1])
        for ac in pi_b_acts:
            plt.plot(ac[0], ac[1], markersize=10, marker='x', c='black', alpha=0.5)
        plt.plot(action[0], action[1], markersize=10, marker='x', c='r')
        plt.text(action[0], action[1], 'pi_e')
        plt.plot(argmaxes[i][0], argmaxes[i][1], markersize=10, marker='x', c='g')
        plt.text(argmaxes[i][0], argmaxes[i][1], 'max')
        plt.colorbar()  # Adds a colorbar to the side
        plt.title('Heatmap')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.savefig(f'gaussian_{seed}/{num_timesteps}_q{i}.pdf')



    return ep_r

results = []

# if ALGO == 'OURS':
#     EXP_NAME = EXP_NAME + "_nf"



class evaluateCallback(BaseCallback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def _on_step(self):
        if self.num_timesteps % (self.cfg.TOTAL_TIMESTEPS / 100) == 0:
            result = evaluate(make_env, self.cfg.ENV_NAME, 1, "ppp", self.model, num_timesteps=self.num_timesteps, device=self.cfg.DEVICE)
            results.append(result)

            if isinstance(result, torch.Tensor):
                result = result.item()
            self.model.logger.record('eval/ep_r', result)
            # print('noskew')
            # print(results[-1][-1])
        return True

from stable_baselines3.common.noise import NormalActionNoise

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:


    os.makedirs(f'runs/{cfg.EXP_NAME}', exist_ok=True)
    os.chdir(f'runs/{cfg.EXP_NAME}')
    os.makedirs(f'gaussian_{cfg.seed}', exist_ok=True)

    checkpoint_callback = CheckpointCallback(save_freq=cfg.TOTAL_TIMESTEPS//100, save_path='./checkpoints/')
    
    env = make_vec_env(cfg.ENV_NAME, n_envs=1, vec_env_cls=SubprocVecEnv)
    if cfg.ALGO == 'TD3':
        model = TD3("MlpPolicy", 
                    env, device=cfg.DEVICE, 
                    policy_kwargs=cfg.policy_kwargs,
                    action_noise=NormalActionNoise(numpy.zeros(2), numpy.ones(2) * cfg.PARAM), 
                    tensorboard_log=f'.', seed=cfg.seed)
    elif cfg.ALGO == 'A2C':
        model = A2C("MlpPolicy", 
                    env, device=cfg.DEVICE, 
                    policy_kwargs=cfg.policy_kwargs,
                    ent_coef=cfg.PARAM, 
                    tensorboard_log=f'.', seed=cfg.seed)
    elif cfg.ALGO == 'SAC':
        model = SAC("MlpPolicy", 
                    env, device=cfg.DEVICE, 
                    policy_kwargs=cfg.policy_kwargs,
                    ent_coef=cfg.PARAM, 
                    tensorboard_log=f'.', seed=cfg.seed)
    elif cfg.ALGO == 'PPO':
        model = PPO("MlpPolicy", 
                    env, device=cfg.DEVICE, 
                    policy_kwargs=cfg.policy_kwargs,
                    ent_coef=cfg.PARAM,
                    tensorboard_log=f'.', seed=cfg.seed)
    elif cfg.ALGO == 'OURS':
        policy_kwargs = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(cfg.policy_kwargs)), cfg.ours_policy_kwargs)
        model = OURS("MlpPolicy", 
                    env, device=cfg.DEVICE, 
                    policy_kwargs=OmegaConf.merge(policy_kwargs, cfg.ours_policy_kwargs), 
                    ablation_mode=cfg.ablation_mode,
                    ent_coef=cfg.PARAM, 
                    tensorboard_log=f'.', seed=cfg.seed)

    model.learn(total_timesteps=cfg.TOTAL_TIMESTEPS, callback=[evaluateCallback(cfg), checkpoint_callback])
    # results.append(evaluate(make_env, "AbsEnv-v0", 1, "ppp", model, skew=10))

    # with open(f'gaussian_{seed}/scalars/charts/eval_return', 'w') as f:
    #     f.write('value' + '\n')
    #     for va in results:
    #         f.write(str(va[0][0]))
    #         f.write('\n')
        

if __name__=="__main__":
    main()


# print(results)
# print(numpy.mean(results))