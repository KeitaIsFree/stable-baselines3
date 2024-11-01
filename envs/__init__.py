

import gymnasium as gym


gym.envs.registration.register(
    id='AbsEnv-v0',
    entry_point='envs.abs_env:AbsEnv'
)

gym.envs.registration.register(
    id='AbsExploreEnv-v0',
    entry_point='envs.absexplore_env:AbsExploreEnv'
)

gym.envs.registration.register(
    id='GolfEnv-v0',
    entry_point='envs.golf_env:GolfEnv'
)
