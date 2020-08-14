from gym.envs.registration import register


register(
    id='SlipperyWalkSeven_ev-v0',
    entry_point='gym_walk.envs:WalkEnv',
    # left-most and right-most states are terminal
    kwargs={'n_states': 7, 'p_stay': 0.5*2/3., 'p_backward': 0.5*1/3.},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
register(
    id='Investment-v0',
    entry_point='gym_walk.envs:WalkEnv',
    # left-most and right-most states are terminal
    kwargs={'n_states': 7, 'p_stay': 0.5*2/3., 'p_backward': 0.5*1/3.},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
