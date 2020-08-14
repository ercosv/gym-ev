from gym.envs.registration import register


register(
    id='SlipperyWalkSeven_ev-v0',
    entry_point='gym_ev.env:WalkEnv_ev',
    # left-most and right-most states are terminal
    kwargs={'n_states': 7, 'p_stay': 0.5*2/3., 'p_backward': 0.5*1/3.},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
register(
    id='Investment-v0',
    entry_point='gym_ev.env:WalkEnv_ev',
    # left-most and right-most states are terminal
    kwargs={'nQ': 7, 'nR': 3, 'p11': 0.8., 'p12': 0.15., 'p13': 0.05., 'p21': 0.15., 'p22': 0.7., 'p23': 0.15., 'p31': 0.05., 'p32': 0.15., 'p33': 0.7.},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
