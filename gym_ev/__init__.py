from gym.envs.registration import register

register(
    id='Investment-v0',
    entry_point='gym_ev.env:WalkEnv_ev',
    # left-most and right-most states are terminal
    kwargs={'nQ': 7, 'nR': 3, 'p11': 0.8., 'p12': 0.15., 'p13': 0.05., 'p21': 0.15., 'p22': 0.7., 'p23': 0.15., 'p31': 0.05., 'p32': 0.15., 'p33': 0.8.},
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
