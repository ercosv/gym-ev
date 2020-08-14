from gym.envs.registration import register

register(
    id='Investment-v1',
    entry_point='gym_ev.env:WalkEnv_ev',
    # left-most and right-most states are terminal
    kwargs={'n_states': 300, 'nQ': 100, 'nR': 3, 'p11': 0.93, 'p12': 0.06, 'p13': 0.01, 'p21': 0.01, 'p22': 0.98, 'p23': 0.01 'p31': 0.01, 'p32': 0.06, 'p33': 0.93},   
    max_episode_steps=100,
    reward_threshold=1.0,
    nondeterministic=True,
)
