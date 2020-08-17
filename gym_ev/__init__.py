from gym.envs.registration import register

register(
    id='Investment-v1',
    entry_point='gym_ev.env:WalkEnv_ev',
    # left-most and right-most states are terminal
    kwargs={'n_states': 500, 'nQ': 100, 'nR': 5},
    max_episode_steps=300,
    reward_threshold=1.0,
    nondeterministic=True,
)
