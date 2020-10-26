from gym.envs.registration import register

register(
    id='SnakeThree-v0',
    entry_point='dail.test_env:SnakeEnv',
)


