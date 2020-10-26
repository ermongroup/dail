from gym.envs.registration import register

# MuJoCO environments
register(id='Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof:Reacher2DOFEnv', max_episode_steps=60)
register(id='Reacher2DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_corner:Reacher2DOFCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_wall:Reacher2DOFWallEnv', max_episode_steps=60)

# dynamics altered
register(id='Reacher2DOFSto-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_sto:Reacher2DOFStoEnv', max_episode_steps=60)
register(id='Reacher2DOFAct-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act:Reacher2DOFActEnv', max_episode_steps=60)
register(id='Reacher2DOFActCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_corner:Reacher2DOFActCornerEnv', max_episode_steps=60)
register(id='Reacher2DOFActWall-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_wall:Reacher2DOFActWallEnv', max_episode_steps=60)

# embodiment altered
register(id='Reacher3DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=60)
register(id='Reacher3DOFCorner-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_corner:Reacher3DOFCornerEnv', max_episode_steps=60)
register(id='Reacher3DOFWall-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_wall:Reacher3DOFWallEnv', max_episode_steps=60)

# push environments
register(id='Reacher2DOFPush-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_push:Reacher2DOFPushEnv', max_episode_steps=500)
register(id='Reacher2DOFActPush-v0', entry_point='dail.reacher_env.mujoco.reacher_2dof_act_push:Reacher2DOFActPushEnv', max_episode_steps=500)
register(id='Reacher3DOFPush-v0', entry_point='dail.reacher_env.mujoco.reacher_3dof_push:Reacher3DOFPushEnv', max_episode_steps=500)

# viewpoint altered
register(id='TP_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.tp_reacher_2dof:TP_Reacher2DOFEnv', max_episode_steps=60)
register(id='TP_WRITE_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.tp_write_reacher_2dof:TP_WRITE_Reacher2DOFEnv', max_episode_steps=500)
register(id='WRITE_Reacher2DOF-v0', entry_point='dail.reacher_env.mujoco.write_reacher_2dof:WRITE_Reacher2DOFEnv', max_episode_steps=500)


# Longer reachers
register(id='Reacher4DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_4dof:Reacher4DOFEnv', max_episode_steps=60)
register(id='Reacher5DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_5dof:Reacher5DOFEnv', max_episode_steps=70)
register(id='Reacher6DOF-v0', entry_point='dail.reacher_env.mujoco.reacher_6dof:Reacher6DOFEnv', max_episode_steps=80)
