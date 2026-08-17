"""RL configuration for Unitree B2 velocity tasks."""

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

UNITREE_B2_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-Unitree-B2"
UNITREE_B2_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Unitree-B2"


def unitree_b2_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_b2_rough", 20_000)


def unitree_b2_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_b2_flat", 5_000)
