"""RL configuration for Unitree Go2 velocity tasks."""

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import (
    quadruped_ppo_runner_cfg,
)

UNITREE_GO2_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-Unitree-Go2"
UNITREE_GO2_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Unitree-Go2"


def unitree_go2_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_go2_rough", 3_000)


def unitree_go2_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_go2_flat", 1_000)
