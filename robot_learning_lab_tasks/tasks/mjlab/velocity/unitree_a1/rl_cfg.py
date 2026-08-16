"""RL configuration for Unitree A1 velocity tasks."""

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import (
    quadruped_ppo_runner_cfg,
)

UNITREE_A1_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-Unitree-A1"
UNITREE_A1_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Unitree-A1"


def unitree_a1_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_rough", 3_000)


def unitree_a1_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_flat", 1_000)
