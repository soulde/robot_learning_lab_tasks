"""RL configuration for DeepRobotics Lite3 velocity tasks."""

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import (
    quadruped_ppo_runner_cfg,
)

DEEPROBOTICS_LITE3_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-DeepRobotics-Lite3"
DEEPROBOTICS_LITE3_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-DeepRobotics-Lite3"


def deeprobotics_lite3_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("deeprobotics_lite3_rough", 3_000)


def deeprobotics_lite3_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("deeprobotics_lite3_flat", 1_000)
