"""RL configurations for Unitree A1 handstand tasks."""

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

UNITREE_A1_HANDSTAND_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-HandStand-Unitree-A1"
UNITREE_A1_HANDSTAND_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-HandStand-Unitree-A1"


def unitree_a1_handstand_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_handstand_rough", 5_000)


def unitree_a1_handstand_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_handstand_flat", 2_000)
