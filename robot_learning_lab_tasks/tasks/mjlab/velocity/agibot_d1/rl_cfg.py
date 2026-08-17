from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

AGIBOT_D1_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-Agibot-D1"
AGIBOT_D1_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Agibot-D1"


def agibot_d1_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("agibot_d1_rough", 20_000)


def agibot_d1_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("agibot_d1_flat", 5_000)
