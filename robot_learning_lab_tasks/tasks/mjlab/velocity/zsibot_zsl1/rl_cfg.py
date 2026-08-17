from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

ZSIBOT_ZSL1_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-Zsibot-ZSL1"
ZSIBOT_ZSL1_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Zsibot-ZSL1"


def zsibot_zsl1_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("zsibot_zsl1_rough", 50_000)


def zsibot_zsl1_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("zsibot_zsl1_flat", 5_000)
