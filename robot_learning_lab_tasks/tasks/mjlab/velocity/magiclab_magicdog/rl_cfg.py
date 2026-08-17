from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

MAGICDOG_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-MagicLab-Dog"
MAGICDOG_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-MagicLab-Dog"


def magicdog_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("magicdog_rough", 20_000)


def magicdog_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("magicdog_flat", 5_000)
