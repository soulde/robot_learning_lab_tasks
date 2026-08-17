from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg

ANYMAL_D_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-ANYmal-D"
ANYMAL_D_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-ANYmal-D"


def anymal_d_rough_ppo_runner_cfg():
    cfg = quadruped_ppo_runner_cfg("anymal_d_rough", 1_500)
    cfg.algorithm.entropy_coef = 0.005
    cfg.save_interval = 50
    return cfg


def anymal_d_flat_ppo_runner_cfg():
    cfg = quadruped_ppo_runner_cfg("anymal_d_flat", 300)
    cfg.actor.hidden_dims = (128, 128, 128)
    cfg.critic.hidden_dims = (128, 128, 128)
    cfg.algorithm.entropy_coef = 0.005
    cfg.save_interval = 50
    return cfg
