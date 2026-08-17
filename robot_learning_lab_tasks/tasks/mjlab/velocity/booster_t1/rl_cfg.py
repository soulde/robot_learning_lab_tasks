"""RL configurations for Booster T1 velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import dr02_rough_ppo_runner_cfg


def booster_t1_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = "booster_t1_rough"
    cfg.save_interval = 50
    cfg.algorithm.entropy_coef = 0.008
    return cfg


def booster_t1_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = booster_t1_rough_ppo_runner_cfg()
    cfg.experiment_name = "booster_t1_flat"
    cfg.max_iterations = 1_500
    return cfg
