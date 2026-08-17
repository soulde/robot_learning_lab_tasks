"""RL configurations for MagicBot velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import dr02_rough_ppo_runner_cfg


def magicbot_ppo_runner_cfg(model: str, *, flat: bool) -> RslRlOnPolicyRunnerCfg:
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = f"magiclab_bot_{model}_{'flat' if flat else 'rough'}"
    cfg.max_iterations = 1_500 if flat else 3_000
    cfg.save_interval = 50
    cfg.algorithm.entropy_coef = 0.008
    return cfg
