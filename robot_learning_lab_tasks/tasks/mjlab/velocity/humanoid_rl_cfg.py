"""Shared RL configuration for humanoid velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import dr02_rough_ppo_runner_cfg


def runner_cfg(model: str, *, flat: bool) -> RslRlOnPolicyRunnerCfg:
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = f"{model}_{'flat' if flat else 'rough'}"
    cfg.max_iterations = (
        (1_000 if model == "unitree_h1" else 1_500) if flat else (5_000 if model == "openloong_loong" else 3_000)
    )
    if model != "openloong_loong":
        cfg.save_interval = 50
        cfg.algorithm.entropy_coef = 0.008
    return cfg
