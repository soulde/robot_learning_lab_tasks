"""RL configurations for FFTAI GR1 velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import dr02_rough_ppo_runner_cfg


def gr1_ppo_runner_cfg(model: str, *, flat: bool) -> RslRlOnPolicyRunnerCfg:
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = f"fftai_{model}_{'flat' if flat else 'rough'}"
    cfg.max_iterations = 1_500 if flat else 3_000
    return cfg
