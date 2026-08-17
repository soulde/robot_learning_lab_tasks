"""RL configuration for Unitree wheeled velocity tasks."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_rl_cfgs import quadruped_ppo_runner_cfg


def runner_cfg(model: str, *, flat: bool) -> RslRlOnPolicyRunnerCfg:
    return quadruped_ppo_runner_cfg(
        experiment_name=f"unitree_{model}_{'flat' if flat else 'rough'}",
        max_iterations=1_500 if flat else 3_000,
    )
