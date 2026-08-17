"""RL configuration for the DR02 Pro velocity task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import dr02_rough_ppo_runner_cfg


def dr02_pro_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create the DR02 Pro rough walking PPO configuration."""
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = "deeprobotics_dr02_pro_rough"
    return cfg
