"""MagicBot velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import gen1_flat_env_cfg, gen1_rough_env_cfg, z1_flat_env_cfg, z1_rough_env_cfg
from .rl_cfg import magicbot_ppo_runner_cfg

for task_id, env_fn, model, flat in (
    ("RobotLab-MJLab-Velocity-Rough-MagicLab-Bot-Gen1", gen1_rough_env_cfg, "gen1", False),
    ("RobotLab-MJLab-Velocity-Flat-MagicLab-Bot-Gen1", gen1_flat_env_cfg, "gen1", True),
    ("RobotLab-MJLab-Velocity-Rough-MagicLab-Bot-Z1", z1_rough_env_cfg, "z1", False),
    ("RobotLab-MJLab-Velocity-Flat-MagicLab-Bot-Z1", z1_flat_env_cfg, "z1", True),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_fn(),
        play_env_cfg=env_fn(play=True),
        rl_cfg=magicbot_ppo_runner_cfg(model, flat=flat),
        runner_cls=VelocityOnPolicyRunner,
    )
