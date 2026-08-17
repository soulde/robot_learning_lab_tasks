"""Booster T1 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import booster_t1_flat_env_cfg, booster_t1_rough_env_cfg
from .rl_cfg import booster_t1_flat_ppo_runner_cfg, booster_t1_rough_ppo_runner_cfg

for task_id, env_cfg_fn, rl_cfg_fn in (
    ("RobotLab-MJLab-Velocity-Rough-Booster-T1", booster_t1_rough_env_cfg, booster_t1_rough_ppo_runner_cfg),
    ("RobotLab-MJLab-Velocity-Flat-Booster-T1", booster_t1_flat_env_cfg, booster_t1_flat_ppo_runner_cfg),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg_fn(),
        play_env_cfg=env_cfg_fn(play=True),
        rl_cfg=rl_cfg_fn(),
        runner_cls=VelocityOnPolicyRunner,
    )
