"""Zsibot ZSL1W velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.wheeled_rl_cfg import runner_cfg
from .env_cfgs import zsibot_zsl1w_flat_env_cfg, zsibot_zsl1w_rough_env_cfg

for terrain, env_fn, flat in (("Rough", zsibot_zsl1w_rough_env_cfg, False),
                              ("Flat", zsibot_zsl1w_flat_env_cfg, True)):
    register_mjlab_task(
        task_id=f"RobotLab-MJLab-Velocity-{terrain}-Zsibot-ZSL1W",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True),
        rl_cfg=runner_cfg("zsibot_zsl1w", flat=flat), runner_cls=VelocityOnPolicyRunner,
    )
