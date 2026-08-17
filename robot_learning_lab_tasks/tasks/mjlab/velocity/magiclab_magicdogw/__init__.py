"""MagicLab MagicDog-W velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.wheeled_rl_cfg import runner_cfg
from .env_cfgs import magiclab_magicdogw_flat_env_cfg, magiclab_magicdogw_rough_env_cfg

for terrain, env_fn, flat in (("Rough", magiclab_magicdogw_rough_env_cfg, False),
                              ("Flat", magiclab_magicdogw_flat_env_cfg, True)):
    register_mjlab_task(
        task_id=f"RobotLab-MJLab-Velocity-{terrain}-MagicLab-Dog-W",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True),
        rl_cfg=runner_cfg("magiclab_magicdogw", flat=flat), runner_cls=VelocityOnPolicyRunner,
    )
