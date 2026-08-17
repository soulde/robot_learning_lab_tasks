"""DDTRobot Tita velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.wheeled_rl_cfg import runner_cfg

from .env_cfgs import ddtrobot_tita_flat_env_cfg, ddtrobot_tita_rough_env_cfg

for terrain, env_fn, flat in (("Rough", ddtrobot_tita_rough_env_cfg, False),
                              ("Flat", ddtrobot_tita_flat_env_cfg, True)):
    register_mjlab_task(
        task_id=f"RobotLab-MJLab-Velocity-{terrain}-DDTRobot-Tita",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True),
        rl_cfg=runner_cfg("ddtrobot_tita", flat=flat), runner_cls=VelocityOnPolicyRunner,
    )
