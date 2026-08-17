"""RobotEra XBot velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from robot_learning_lab_tasks.tasks.mjlab.velocity.humanoid_rl_cfg import runner_cfg
from .env_cfgs import robotera_xbot_flat_env_cfg, robotera_xbot_rough_env_cfg

for terrain, env_fn, flat in (("Rough", robotera_xbot_rough_env_cfg, False),
                              ("Flat", robotera_xbot_flat_env_cfg, True)):
    register_mjlab_task(task_id=f"RobotLab-MJLab-Velocity-{terrain}-RobotEra-Xbot",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True), rl_cfg=runner_cfg("robotera_xbot", flat=flat),
        runner_cls=VelocityOnPolicyRunner)
