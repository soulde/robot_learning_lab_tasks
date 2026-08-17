"""Unitree H1 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from robot_learning_lab_tasks.tasks.mjlab.velocity.humanoid_rl_cfg import runner_cfg
from .env_cfgs import unitree_h1_flat_env_cfg, unitree_h1_rough_env_cfg

for terrain, env_fn, flat in (("Rough", unitree_h1_rough_env_cfg, False),
                              ("Flat", unitree_h1_flat_env_cfg, True)):
    register_mjlab_task(task_id=f"RobotLab-MJLab-Velocity-{terrain}-Unitree-H1",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True), rl_cfg=runner_cfg("unitree_h1", flat=flat),
        runner_cls=VelocityOnPolicyRunner)
