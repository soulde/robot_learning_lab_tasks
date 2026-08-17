"""Unitree Go2W velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import unitree_go2w_flat_env_cfg, unitree_go2w_rough_env_cfg
from .rl_cfg import runner_cfg

for terrain, env_fn, flat in (("Rough", unitree_go2w_rough_env_cfg, False),
                              ("Flat", unitree_go2w_flat_env_cfg, True)):
    register_mjlab_task(task_id=f"RobotLab-MJLab-Velocity-{terrain}-Unitree-Go2W",
        env_cfg=env_fn(), play_env_cfg=env_fn(play=True), rl_cfg=runner_cfg("go2w", flat=flat),
        runner_cls=VelocityOnPolicyRunner)
