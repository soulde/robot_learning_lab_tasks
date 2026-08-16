"""DR02 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.env_cfgs import (
    dr02_flat_env_cfg,
    dr02_rough_env_cfg,
)
from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.rl_cfg import (
    dr02_flat_ppo_runner_cfg,
    dr02_rough_ppo_runner_cfg,
)

DR02_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-DR02"
DR02_FLAT_TASK = "RobotLab-MJLab-Velocity-Flat-DR02"

register_mjlab_task(
    task_id=DR02_ROUGH_TASK,
    env_cfg=dr02_rough_env_cfg(),
    play_env_cfg=dr02_rough_env_cfg(play=True),
    rl_cfg=dr02_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=DR02_FLAT_TASK,
    env_cfg=dr02_flat_env_cfg(),
    play_env_cfg=dr02_flat_env_cfg(play=True),
    rl_cfg=dr02_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
