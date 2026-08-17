"""Unitree B2 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_b2.env_cfgs import (
    unitree_b2_flat_env_cfg,
    unitree_b2_rough_env_cfg,
)
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_b2.rl_cfg import (
    UNITREE_B2_FLAT_TASK,
    UNITREE_B2_ROUGH_TASK,
    unitree_b2_flat_ppo_runner_cfg,
    unitree_b2_rough_ppo_runner_cfg,
)

register_mjlab_task(
    task_id=UNITREE_B2_ROUGH_TASK,
    env_cfg=unitree_b2_rough_env_cfg(),
    play_env_cfg=unitree_b2_rough_env_cfg(play=True),
    rl_cfg=unitree_b2_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id=UNITREE_B2_FLAT_TASK,
    env_cfg=unitree_b2_flat_env_cfg(),
    play_env_cfg=unitree_b2_flat_env_cfg(play=True),
    rl_cfg=unitree_b2_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
