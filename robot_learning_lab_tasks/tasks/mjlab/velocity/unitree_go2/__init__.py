"""Unitree Go2 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_go2.env_cfgs import (
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_go2.rl_cfg import (
    UNITREE_GO2_FLAT_TASK,
    UNITREE_GO2_ROUGH_TASK,
    unitree_go2_flat_ppo_runner_cfg,
    unitree_go2_rough_ppo_runner_cfg,
)

register_mjlab_task(
    task_id=UNITREE_GO2_ROUGH_TASK,
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id=UNITREE_GO2_FLAT_TASK,
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
