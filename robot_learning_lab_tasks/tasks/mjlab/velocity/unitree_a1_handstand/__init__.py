"""Unitree A1 handstand task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_a1_handstand.env_cfgs import (
    unitree_a1_handstand_flat_env_cfg,
    unitree_a1_handstand_rough_env_cfg,
)
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_a1_handstand.rl_cfg import (
    UNITREE_A1_HANDSTAND_FLAT_TASK,
    UNITREE_A1_HANDSTAND_ROUGH_TASK,
    unitree_a1_handstand_flat_ppo_runner_cfg,
    unitree_a1_handstand_rough_ppo_runner_cfg,
)

register_mjlab_task(
    task_id=UNITREE_A1_HANDSTAND_ROUGH_TASK,
    env_cfg=unitree_a1_handstand_rough_env_cfg(),
    play_env_cfg=unitree_a1_handstand_rough_env_cfg(play=True),
    rl_cfg=unitree_a1_handstand_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id=UNITREE_A1_HANDSTAND_FLAT_TASK,
    env_cfg=unitree_a1_handstand_flat_env_cfg(),
    play_env_cfg=unitree_a1_handstand_flat_env_cfg(play=True),
    rl_cfg=unitree_a1_handstand_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
