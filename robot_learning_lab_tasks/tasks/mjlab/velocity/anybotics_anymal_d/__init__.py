from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import anymal_d_flat_env_cfg, anymal_d_rough_env_cfg
from .rl_cfg import (
    ANYMAL_D_FLAT_TASK,
    ANYMAL_D_ROUGH_TASK,
    anymal_d_flat_ppo_runner_cfg,
    anymal_d_rough_ppo_runner_cfg,
)

register_mjlab_task(
    ANYMAL_D_ROUGH_TASK,
    anymal_d_rough_env_cfg(),
    anymal_d_rough_env_cfg(play=True),
    anymal_d_rough_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
register_mjlab_task(
    ANYMAL_D_FLAT_TASK,
    anymal_d_flat_env_cfg(),
    anymal_d_flat_env_cfg(play=True),
    anymal_d_flat_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
