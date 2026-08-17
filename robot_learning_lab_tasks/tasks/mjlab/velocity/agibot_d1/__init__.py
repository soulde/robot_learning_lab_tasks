from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import agibot_d1_flat_env_cfg, agibot_d1_rough_env_cfg
from .rl_cfg import (
    AGIBOT_D1_FLAT_TASK,
    AGIBOT_D1_ROUGH_TASK,
    agibot_d1_flat_ppo_runner_cfg,
    agibot_d1_rough_ppo_runner_cfg,
)

register_mjlab_task(
    AGIBOT_D1_ROUGH_TASK,
    agibot_d1_rough_env_cfg(),
    agibot_d1_rough_env_cfg(play=True),
    agibot_d1_rough_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
register_mjlab_task(
    AGIBOT_D1_FLAT_TASK,
    agibot_d1_flat_env_cfg(),
    agibot_d1_flat_env_cfg(play=True),
    agibot_d1_flat_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
