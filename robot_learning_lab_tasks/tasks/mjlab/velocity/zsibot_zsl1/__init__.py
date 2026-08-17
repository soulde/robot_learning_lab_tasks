from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import zsibot_zsl1_flat_env_cfg, zsibot_zsl1_rough_env_cfg
from .rl_cfg import (
    ZSIBOT_ZSL1_FLAT_TASK,
    ZSIBOT_ZSL1_ROUGH_TASK,
    zsibot_zsl1_flat_ppo_runner_cfg,
    zsibot_zsl1_rough_ppo_runner_cfg,
)

register_mjlab_task(
    ZSIBOT_ZSL1_ROUGH_TASK,
    zsibot_zsl1_rough_env_cfg(),
    zsibot_zsl1_rough_env_cfg(play=True),
    zsibot_zsl1_rough_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
register_mjlab_task(
    ZSIBOT_ZSL1_FLAT_TASK,
    zsibot_zsl1_flat_env_cfg(),
    zsibot_zsl1_flat_env_cfg(play=True),
    zsibot_zsl1_flat_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
