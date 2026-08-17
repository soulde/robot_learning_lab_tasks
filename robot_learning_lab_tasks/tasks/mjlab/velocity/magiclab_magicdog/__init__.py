from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import magicdog_flat_env_cfg, magicdog_rough_env_cfg
from .rl_cfg import (
    MAGICDOG_FLAT_TASK,
    MAGICDOG_ROUGH_TASK,
    magicdog_flat_ppo_runner_cfg,
    magicdog_rough_ppo_runner_cfg,
)

register_mjlab_task(
    MAGICDOG_ROUGH_TASK,
    magicdog_rough_env_cfg(),
    magicdog_rough_env_cfg(play=True),
    magicdog_rough_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
register_mjlab_task(
    MAGICDOG_FLAT_TASK,
    magicdog_flat_env_cfg(),
    magicdog_flat_env_cfg(play=True),
    magicdog_flat_ppo_runner_cfg(),
    VelocityOnPolicyRunner,
)
