"""Unitree G1 AMP task registration."""

from mjlab.tasks.registry import register_mjlab_task
from rll_rl import AMPRunner

from .env_cfgs import unitree_g1_amp_flat_env_cfg, unitree_g1_dex3_amp_flat_env_cfg
from .rl_cfg import unitree_g1_amp_runner_cfg

register_mjlab_task(
    task_id="RobotLab-MJLab-AMP-Flat-Unitree-G1",
    env_cfg=unitree_g1_amp_flat_env_cfg(),
    play_env_cfg=unitree_g1_amp_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_amp_runner_cfg(),
    runner_cls=AMPRunner,
)

register_mjlab_task(
    task_id="RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3",
    env_cfg=unitree_g1_dex3_amp_flat_env_cfg(),
    play_env_cfg=unitree_g1_dex3_amp_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_amp_runner_cfg(),
    runner_cls=AMPRunner,
)
