"""FFTAI GR1 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import gr1t1_flat_env_cfg, gr1t1_rough_env_cfg, gr1t2_flat_env_cfg, gr1t2_rough_env_cfg
from .rl_cfg import gr1_ppo_runner_cfg

for task_id, env_fn, model, flat in (
    ("RobotLab-MJLab-Velocity-Rough-FFTAI-GR1T1", gr1t1_rough_env_cfg, "gr1t1", False),
    ("RobotLab-MJLab-Velocity-Flat-FFTAI-GR1T1", gr1t1_flat_env_cfg, "gr1t1", True),
    ("RobotLab-MJLab-Velocity-Rough-FFTAI-GR1T2", gr1t2_rough_env_cfg, "gr1t2", False),
    ("RobotLab-MJLab-Velocity-Flat-FFTAI-GR1T2", gr1t2_flat_env_cfg, "gr1t2", True),
):
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_fn(),
        play_env_cfg=env_fn(play=True),
        rl_cfg=gr1_ppo_runner_cfg(model, flat=flat),
        runner_cls=VelocityOnPolicyRunner,
    )
