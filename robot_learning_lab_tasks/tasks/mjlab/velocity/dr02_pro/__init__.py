"""DR02 Pro velocity task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02_pro.env_cfgs import dr02_pro_rough_env_cfg
from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02_pro.rl_cfg import dr02_pro_rough_ppo_runner_cfg

DR02_PRO_ROUGH_TASK = "RobotLab-MJLab-Velocity-Rough-Deeprobotics-DR02-Pro"

register_mjlab_task(
    task_id=DR02_PRO_ROUGH_TASK,
    env_cfg=dr02_pro_rough_env_cfg(),
    play_env_cfg=dr02_pro_rough_env_cfg(play=True),
    rl_cfg=dr02_pro_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
