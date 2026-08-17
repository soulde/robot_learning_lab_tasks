"""DR02 Pro velocity environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from robot_learning_lab_zoo.assets.mjlab.deeprobotics_dr02 import DEEPROBOTICS_DR02_PRO_CFG

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards as lab_rewards
from robot_learning_lab_tasks.tasks.mjlab.velocity.dr02.env_cfgs import dr02_rough_env_cfg


def dr02_pro_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the DR02 Pro rough terrain velocity configuration."""
    cfg = dr02_rough_env_cfg(play=play)
    cfg.sim.nconmax = 500
    cfg.scene.entities["robot"] = deepcopy(DEEPROBOTICS_DR02_PRO_CFG)
    cfg.rewards["joint_deviation_torso_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=("waist_z_joint", "waist_x_joint", "waist_y_joint")
            )
        },
    )
    # The converted upstream model has fixed neck links, so there are no neck
    # joints on which to apply Isaac Lab's optional head-deviation reward.
    return cfg
