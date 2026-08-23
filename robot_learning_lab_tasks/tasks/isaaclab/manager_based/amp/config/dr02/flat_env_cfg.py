"""Isaac Lab AMP environment configuration for Deeprobotics DR02 Pro."""

import os
from pathlib import Path

from isaaclab.utils import configclass

from robot_learning_lab_zoo.assets.isaaclab.deeprobotics import DEEPROBOTICS_DR02_PRO_CFG
from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.tracking_env_cfg import AMPEnvCfg


_ROBOT_DATA_ROOT = Path(os.environ.get("GMR_PRIVATE_DIR", Path.home() / "GMR-private")) / "retarget_data" / "dr02"
DR02_JOINT_NAMES = (
    "waist_z_joint", "waist_x_joint", "waist_y_joint",
    "left_shoulder_y_joint", "left_shoulder_x_joint", "left_shoulder_z_joint", "left_elbow_joint",
    "left_wrist_z_joint", "left_wrist_y_joint", "left_wrist_x_joint",
    "right_shoulder_y_joint", "right_shoulder_x_joint", "right_shoulder_z_joint", "right_elbow_joint",
    "right_wrist_z_joint", "right_wrist_y_joint", "right_wrist_x_joint",
    "left_hip_y_joint", "left_hip_x_joint", "left_hip_z_joint", "left_knee_joint",
    "left_ankle_y_joint", "left_ankle_x_joint", "right_hip_y_joint", "right_hip_x_joint",
    "right_hip_z_joint", "right_knee_joint", "right_ankle_y_joint", "right_ankle_x_joint",
)
DR02_AMP_BODY_NAMES = (
    "base_link", "waist_z_link", "waist_x_link", "body",
    "left_shoulder_y_link", "left_shoulder_x_link", "left_shoulder_z_link", "left_elbow_link",
    "left_wrist_z_link", "left_wrist_y_link", "left_wrist_x_link",
    "right_shoulder_y_link", "right_shoulder_x_link", "right_shoulder_z_link", "right_elbow_link",
    "right_wrist_z_link", "right_wrist_y_link", "right_wrist_x_link",
    "left_hip_y_link", "left_hip_x_link", "left_hip_z_link", "left_knee_link",
    "left_ankle_y_link", "left_ankle_x_link", "right_hip_y_link", "right_hip_x_link",
    "right_hip_z_link", "right_knee_link", "right_ankle_y_link", "right_ankle_x_link",
)
DR02_AMP_KEY_BODY_NAMES = (
    "base_link", "left_hip_x_link", "left_knee_link", "left_ankle_x_link",
    "right_hip_x_link", "right_knee_link", "right_ankle_x_link", "body",
    "left_shoulder_x_link", "left_elbow_link", "left_wrist_x_link",
    "right_shoulder_x_link", "right_elbow_link", "right_wrist_x_link",
)


def dr02_amp_motion_dir() -> str:
    return str(_ROBOT_DATA_ROOT / "datasets")


def dr02_amp_body_names_path() -> str:
    return str(_ROBOT_DATA_ROOT / "bodies.json")


@configclass
class DeeproboticsDR02ProAMPFlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DEEPROBOTICS_DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.events.randomize_com_positions.params["asset_cfg"].body_names = "base_link"
        self.actions.joint_pos.scale = 0.25
        self.observations.amp.joint_position.params["asset_cfg"].joint_names = list(DR02_JOINT_NAMES)
        self.observations.amp.joint_velocity.params["asset_cfg"].joint_names = list(DR02_JOINT_NAMES)
        self.observations.amp.link_positions.params["asset_cfg"].body_names = list(DR02_AMP_KEY_BODY_NAMES)
        self.episode_length_s = 20.0
