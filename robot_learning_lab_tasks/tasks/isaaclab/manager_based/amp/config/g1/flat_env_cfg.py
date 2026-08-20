# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from robot_learning_lab_zoo.assets.isaaclab.unitree import (
    UNITREE_G1_29DOF_ACTION_SCALE,
    UNITREE_G1_29DOF_CFG,
    UNITREE_G1_29DOF_DEX3_ACTION_SCALE,
    UNITREE_G1_29DOF_DEX3_BACKPACK_ACTION_SCALE,
    UNITREE_G1_29DOF_DEX3_BACKPACK_CFG,
    UNITREE_G1_29DOF_DEX3_CFG,
)

from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.tracking_env_cfg import AMPEnvCfg

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1_AMP_LINK_NAMES = [
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


@configclass
class UnitreeG1AMPFlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.observations.amp.joint_position.params["asset_cfg"].joint_names = G1_JOINT_NAMES
        self.observations.amp.joint_velocity.params["asset_cfg"].joint_names = G1_JOINT_NAMES
        self.observations.amp.link_positions.params["asset_cfg"].body_names = G1_AMP_LINK_NAMES

        self.episode_length_s = 30.0


@configclass
class UnitreeG1Dex3AMPFlatEnvCfg(UnitreeG1AMPFlatEnvCfg):
    """G1 Dex3 AMP environment with body-only reference observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_DEX3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_DEX3_ACTION_SCALE


@configclass
class UnitreeG1Dex3BackpackAMPFlatEnvCfg(UnitreeG1Dex3AMPFlatEnvCfg):
    """G1 Dex3 AMP environment with a fixed 1 kg backpack."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_DEX3_BACKPACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_DEX3_BACKPACK_ACTION_SCALE
