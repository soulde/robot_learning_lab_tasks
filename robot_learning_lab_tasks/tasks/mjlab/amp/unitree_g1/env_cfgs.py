"""Unitree G1 AMP environment configuration for MJLab."""

from dataclasses import dataclass
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from robot_learning_lab_zoo import ROBOTS_DIR

from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_g1.env_cfgs import (
    unitree_g1_dex3_backpack_flat_env_cfg,
    unitree_g1_dex3_flat_env_cfg,
    unitree_g1_flat_env_cfg,
)

from .. import mdp

G1_JOINT_NAMES = (
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
)

G1_AMP_LINK_NAMES = (
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
)


@dataclass(frozen=True)
class G1AmpCfg:
    dt: float
    amp_motion_weights: dict[str, float] | None
    joint_names: tuple[str, ...]
    amp_anchor_base: str
    amp_anchor_links: tuple[str, ...]
    urdf_path: str
    preload_transitions: bool = False
    amp_num_preload_transitions: int = 100_000


def _amp_observations() -> ObservationGroupCfg:
    robot = SceneEntityCfg("robot", joint_names=G1_JOINT_NAMES)
    links = SceneEntityCfg("robot", body_names=G1_AMP_LINK_NAMES)
    return ObservationGroupCfg(
        terms={
            "root_height": ObservationTermCfg(func=mdp.root_height),
            "root_orientation": ObservationTermCfg(func=mdp.root_orientation),
            "root_linear_velocity": ObservationTermCfg(func=mdp.root_linear_velocity),
            "root_angular_velocity": ObservationTermCfg(func=mdp.root_angular_velocity),
            "joint_position": ObservationTermCfg(func=mdp.joint_position, params={"asset_cfg": robot}),
            "joint_velocity": ObservationTermCfg(func=mdp.joint_velocity, params={"asset_cfg": robot}),
            "link_positions": ObservationTermCfg(func=mdp.link_positions, params={"asset_cfg": links}),
        },
        concatenate_terms=True,
        enable_corruption=False,
    )


def unitree_g1_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg.observations["amp"] = _amp_observations()
    cfg.episode_length_s = 20.0 if not play else int(1e9)
    cfg.amp = G1AmpCfg(
        dt=cfg.sim.mujoco.timestep * cfg.decimation,
        amp_motion_weights=None,
        joint_names=G1_JOINT_NAMES,
        amp_anchor_base="pelvis",
        amp_anchor_links=G1_AMP_LINK_NAMES,
        urdf_path=str(ROBOTS_DIR / "unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf"),
    )
    return cfg


def unitree_g1_dex3_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create G1 Dex3 AMP config with body-only demonstration features."""
    cfg = unitree_g1_dex3_flat_env_cfg(play=play)
    cfg.observations["amp"] = _amp_observations()
    cfg.episode_length_s = 20.0 if not play else int(1e9)
    cfg.amp = G1AmpCfg(
        dt=cfg.sim.mujoco.timestep * cfg.decimation,
        amp_motion_weights=None,
        joint_names=G1_JOINT_NAMES,
        amp_anchor_base="pelvis",
        amp_anchor_links=G1_AMP_LINK_NAMES,
        urdf_path=str(ROBOTS_DIR / "unitree/g1_description/urdf/g1_29dof_with_hand_rev_1_0.urdf"),
    )
    return cfg


def unitree_g1_dex3_backpack_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_dex3_backpack_flat_env_cfg(play=play)
    cfg.observations["amp"] = _amp_observations()
    cfg.episode_length_s = 20.0 if not play else int(1e9)
    cfg.amp = G1AmpCfg(
        dt=cfg.sim.mujoco.timestep * cfg.decimation,
        amp_motion_weights=None,
        joint_names=G1_JOINT_NAMES, amp_anchor_base="pelvis", amp_anchor_links=G1_AMP_LINK_NAMES,
        urdf_path=str(ROBOTS_DIR / "unitree/g1_description/urdf/g1_29dof_with_hand_backpack_1kg.urdf"),
    )
    return cfg
