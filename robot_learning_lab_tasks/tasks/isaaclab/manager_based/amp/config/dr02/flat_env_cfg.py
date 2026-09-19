"""Isaac Lab AMP environment configuration for Deeprobotics DR02 Pro."""

import os
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
from isaaclab_tasks.utils import PresetCfg

from robot_learning_lab_zoo.assets.isaaclab.deeprobotics import DEEPROBOTICS_DR02_PRO_CFG
from robot_learning_lab_tasks.motion_dataset import motion_data_root
from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.tracking_env_cfg import AMPEnvCfg

_ROBOT_DATA_ROOT = motion_data_root()


def dr02_amp_motion_dir() -> str:
    # Config and motion npz files live together in the robot data dir
    # (RLL_MOTION_DATA_DIR).
    return str(_ROBOT_DATA_ROOT)


def dr02_amp_body_names_path() -> str:
    return str(_ROBOT_DATA_ROOT / "bodies.json")


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
# AMP motion selection: every dataset take is included EXCEPT the entries
# below (verified by PD replay / physical checks on 2026-08-26). Add or
# remove a regex here to change the selection; the discriminator dataset and
# the RSI reset library both derive from this single list.
DR02_AMP_MOTION_EXCLUDES = (
    # absolute root height up to 1.7 m does not exist on the flat plane
    r"walking_upstairs\d+_stageii\.npz",
    # 4.4-4.7 m/s sprint takes, joint velocities up to 28 rad/s beyond motor
    # velocity limits (PD replay tracking error 3x worse than walking)
    r"run0[1-4]_stageii\.npz",
    # walking_run accelerations beyond the forward command cap (1.7-2.5 m/s)
    r"walking_run0[12]_stageii\.npz",
    # calibration debug take
    r"dr02_calibrated_debug_stageii\.npz",
    # circle walking is heavily over-represented (154 of 236 takes): keep
    # only take 01 of each circle kind
    r"\d+_WalkIn(?!.*01_stageii\.npz$).*\.npz",
    # soma-retargeted set (50 Hz, phase-resampled): stairs needs its own
    # terrain, run and the fast jog_ff family exceed the 2 m/s command cap
    r"stairs_climbing.*_phase\d+_30hz\.npz",
    r"run_.*_phase\d+_30hz\.npz",
    r"jog_ff_.*_phase\d+_30hz\.npz",
)


# RSI reset sampling windows per motion kind (see AmpMotionLibrary).
DR02_AMP_RESET_FRAMES: dict[str, int | None] = {
    "default": 25,
    # future recovery motions may sample every frame:
    # r"(fall|push)_recovery.*\.npz": None,
    # future stair motions only start from their first steps:
    # r"stairs.*\.npz": 5,
}


def dr02_amp_motion_files() -> list[str]:
    """Basenames of all dataset motions except the excluded takes."""
    import glob
    import re

    excludes = [re.compile(pattern) for pattern in DR02_AMP_MOTION_EXCLUDES]
    discovered = sorted(
        os.path.basename(path) for path in glob.glob(os.path.join(dr02_amp_motion_dir(), "*.npz"))
    )
    selected = [name for name in discovered if not any(p.fullmatch(name) for p in excludes)]
    if not selected:
        raise ValueError(f"DR02 AMP motion selection is empty in {dr02_amp_motion_dir()}")
    return selected


@configclass
class DR02AMPFlatPhysicsCfg(PresetCfg):
    """Physics backends for flat DR02 AMP; PhysX remains the default."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    default: PhysxCfg = isaacsim_physx
    newton_mjwarp: NewtonCfg = RoughPhysicsCfg().newton_mjwarp


@configclass
class DeeproboticsDR02ProAMPFlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = DR02AMPFlatPhysicsCfg()
        self.scene.robot = DEEPROBOTICS_DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # This is the simulator/action/policy order.  The AMP list below is
        # the motion-file order and may remain different; reset logic maps it
        # into the articulation order by joint name.
        sim_joint_names = list(DR02_JOINT_NAMES)
        self.actions.joint_pos.joint_names = sim_joint_names
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = sim_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = sim_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = sim_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = sim_joint_names
        self.events.randomize_com_positions.params["asset_cfg"].body_names = "base_link"
        self.actions.joint_pos.scale = 0.25
        self.observations.amp.joint_position.params["asset_cfg"].joint_names = list(DR02_JOINT_NAMES)
        self.observations.amp.joint_velocity.params["asset_cfg"].joint_names = list(DR02_JOINT_NAMES)
        self.observations.amp.link_positions.params["asset_cfg"].body_names = list(DR02_AMP_KEY_BODY_NAMES)
        # Reference state initialization: reset into expert motion frames,
        # matching the selection used by the AMP discriminator dataset.
        self.events.reset_amp_reference_state.params["motion_dir"] = dr02_amp_motion_dir()
        self.events.reset_amp_reference_state.params["motion_files"] = dr02_amp_motion_files()
        self.events.reset_amp_reference_state.params["joint_names"] = list(DR02_JOINT_NAMES)
        # RSI reset sampling follows the velocity command curriculum: jog and
        # fast motion pools join the resets as the speed range widens.
        self.events.reset_amp_reference_state.params["rsi_curriculum"] = True
        # Per-motion RSI sampling windows: entries are regex (on basename)
        # -> frame count, None/0 samples the whole motion; "default" applies
        # otherwise. Walking motions use the first 25 frames (0.5 s at
        # 50 fps) for clean initial postures.
        self.events.reset_amp_reference_state.params["reset_frames"] = DR02_AMP_RESET_FRAMES
        # Command ranges: 1.5x the chocolate rough curriculum limits
        # (x [-1.3, 2.4], y [-1.8, 1.8], yaw [-1, 1]); the command
        # curriculum below starts each run at a small slice of these.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.95, 3.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.8, 1.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # Infrequent pushes: 1-3 s pushes overwhelm unstable early gaits.
        self.events.randomize_push_robot.interval_range_s = (10.0, 15.0)
        # Shorter episodes let the terrain curriculum react before long
        # failed rollouts dominate it.
        # 2000 simulation steps at the 50 Hz environment step.
        self.episode_length_s = 40.0
        # No velocity command curriculum: the full command range is sampled
        # from the start, so disable both command curriculum terms.
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        # Trunk-contact-only termination: no bad_orientation, no root_height.
        self.terminations.root_height = None
        # Ignore brief upper-body contact transients; terminate after sustained
        # contact so the broadened pelvis/upper-body mask remains learnable.
        self.terminations.torso_contact.params["grace_time"] = 0.2
        # Rewards synced with the chocolate AMP (AmpPose variant) scales:
        # task tracking + RMS tracking penalties + termination/collision
        # floors + regularizers; pose and gait shaping stays with AMP.
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_rms.weight = -0.02
        self.rewards.track_ang_vel_z_rms.weight = -0.01
        # Falling must cost, otherwise the policy learns to terminate early
        # and dodge penalties (chocolate termination floor).
        self.rewards.is_alive.weight = 0.2
        self.rewards.is_terminated.weight = -300.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_torques_l2.weight = -1e-4
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_ankle_x_link$).*"]
