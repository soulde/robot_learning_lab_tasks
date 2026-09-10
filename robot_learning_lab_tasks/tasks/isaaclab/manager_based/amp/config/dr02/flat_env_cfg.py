"""Isaac Lab AMP environment configuration for Deeprobotics DR02 Pro."""

from isaaclab.utils import configclass

from robot_learning_lab_zoo.assets.isaaclab.deeprobotics import DEEPROBOTICS_DR02_PRO_CFG
from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.tracking_env_cfg import AMPEnvCfg

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
class DeeproboticsDR02ProAMPFlatEnvCfg(AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = DEEPROBOTICS_DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        # Asymmetric commands: x [-1, 2], y [-1, 1] m/s. Backward cap below
        # the forward cap; forward 2 m/s leaves headroom above the
        # walking_run takes (~1.0-1.1 m/s mean).
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        # Infrequent pushes: 1-3 s pushes overwhelm unstable early gaits.
        self.events.randomize_push_robot.interval_range_s = (10.0, 15.0)
        # Shorter episodes let the command/terrain curricula react before
        # long failed rollouts dominate them.
        self.episode_length_s = 10.0
        # Stronger illegal-contact penalty.
        self.rewards.undesired_contacts.weight = -5.0
        # Terminate when the torso tilts beyond 70 degrees (matches the
        # contact-based termination for unrecoverable falls).
        from isaaclab.envs.mdp.terminations import bad_orientation
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        self.terminations.bad_orientation = DoneTerm(
            func=bad_orientation, params={"limit_angle": 70.0 * 3.141592653589793 / 180.0}
        )
        # Trunk-contact-only termination (root_height term disabled); height
        # is shaped by reward instead of hard termination.
        self.terminations.root_height = None
        self.rewards.base_height.weight = -5.0
        self.rewards.base_height.params["target_height"] = 0.95
        self.rewards.is_terminated.weight = -10.0
        # episode length set above (10 s) for faster curriculum reaction
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.is_terminated.weight = -200.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_ankle_x_link$).*"]
