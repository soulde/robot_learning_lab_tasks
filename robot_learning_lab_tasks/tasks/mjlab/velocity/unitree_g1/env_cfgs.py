"""Unitree G1 velocity environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from robot_learning_lab_zoo.assets.mjlab.unitree import (
    G1_ACTION_SCALE,
    G1_DEX3_ACTION_SCALE,
    G1_FOOT_GEOM_NAMES,
    G1_FOOT_SITE_NAMES,
    UNITREE_G1_29DOF_CFG,
    UNITREE_G1_29DOF_DEX3_BACKPACK_CFG,
    UNITREE_G1_29DOF_DEX3_CFG,
)

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards as lab_rewards


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 70

    cfg.scene.entities = {"robot": deepcopy(UNITREE_G1_29DOF_CFG)}

    # Set raycast sensor frame to G1 pelvis.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "pelvis"

    site_names = G1_FOOT_SITE_NAMES
    geom_names = G1_FOOT_GEOM_NAMES

    # Wire foot height scan to per-foot sites.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE
    joint_pos_action.clip = {".*": (-100.0, 100.0)}

    cfg.viewer.body_name = "pelvis"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)

    # Rationale for std values:
    # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
    # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
    # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
    # - Waist roll/pitch stay tight to keep the torso upright and stable.
    # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
    # - Wrists are loose (0.3) since they don't affect balance much.
    # Running values are ~1.5-2x walking values to accommodate larger motion range.
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist.
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,
        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        # Waist.
        r".*waist_yaw.*": 0.3,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.2,
        # Arms.
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["dof_pos_limits"].weight = -0.5
    cfg.rewards["joint_deviation_hip_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*hip_yaw.*", ".*hip_roll.*"))},
    )
    cfg.rewards["joint_deviation_arms_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*shoulder.*", ".*elbow.*"))},
    )
    cfg.rewards["joint_deviation_torso_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
    )
    cfg.rewards["joint_pos_penalty"] = RewardTermCfg(
        func=lab_rewards.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    cfg.rewards["upward"] = RewardTermCfg(
        func=lab_rewards.upward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["action_rate_l2"].weight = -0.005
    cfg.rewards["air_time"].weight = 0.25
    cfg.rewards["air_time"].params["threshold_max"] = 0.4

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5
            cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain velocity configuration."""
    cfg = unitree_g1_rough_env_cfg(play=play)

    cfg.sim.njmax = 500
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.terminations.pop("out_of_terrain_bounds", None)

    # Disable terrain curriculum (not present in play mode since rough clears all).
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg


def _add_dex3_pose_std(cfg: ManagerBasedRlEnvCfg) -> None:
    """Cover the Dex3 hand joints missed by the G1 pose std patterns."""
    for key, std in (("std_walking", 0.3), ("std_running", 0.3)):
        params = cfg.rewards["pose"].params[key]
        params[r".*_hand_.*_joint"] = std


def unitree_g1_dex3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the G1 Dex3 rough-terrain velocity configuration."""
    cfg = unitree_g1_rough_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(UNITREE_G1_29DOF_DEX3_CFG)
    cfg.actions["joint_pos"].scale = G1_DEX3_ACTION_SCALE
    _add_dex3_pose_std(cfg)
    return cfg


def unitree_g1_dex3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the G1 Dex3 flat-terrain velocity configuration."""
    cfg = unitree_g1_flat_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(UNITREE_G1_29DOF_DEX3_CFG)
    cfg.actions["joint_pos"].scale = G1_DEX3_ACTION_SCALE
    _add_dex3_pose_std(cfg)
    return cfg


def unitree_g1_dex3_backpack_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_dex3_rough_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(UNITREE_G1_29DOF_DEX3_BACKPACK_CFG)
    return cfg


def unitree_g1_dex3_backpack_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_g1_dex3_flat_env_cfg(play=play)
    cfg.scene.entities["robot"] = deepcopy(UNITREE_G1_29DOF_DEX3_BACKPACK_CFG)
    return cfg
