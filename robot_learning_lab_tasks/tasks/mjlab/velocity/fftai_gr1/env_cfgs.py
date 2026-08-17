"""FFTAI GR1T1 and GR1T2 velocity environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from robot_learning_lab_zoo.assets.mjlab.fftai_gr1 import (
    FFTAI_GR1T1_CFG,
    FFTAI_GR1T2_CFG,
    GR1_ACTION_SCALE,
    GR1_FOOT_GEOM_NAMES,
    GR1_FOOT_SITE_NAMES,
)

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards as lab_rewards
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)


def _configure_gr1(cfg: ManagerBasedRlEnvCfg, *, t2: bool) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities["robot"] = deepcopy(FFTAI_GR1T2_CFG if t2 else FFTAI_GR1T1_CFG)
    cfg.sim.nconmax = 400
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "base"
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in GR1_FOOT_SITE_NAMES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in {"feet_ground_contact", "self_collision"}
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=GR1_FOOT_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = GR1_ACTION_SCALE
    action.clip = {".*": (-100.0, 100.0)}
    cfg.viewer.body_name = "base"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = GR1_FOOT_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_angular_velocity"].weight = 5.0
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base",)
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(".*_ankle_.*",))
    cfg.rewards["action_rate_l2"].weight = -0.005
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["air_time"].params["threshold_max"] = 0.6
    cfg.rewards["foot_slip"].weight = -0.2
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.3}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.5}
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = GR1_FOOT_SITE_NAMES
    for name in ("joint_deviation_hip_l1", "joint_deviation_arms_l1", "joint_deviation_torso_l1"):
        cfg.rewards.pop(name, None)
    cfg.rewards["joint_deviation_other_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(".*head_.*", ".*_hip_yaw", ".*_hip_roll", ".*_shoulder_.*", ".*_wrist_.*"),
            )
        },
    )
    cfg.rewards["joint_deviation_torso_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.2 if t2 else -0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*waist_.*",))},
    )
    cfg.rewards["joint_deviation_elbow_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_elbow_pitch",))},
    )
    cfg.rewards["joint_deviation_knee_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_pitch",))},
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-1.0, 1.0)
    twist.ranges.lin_vel_y = (-1.0, 1.0)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    return cfg


def gr1t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure_gr1(unitree_g1_rough_env_cfg(play=play), t2=False)


def gr1t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure_gr1(unitree_g1_flat_env_cfg(play=play), t2=False)


def gr1t2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure_gr1(unitree_g1_rough_env_cfg(play=play), t2=True)


def gr1t2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure_gr1(unitree_g1_flat_env_cfg(play=play), t2=True)
