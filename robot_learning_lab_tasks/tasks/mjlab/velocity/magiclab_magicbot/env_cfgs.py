"""MagicLab MagicBot velocity environment configurations."""

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
from robot_learning_lab_zoo.assets.mjlab.magiclab_magicbot import (
    MAGICBOT_ACTION_JOINT_NAMES,
    MAGICBOT_FOOT_GEOM_NAMES,
    MAGICBOT_FOOT_SITE_NAMES,
    MAGICLAB_BOT_GEN1_CFG,
    MAGICLAB_BOT_Z1_CFG,
)

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards as lab_rewards
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)


def _configure(cfg: ManagerBasedRlEnvCfg, *, z1: bool) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities["robot"] = deepcopy(MAGICLAB_BOT_Z1_CFG if z1 else MAGICLAB_BOT_GEN1_CFG)
    cfg.sim.nconmax = 300
    actor = cfg.observations["actor"]
    actor.terms.pop("base_lin_vel", None)
    actor.terms.pop("height_scan", None)
    joint_cfg = SceneEntityCfg("robot", joint_names=MAGICBOT_ACTION_JOINT_NAMES)
    actor.terms["joint_pos"].params["asset_cfg"] = joint_cfg
    actor.terms["joint_vel"].params["asset_cfg"] = joint_cfg
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "pelvis"
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in MAGICBOT_FOOT_SITE_NAMES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in {"feet_ground_contact", "self_collision"}
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=MAGICBOT_FOOT_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.joint_names = MAGICBOT_ACTION_JOINT_NAMES
    action.scale = 0.25
    action.clip = {".*": (-100.0, 100.0)}
    cfg.viewer.body_name = "pelvis"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = MAGICBOT_FOOT_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["dof_pos_limits"].weight = -0.5
    cfg.rewards["action_rate_l2"].weight = -0.005
    cfg.rewards["air_time"].weight = 0.25
    cfg.rewards["air_time"].params["threshold_max"] = 0.4
    cfg.rewards["foot_slip"].weight = -0.2
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.3}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.5}
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = MAGICBOT_FOOT_SITE_NAMES
    for name in ("joint_deviation_hip_l1", "joint_deviation_arms_l1", "joint_deviation_torso_l1"):
        cfg.rewards.pop(name, None)
    cfg.rewards["joint_deviation_hip_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*HIP_YAW.*", ".*HIP_ROLL.*"))},
    )
    cfg.rewards["joint_deviation_arms_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*a1",))},
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (0.0, 1.0) if z1 else (-1.0, 1.0)
    twist.ranges.lin_vel_y = (0.0, 0.0) if z1 else (-1.0, 1.0)
    twist.ranges.ang_vel_z = (0.0, 0.0) if z1 else (-1.0, 1.0)
    return cfg


def gen1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure(unitree_g1_rough_env_cfg(play=play), z1=False)


def gen1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure(unitree_g1_flat_env_cfg(play=play), z1=False)


def z1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure(unitree_g1_rough_env_cfg(play=play), z1=True)


def z1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return _configure(unitree_g1_flat_env_cfg(play=play), z1=True)
