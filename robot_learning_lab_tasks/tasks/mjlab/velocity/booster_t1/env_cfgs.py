"""Booster T1 velocity environment configurations."""

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
from robot_learning_lab_zoo.assets.mjlab.booster_t1 import (
    BOOSTER_T1_ACTION_SCALE,
    BOOSTER_T1_CFG,
    BOOSTER_T1_FOOT_GEOM_NAMES,
    BOOSTER_T1_FOOT_SITE_NAMES,
)

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards as lab_rewards
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_g1.env_cfgs import (
    unitree_g1_flat_env_cfg,
    unitree_g1_rough_env_cfg,
)


def _configure_t1(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    cfg.scene.entities["robot"] = deepcopy(BOOSTER_T1_CFG)
    cfg.sim.nconmax = 300
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "Trunk"
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in BOOSTER_T1_FOOT_SITE_NAMES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name not in {"feet_ground_contact", "self_collision"}
    ) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=BOOSTER_T1_FOOT_GEOM_NAMES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = BOOSTER_T1_ACTION_SCALE
    action.clip = {".*": (-100.0, 100.0)}
    cfg.viewer.body_name = "Trunk"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = BOOSTER_T1_FOOT_GEOM_NAMES
    cfg.events["base_com"].params["asset_cfg"].body_names = ("Trunk",)

    cfg.rewards["track_linear_velocity"].weight = 4.5
    cfg.rewards["track_angular_velocity"].weight = 2.5
    cfg.rewards["body_ang_vel"].weight = -0.1
    cfg.rewards["dof_pos_limits"].weight = -1.0
    cfg.rewards["action_rate_l2"].weight = -0.075
    cfg.rewards["air_time"].weight = 2.0
    cfg.rewards["air_time"].params["threshold_max"] = 0.4
    cfg.rewards["foot_slip"].weight = -0.4
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("Trunk",)
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {".*": 0.3}
    cfg.rewards["pose"].params["std_running"] = {".*": 0.5}
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = BOOSTER_T1_FOOT_SITE_NAMES
    cfg.rewards.pop("joint_deviation_hip_l1", None)
    cfg.rewards.pop("joint_deviation_arms_l1", None)
    cfg.rewards.pop("joint_deviation_torso_l1", None)
    cfg.rewards["joint_deviation_hip_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_Hip_Yaw", ".*_Hip_Roll"))},
    )
    cfg.rewards["joint_deviation_arms_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_Shoulder_.*", ".*_Elbow_.*"))},
    )
    cfg.rewards["joint_deviation_torso_l1"] = RewardTermCfg(
        func=lab_rewards.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("Waist",))},
    )
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-1.0, 1.0)
    twist.ranges.lin_vel_y = (-1.0, 1.0)
    twist.ranges.ang_vel_z = (-1.0, 1.0)
    return cfg


def booster_t1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 rough terrain velocity configuration."""
    return _configure_t1(unitree_g1_rough_env_cfg(play=play))


def booster_t1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Booster T1 flat terrain velocity configuration."""
    cfg = _configure_t1(unitree_g1_flat_env_cfg(play=play))
    cfg.rewards["lin_vel_z_l2"] = RewardTermCfg(
        func=lab_rewards.lin_vel_z_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg
