"""Unitree H1 velocity environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from robot_learning_lab_zoo.assets.mjlab.unitree import UNITREE_H1_CFG

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards

BASE_BODY = "torso_link"
FOOT_BODIES = ("Link_ankle_l_roll", "Link_ankle_r_roll")
FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def unitree_h1_actions_cfg():
    return {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=0.25,
        clip={".*": (-100.0, 100.0)}, use_default_offset=True, preserve_order=True)}


def unitree_h1_rewards_cfg():
    return {
        "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=3.0,
            params={"command_name": "twist", "std": 0.5}),
        "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=3.0,
            params={"command_name": "twist", "std": 0.5}),
        "body_ang_vel": RewardTermCfg(func=mdp.body_angular_velocity_penalty, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,))}),
        "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-0.5),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
        "air_time": RewardTermCfg(func=mdp.feet_air_time, weight=1.0,
            params={"sensor_name": "feet_ground_contact", "threshold_min": 0.05, "threshold_max": 0.4,
                    "command_name": "twist", "command_threshold": 0.5}),
        "foot_slip": RewardTermCfg(func=mdp.feet_slip, weight=-0.2,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.05,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)}),
        "joint_deviation_hip_l1": RewardTermCfg(func=rewards.joint_deviation_l1, weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"))}),
        "joint_deviation_arms_l1": RewardTermCfg(func=rewards.joint_deviation_l1, weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_shoulder_.*_joint", ".*_elbow_joint"))}),
        "joint_deviation_torso_l1": RewardTermCfg(func=rewards.joint_deviation_l1, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=("torso_joint",))}),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    }


def unitree_h1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500; cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 500; cfg.sim.njmax = 1000
    cfg.scene.entities = {"robot": deepcopy(UNITREE_H1_CFG)}
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = BASE_BODY
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in FOOT_SITES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
    feet_contact = ContactSensorCfg(name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
        reduce="netforce", num_slots=1, track_air_time=True)
    self_collision = ContactSensorCfg(name="self_collision",
        primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact, self_collision)
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.actions = unitree_h1_actions_cfg(); cfg.rewards = unitree_h1_rewards_cfg()
    cfg.viewer.body_name = BASE_BODY
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)
    twist = cfg.commands["twist"]; assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-1.0, 1.0); twist.ranges.lin_vel_y = (-1.0, 1.0); twist.ranges.ang_vel_z = (-1.0, 1.0)
    if play:
        cfg.episode_length_s = int(1e9); cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None); cfg.terminations.pop("out_of_terrain_bounds", None); cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})
    return cfg


def unitree_h1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = unitree_h1_rough_env_cfg(play=play); assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None); cfg.curriculum.pop("terrain_levels", None)
    return cfg
