"""DeepRobotics Lite3 velocity environment configurations."""

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
from robot_learning_lab_zoo.assets.mjlab.deeprobotics_lite3 import (
    DEEPROBOTICS_LITE3_ACTION_SCALE, DEEPROBOTICS_LITE3_CFG, DEEPROBOTICS_LITE3_FOOT_BODY_NAMES,
    DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES, DEEPROBOTICS_LITE3_FOOT_SITE_NAMES, DEEPROBOTICS_LITE3_JOINT_NAMES,
)

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards

BASE_BODY = "TORSO"
FOOT_BODIES = DEEPROBOTICS_LITE3_FOOT_BODY_NAMES
FOOT_SITES = DEEPROBOTICS_LITE3_FOOT_SITE_NAMES
FOOT_GEOMS = DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES


def deeprobotics_lite3_actions_cfg():
    return {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=DEEPROBOTICS_LITE3_JOINT_NAMES,
        scale=DEEPROBOTICS_LITE3_ACTION_SCALE,
        clip={".*": (-100.0, 100.0)}, use_default_offset=True, preserve_order=True)}


def deeprobotics_lite3_rewards_cfg():
    return {
        "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=3.0,
            params={"command_name": "twist", "std": 0.5}),
        "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=1.5,
            params={"command_name": "twist", "std": 0.5}),
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "body_ang_vel": RewardTermCfg(func=mdp.body_angular_velocity_penalty, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,))}),
        "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "air_time": RewardTermCfg(func=mdp.feet_air_time, weight=0.0,
            params={"sensor_name": "feet_ground_contact", "threshold_min": 0.05, "threshold_max": 0.5,
                    "command_name": "twist", "command_threshold": 0.5}),
        "foot_slip": RewardTermCfg(func=mdp.feet_slip, weight=0.0,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.05,
                    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)}),
        "stand_still": RewardTermCfg(func=rewards.stand_still, weight=-2.0,
            params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", joint_names=DEEPROBOTICS_LITE3_JOINT_NAMES)}),
        "joint_pos_penalty": RewardTermCfg(func=rewards.joint_pos_penalty, weight=-1.0,
            params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", joint_names=DEEPROBOTICS_LITE3_JOINT_NAMES),
                    "stand_still_scale": 5.0, "velocity_threshold": 0.5, "command_threshold": 0.1}),
        "joint_mirror": RewardTermCfg(func=rewards.joint_mirror, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DEEPROBOTICS_LITE3_JOINT_NAMES),
                    "mirror_joints": [["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
                                      ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"]]}),
        "feet_contact_without_cmd": RewardTermCfg(func=rewards.feet_contact_without_cmd, weight=0.1,
            params={"command_name": "twist", "sensor_name": "feet_ground_contact"}),
        "feet_height_body": RewardTermCfg(func=rewards.feet_height_body, weight=0.0,
            params={"command_name": "twist", "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
                    "target_height": -0.2, "tanh_mult": 2.0}),
        "upward": RewardTermCfg(func=rewards.upward, weight=1.0),
        "feet_air_time_variance": RewardTermCfg(func=rewards.feet_air_time_variance, weight=0.0,
            params={"sensor_name": "feet_ground_contact"}),
        "feet_gait": RewardTermCfg(func=rewards.feet_gait, weight=0.0,
            params={"std": 0.7071067811865476, "command_name": "twist", "max_err": 0.2,
                    "velocity_threshold": 0.5, "command_threshold": 0.1,
                    "synced_feet_pair_names": (("FL_FOOT_collision", "HR_FOOT_collision"),
                                                ("FR_FOOT_collision", "HL_FOOT_collision")),
                    "sensor_name": "feet_ground_contact"}),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    }


def deeprobotics_lite3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500; cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 500; cfg.sim.njmax = 1000
    cfg.scene.entities = {"robot": deepcopy(DEEPROBOTICS_LITE3_CFG)}
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
    nonfoot_contact = ContactSensorCfg(name="nonfoot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=".*_collision.*", entity="robot", exclude=FOOT_GEOMS),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
        reduce="none", num_slots=1, history_length=4)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact, self_collision, nonfoot_contact)
    cfg.rewards = deeprobotics_lite3_rewards_cfg()
    cfg.rewards["undesired_contacts"] = RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 1.0})
    for group in cfg.observations.values():
        if "base_lin_vel" in group.terms:
            group.terms["base_lin_vel"].func = envs_mdp.base_lin_vel
            group.terms["base_lin_vel"].params = {"asset_cfg": SceneEntityCfg("robot")}
        group.terms["base_ang_vel"].func = envs_mdp.base_ang_vel
        group.terms["base_ang_vel"].params = {"asset_cfg": SceneEntityCfg("robot")}
        group.terms["projected_gravity"].func = envs_mdp.projected_gravity
        group.terms["projected_gravity"].params = {"asset_cfg": SceneEntityCfg("robot")}
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.actions = deeprobotics_lite3_actions_cfg()
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


def deeprobotics_lite3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = deeprobotics_lite3_rough_env_cfg(play=play); assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None); cfg.curriculum.pop("terrain_levels", None)
    return cfg
