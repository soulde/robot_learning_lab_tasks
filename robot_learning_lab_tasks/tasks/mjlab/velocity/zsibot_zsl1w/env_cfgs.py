"""Zsibot ZSL1W velocity environment configurations."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RayCastSensorCfg, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from robot_learning_lab_zoo.assets.mjlab.zsibot import ZSIBOT_ZSL1W_CFG, ZSL1W_WHEEL_BODIES

from robot_learning_lab_tasks.tasks.mjlab.velocity import observations, rewards

LEGS = ("FAR", "FBL", "RAR", "RBL")
LEG_JOINTS = tuple(f"{leg}_{joint}_JOINT" for leg in LEGS for joint in ("ABAD", "HIP", "KNEE"))
WHEEL_JOINTS = tuple(f"{leg}_FOOT_JOINT" for leg in LEGS)
ALL_JOINTS = LEG_JOINTS + WHEEL_JOINTS
WHEEL_SITES = tuple(f"{name}_site" for name in ZSL1W_WHEEL_BODIES)
WHEEL_GEOMS = tuple(f"{name}_collision" for name in ZSL1W_WHEEL_BODIES)


def zsibot_zsl1w_actions_cfg():
    """Build the M20 policy action mapping."""
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=LEG_JOINTS,
            scale={".*_ABAD_JOINT": 0.125, "^(?!.*_ABAD_JOINT).*": 0.25},
            clip={".*": (-100.0, 100.0)}, use_default_offset=True, preserve_order=True,
        ),
        "joint_vel": JointVelocityActionCfg(
            entity_name="robot", actuator_names=WHEEL_JOINTS, scale=5.0,
            clip={".*": (-100.0, 100.0)}, use_default_offset=True, preserve_order=True,
        ),
    }


def zsibot_zsl1w_rewards_cfg():
    """Build the complete M20 reward configuration."""
    leg_cfg = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
    return {
        "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=3.0,
            params={"command_name": "twist", "std": 0.5}),
        "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=1.5,
            params={"command_name": "twist", "std": 0.5}),
        "lin_vel_z_l2": RewardTermCfg(func=rewards.lin_vel_z_l2, weight=-2.0),
        "body_ang_vel": RewardTermCfg(func=mdp.body_angular_velocity_penalty, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("BASE_LINK",))}),
        "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0,
            params={"asset_cfg": leg_cfg}),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
        "stand_still": RewardTermCfg(func=rewards.stand_still, weight=-2.0,
            params={"command_name": "twist", "asset_cfg": leg_cfg}),
        "joint_pos_penalty": RewardTermCfg(func=rewards.joint_pos_penalty, weight=-1.0,
            params={"command_name": "twist", "asset_cfg": leg_cfg, "stand_still_scale": 5.0,
                    "velocity_threshold": 0.5, "command_threshold": 0.1}),
        "upward": RewardTermCfg(func=rewards.upward, weight=1.0),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0}),
        "undesired_contacts": RewardTermCfg(func=mdp.self_collision_cost, weight=-1.0,
            params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 1.0}),
    }


def zsibot_zsl1w_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.njmax = 2500
    cfg.sim.nconmax = 1000
    cfg.scene.entities = {"robot": deepcopy(ZSIBOT_ZSL1W_CFG)}

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg) and isinstance(sensor.frame, ObjRef)
            sensor.frame.name = "BASE_LINK"
        elif sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in WHEEL_SITES)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_contact = ContactSensorCfg(
        name="feet_ground_contact", primary=ContactMatch(mode="geom", pattern=WHEEL_GEOMS, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
        reduce="netforce", num_slots=1, track_air_time=True, history_length=4,
    )
    self_collision = ContactSensorCfg(
        name="self_collision", primary=ContactMatch(mode="subtree", pattern="BASE_LINK", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="BASE_LINK", entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1, history_length=4,
    )
    nonfoot_contact = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(mode="geom", pattern=".*_collision.*", entity="robot", exclude=WHEEL_GEOMS),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
        reduce="none", num_slots=1, history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_contact, self_collision, nonfoot_contact)

    all_cfg = SceneEntityCfg("robot", joint_names=ALL_JOINTS)
    wheel_cfg = SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)
    for group in cfg.observations.values():
        group.terms["joint_pos"].func = observations.joint_pos_rel_without_wheel
        group.terms["joint_pos"].params = {"asset_cfg": all_cfg, "wheel_asset_cfg": wheel_cfg}
        group.terms["joint_vel"].params = {"asset_cfg": all_cfg}
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)

    cfg.actions = zsibot_zsl1w_actions_cfg()
    cfg.viewer.body_name = "BASE_LINK"
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = WHEEL_GEOMS
    cfg.events["base_com"].params["asset_cfg"].body_names = ("BASE_LINK",)

    cfg.rewards = zsibot_zsl1w_rewards_cfg()
    cfg.terminations.pop("illegal_contact", None)
    cfg.curriculum.pop("command_vel", None)
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_x = (-1.0, 1.0); twist.ranges.lin_vel_y = (-1.0, 1.0); twist.ranges.ang_vel_z = (-1.0, 1.0)

    if play:
        cfg.episode_length_s = int(1e9); cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None); cfg.terminations.pop("out_of_terrain_bounds", None); cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(func=envs_mdp.randomize_terrain, mode="reset", params={})
    return cfg


def zsibot_zsl1w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = zsibot_zsl1w_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"; cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None); cfg.curriculum.pop("terrain_levels", None)
    return cfg
