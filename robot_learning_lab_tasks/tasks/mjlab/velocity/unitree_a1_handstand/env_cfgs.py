"""Unitree A1 handstand environment configurations."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from robot_learning_lab_tasks.tasks.mjlab.velocity import rewards
from robot_learning_lab_tasks.tasks.mjlab.velocity.quadruped_env_cfgs import quadruped_rough_env_cfg
from robot_learning_lab_tasks.tasks.mjlab.velocity.unitree_a1.env_cfgs import UNITREE_A1_VELOCITY_CFG

AIR_FOOT_GEOM_NAMES = ("RR_foot_collision", "RL_foot_collision")
AIR_FOOT_SITE_NAMES = ("RR_foot", "RL_foot")


def unitree_a1_handstand_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Unitree A1 rear-leg handstand task on rough terrain."""
    cfg = quadruped_rough_env_cfg(UNITREE_A1_VELOCITY_CFG, play=play)
    cfg.episode_length_s = 10.0 if not play else int(1e9)

    thigh_ground_cfg = ContactSensorCfg(
        name="thigh_ground_contact",
        primary=ContactMatch(mode="body", pattern=".*_thigh", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (thigh_ground_cfg,)

    keep_rewards = {
        "track_linear_velocity",
        "track_angular_velocity",
        "dof_pos_limits",
        "action_rate_l2",
        "joint_torques_l2",
        "joint_acc_l2",
        "joint_power",
    }
    cfg.rewards = {name: term for name, term in cfg.rewards.items() if name in keep_rewards}
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["dof_pos_limits"].weight = -5.0
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["joint_torques_l2"].weight = -1.0e-3
    cfg.rewards["joint_acc_l2"].weight = -2.5e-6
    cfg.rewards["joint_power"].weight = -2.0e-4
    cfg.rewards["undesired_contacts"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": thigh_ground_cfg.name, "force_threshold": 1.0},
    )
    cfg.rewards["handstand_feet_height_exp"] = RewardTermCfg(
        func=rewards.handstand_feet_height_exp,
        weight=10.0,
        params={
            "std": math.sqrt(0.25),
            "target_height": 0.5,
            "asset_cfg": SceneEntityCfg("robot", site_names=AIR_FOOT_SITE_NAMES),
        },
    )
    cfg.rewards["handstand_feet_on_air"] = RewardTermCfg(
        func=rewards.handstand_feet_on_air,
        weight=5.0,
        params={"sensor_name": "feet_ground_contact", "foot_names": AIR_FOOT_GEOM_NAMES},
    )
    cfg.rewards["handstand_feet_air_time"] = RewardTermCfg(
        func=rewards.handstand_feet_air_time,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "foot_names": AIR_FOOT_GEOM_NAMES,
            "threshold": 5.0,
        },
    )
    cfg.rewards["handstand_orientation_l2"] = RewardTermCfg(
        func=rewards.handstand_orientation_l2,
        weight=-1.0,
        params={"target_gravity": (1.0, 0.0, 0.0)},
    )

    cfg.terminations.pop("fell_over", None)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "nonfoot_ground_contact", "force_threshold": 1.0},
    )
    cfg.curriculum.pop("command_vel", None)

    return cfg


def unitree_a1_handstand_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Unitree A1 rear-leg handstand task on flat terrain."""
    cfg = unitree_a1_handstand_rough_env_cfg(play=play)
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)
    if play:
        cfg.events.pop("randomize_terrain", None)
    return cfg
