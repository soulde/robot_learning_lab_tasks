"""Rough-terrain AMP environment configuration for Deeprobotics DR02 Pro.

Same AMP setup as the flat task but the plane is replaced by the Isaac Lab
rough terrain set with a terrain curriculum, following the existing DR02
velocity rough tasks. The robot has no height-scan observations (like the
velocity rough task) and relies on proprioception.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
from isaaclab_tasks.utils import PresetCfg

import robot_learning_lab_tasks.tasks.isaaclab.manager_based.locomotion.velocity.mdp as velocity_mdp

from .flat_env_cfg import DeeproboticsDR02ProAMPFlatEnvCfg


@configclass
class DR02AMPRoughPhysicsCfg(PresetCfg):
    """Keep the existing PhysX run as default; expose Newton MJWarp explicitly."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    default: PhysxCfg = isaacsim_physx
    newton_mjwarp: NewtonCfg = RoughPhysicsCfg().newton_mjwarp


@configclass
class DeeproboticsDR02ProAMPRoughEnvCfg(DeeproboticsDR02ProAMPFlatEnvCfg):
    """DR02 Pro AMP rough terrain training configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = DR02AMPRoughPhysicsCfg()

        # Rough terrain with the standard Isaac Lab terrain set and the
        # velocity-task terrain curriculum.
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=5,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=1.0,
            ),
            debug_vis=False,
        )
        self.sim.physics_material = self.scene.terrain.physics_material
        # Terrain difficulty follows walking speed, like the velocity tasks.
        self.curriculum.terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)
