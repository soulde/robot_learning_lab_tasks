"""Rough-terrain AMP environment configuration for Deeprobotics DR02 Pro.

Same AMP setup as the flat task but the plane is replaced by the Isaac Lab
rough terrain set with a terrain curriculum, following the existing DR02
velocity rough tasks. Terrain heights are privileged critic observations;
the policy and AMP discriminator retain their existing observations.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab_newton.physics import NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
from isaaclab_tasks.utils import PresetCfg

import robot_learning_lab_tasks.tasks.isaaclab.manager_based.locomotion.velocity.mdp as velocity_mdp
import robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp as amp_mdp
from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.tracking_env_cfg import AMPSceneCfg, ObservationsCfg

from .flat_env_cfg import DeeproboticsDR02ProAMPFlatEnvCfg


def _terrain_height_grid(cfg: patterns.GridPatternCfg, device: str):
    """Start yaw-aligned rays above the root without shifting the scan grid."""
    ray_starts, ray_directions = patterns.grid_pattern(cfg, device)
    ray_starts[:, 2] = 2.0
    return ray_starts, ray_directions


@configclass
class DR02AMPRoughSceneCfg(AMPSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            func=_terrain_height_grid,
            resolution=0.1,
            size=(1.6, 1.0),
            ordering="yx",
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class DR02AMPRoughObservationsCfg(ObservationsCfg):
    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        terrain_heights = ObsTerm(
            func=amp_mdp.critic_terrain_heights,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class DR02AMPRoughPhysicsCfg(PresetCfg):
    """Keep the existing PhysX run as default; expose Newton MJWarp explicitly."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    default: PhysxCfg = isaacsim_physx
    newton_mjwarp: NewtonCfg = RoughPhysicsCfg().newton_mjwarp


@configclass
class DeeproboticsDR02ProAMPRoughEnvCfg(DeeproboticsDR02ProAMPFlatEnvCfg):
    """DR02 Pro AMP rough terrain training configuration."""

    scene: DR02AMPRoughSceneCfg = DR02AMPRoughSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DR02AMPRoughObservationsCfg = DR02AMPRoughObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = DR02AMPRoughPhysicsCfg()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Rough terrain with the standard Isaac Lab terrain set and the
        # velocity-task terrain curriculum.
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            # Keep difficulty ordered by row without mutating the shared preset.
            terrain_generator=ROUGH_TERRAINS_CFG.copy().replace(curriculum=True),
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
        # Chocolate AMP Pose spreads RSI spawns within +/-1 m and queries
        # terrain height at each new XY rather than at the cell origin.
        self.events.reset_amp_reference_state.params["xy_range"] = (-1.0, 1.0)
        self.events.reset_amp_reference_state.params["terrain_height_sensor"] = "height_scanner"
        # Terrain difficulty follows walking speed, like the velocity tasks.
        self.curriculum.terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)
