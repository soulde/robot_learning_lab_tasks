# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""DR02 Pro locomotion environments with three configurable RealSense D435 cameras."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .flat_env_cfg import DeeproboticsDR02ProFlatEnvCfg
from .rough_env_cfg import DeeproboticsDR02ProRoughEnvCfg

_D435_INTRINSICS_640X480 = [
    382.6,
    0.0,
    319.5,
    0.0,
    382.6,
    239.5,
    0.0,
    0.0,
    1.0,
]


def make_d435_camera_cfg(
    name: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
    *,
    width: int = 640,
    height: int = 480,
    update_period: float = 1.0 / 30.0,
    data_types: list[str] | None = None,
    clipping_range: tuple[float, float] = (0.1, 10.0),
) -> TiledCameraCfg:
    """Create a D435-like RGB-D camera attached to the DR02 Pro body."""
    if data_types is None:
        data_types = ["rgb", "distance_to_image_plane"]
    spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=_D435_INTRINSICS_640X480,
        width=640,
        height=480,
        clipping_range=clipping_range,
    )
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/body/{name}",
        update_period=update_period,
        offset=TiledCameraCfg.OffsetCfg(pos=position, rot=rotation, convention="world"),
        data_types=data_types,
        spawn=spawn,
        width=width,
        height=height,
        depth_clipping_behavior="zero",
    )


def _add_d435_cameras(scene) -> None:
    # Poses match the fixed d435_* links embedded in the vendor URDF. The
    # camera prims are parented to body because URDF conversion merges fixed
    # joints in the articulation asset.
    scene.d435_head = make_d435_camera_cfg(
        "d435_head",
        (0.11871, 0.0175, 0.5096),
        (0.953718, 0.0, 0.300703, 0.0),
    )
    scene.d435_front = make_d435_camera_cfg(
        "d435_front",
        (0.12559, 0.0175, 0.078407),
        (0.866025, 0.0, 0.500001, 0.0),
    )
    scene.d435_rear = make_d435_camera_cfg(
        "d435_rear",
        (-0.11241, -0.0175, -0.037264),
        (-3.18109e-06, -0.500001, -1.83661e-06, 0.866025),
    )


@configclass
class DeeproboticsDR02ProCameraRoughEnvCfg(DeeproboticsDR02ProRoughEnvCfg):
    """Rough-terrain DR02 Pro task with head, front, and rear D435 streams."""

    def __post_init__(self):
        super().__post_init__()
        _add_d435_cameras(self.scene)


@configclass
class DeeproboticsDR02ProCameraFlatEnvCfg(DeeproboticsDR02ProFlatEnvCfg):
    """Flat-terrain DR02 Pro task with head, front, and rear D435 streams."""

    def __post_init__(self):
        super().__post_init__()
        _add_d435_cameras(self.scene)
