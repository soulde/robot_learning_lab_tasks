"""MJLab observation terms for adversarial motion priors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import subtract_frame_transforms

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")


def _quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(
        torch.finfo(quaternion.dtype).eps
    )
    x, y, z, w = quaternion.unbind(dim=-1)
    column_0 = torch.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)),
        dim=-1,
    )
    column_1 = torch.stack(
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)),
        dim=-1,
    )
    return torch.cat((column_0, column_1), dim=-1)


def root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def root_orientation(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    quat_wxyz = asset.data.root_link_quat_w
    quat_xyzw = torch.cat((quat_wxyz[:, 1:], quat_wxyz[:, :1]), dim=-1)
    return _quaternion_to_rotation_6d(quat_xyzw)


def root_linear_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


def root_angular_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


def joint_position(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_velocity(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def link_positions(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    count = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        asset.data.root_link_pos_w[:, None, :].expand(-1, count, -1),
        asset.data.root_link_quat_w[:, None, :].expand(-1, count, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.flatten(start_dim=1)
