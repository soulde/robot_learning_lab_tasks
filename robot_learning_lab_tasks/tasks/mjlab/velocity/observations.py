"""Robot Learning Lab MJLab observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    wheel_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return relative joint positions with wheel positions replaced by zero."""
    asset = env.scene[asset_cfg.name]
    values = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    num_joints = asset.data.joint_pos.shape[1]
    asset_id_values = (
        range(*asset_cfg.joint_ids.indices(num_joints))
        if isinstance(asset_cfg.joint_ids, slice)
        else asset_cfg.joint_ids
    )
    wheel_id_values = (
        range(*wheel_asset_cfg.joint_ids.indices(num_joints))
        if isinstance(wheel_asset_cfg.joint_ids, slice)
        else wheel_asset_cfg.joint_ids
    )
    wheel_ids = torch.as_tensor(list(wheel_id_values), device=values.device)
    asset_ids = torch.as_tensor(list(asset_id_values), device=values.device)
    wheel_columns = torch.isin(asset_ids, wheel_ids).nonzero(as_tuple=False).squeeze(-1)
    values[:, wheel_columns] = 0.0
    return values
