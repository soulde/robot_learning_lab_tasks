# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply

from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def base_pos_z(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the z position of the robot base."""
    robot = env.scene["robot"]
    return robot.data.root_pos_w[:, 2:3]


def critic_terrain_heights(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5, scale: float = 2.5
) -> torch.Tensor:
    """Chocolate-style terrain heights relative to the robot root."""
    root_z = env.scene["robot"].data.root_pos_w.torch[:, 2:3]
    terrain_z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    return torch.clamp(root_z - offset - terrain_z, -1.0, 1.0) * scale


def amp_root_height(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2:3]


def amp_root_orientation(env: ManagerBasedEnv) -> torch.Tensor:
    quat = env.scene["robot"].data.root_quat_w.torch
    mat = matrix_from_quat(quat)
    # First two ROWS of the rotation matrix, matching the expert dataset
    # (BeyondMimic AMP contract). Slicing [..., :2] would take columns.
    return mat[..., :2, :].reshape(env.num_envs, -1)


def amp_root_linear_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_lin_vel_w.torch


def amp_root_angular_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_ang_vel_w.torch


def amp_joint_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def amp_joint_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def amp_link_positions(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    key_body_names: list[str] | None = None,
    virtual_body_offsets: dict[str, tuple[str, tuple[float, float, float]]] | None = None,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    all_names = list(robot.body_names)
    if key_body_names is None:
        if asset_cfg.body_names is None:
            ordered_names = [all_names[index] for index in asset_cfg.body_ids]
        else:
            ordered_names = (
                [asset_cfg.body_names]
                if isinstance(asset_cfg.body_names, str)
                else list(asset_cfg.body_names)
            )
    else:
        ordered_names = list(key_body_names)

    virtual_body_offsets = virtual_body_offsets or {}
    body_pos_all = robot.data.body_pos_w.torch
    body_quat_all = robot.data.body_quat_w.torch
    body_positions = []
    for name in ordered_names:
        if name in all_names:
            body_positions.append(body_pos_all[:, all_names.index(name)])
            continue
        if name not in virtual_body_offsets:
            raise ValueError(f"AMP key body '{name}' is neither a robot body nor a configured virtual point")
        parent_name, local_offset = virtual_body_offsets[name]
        if parent_name not in all_names:
            raise ValueError(f"AMP virtual point '{name}' parent body '{parent_name}' was not found")
        parent_index = all_names.index(parent_name)
        offset = torch.as_tensor(local_offset, device=body_pos_all.device, dtype=body_pos_all.dtype)
        offset_w = quat_apply(body_quat_all[:, parent_index], offset.expand(env.num_envs, -1))
        body_positions.append(body_pos_all[:, parent_index] + offset_w)

    body_pos = torch.stack(body_positions, dim=1)
    count = len(ordered_names)
    # AMP expert and policy observations both express root-relative points
    # along world axes; only subtract the root translation.
    return (body_pos - robot.data.root_pos_w.torch[:, None, :]).reshape(-1, count * 3)
