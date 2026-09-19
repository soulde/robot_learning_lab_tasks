# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse

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


def amp_root_height(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2:3]


def amp_root_orientation(env: ManagerBasedEnv) -> torch.Tensor:
    quat = env.scene["robot"].data.root_quat_w.torch
    mat = matrix_from_quat(quat)
    # First two ROWS of the rotation matrix, matching the expert dataset
    # (BeyondMimic AMP contract). Slicing [..., :2] would take columns.
    return mat[..., :2, :].reshape(env.num_envs, -1)


def amp_root_linear_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_w.torch


def amp_root_angular_velocity(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_ang_vel_w.torch


def amp_joint_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]


def amp_joint_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def amp_link_positions(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    # SceneEntityCfg resolves body_ids via find_bodies, which returns
    # robot-internal ascending order — NOT the configured body_names order.
    # The expert AMP dataset indexes key bodies in the configured name
    # order, so resolve the ids here per name to keep both sides aligned.
    all_names = list(robot.body_names)
    if asset_cfg.body_names is None:
        ordered_ids = list(asset_cfg.body_ids)
    else:
        names = [asset_cfg.body_names] if isinstance(asset_cfg.body_names, str) else list(asset_cfg.body_names)
        ordered_ids = [all_names.index(name) for name in names]
    body_pos = robot.data.body_pos_w.torch[:, ordered_ids]
    count = body_pos.shape[1]
    # Root-frame projection with the convention-correct quaternion rotate
    # (this IsaacLab EA stack and the NPZ motion contract are xyzw).
    pos_b = quat_apply_inverse(
        robot.data.root_quat_w.torch.repeat_interleave(count, dim=0),
        (body_pos - robot.data.root_pos_w.torch[:, None, :]).reshape(-1, 3),
    ).reshape(-1, count * 3)
    return pos_b
