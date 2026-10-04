# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from types import SimpleNamespace

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, yaw_quat
from isaaclab_tasks.core.velocity.mdp.rewards import track_lin_vel_xy_yaw_frame_exp as official_track_lin_vel_xy_yaw_frame_exp
from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp as official_track_ang_vel_z_exp

from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def horizontal_root_velocity(asset) -> torch.Tensor:
    """World velocity expressed in the root yaw frame; vertical stays vertical."""
    return quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch)


def body_velocity_state(env, body_name, asset_cfg):
    """Read the selected rigid body's pose and physical velocities."""
    asset = env.scene[asset_cfg.name]
    body_id = asset.body_names.index(body_name)
    return (asset.data.body_quat_w.torch[:, body_id],
            asset.data.body_lin_vel_w.torch[:, body_id],
            asset.data.body_ang_vel_w.torch[:, body_id])


def _body_velocity_reward_env(env, body_name, asset_cfg):
    """Adapt a selected body to the official root-based tracking kernels."""
    quat, linear, angular = body_velocity_state(env, body_name, asset_cfg)
    data = SimpleNamespace(root_quat_w=SimpleNamespace(torch=quat),
                           root_lin_vel_w=SimpleNamespace(torch=linear),
                           root_ang_vel_b=SimpleNamespace(torch=quat_apply_inverse(quat, angular)))
    return SimpleNamespace(scene={asset_cfg.name: SimpleNamespace(data=data)}, command_manager=env.command_manager)


def track_body_lin_vel_xy_yaw_frame_exp(env, std, command_name, body_name, asset_cfg=SceneEntityCfg("robot")):
    return official_track_lin_vel_xy_yaw_frame_exp(
        _body_velocity_reward_env(env, body_name, asset_cfg), std, command_name, asset_cfg
    )


def track_body_ang_vel_z_exp(env, std, command_name, body_name, asset_cfg=SceneEntityCfg("robot")):
    return official_track_ang_vel_z_exp(
        _body_velocity_reward_env(env, body_name, asset_cfg), std, command_name, asset_cfg
    )


def track_body_lin_vel_xy_yaw_frame_rms(env, command_name, body_name, asset_cfg=SceneEntityCfg("robot")):
    quat, linear, _ = body_velocity_state(env, body_name, asset_cfg)
    velocity = quat_apply_inverse(yaw_quat(quat), linear)
    error = env.command_manager.get_command(command_name)[:, :2] - velocity[:, :2]
    return torch.sqrt(error.square().mean(dim=-1) + 1e-8)


def track_body_ang_vel_z_rms(env, command_name, body_name, asset_cfg=SceneEntityCfg("robot")):
    quat, _, angular = body_velocity_state(env, body_name, asset_cfg)
    error = env.command_manager.get_command(command_name)[:, 2] - quat_apply_inverse(quat, angular)[:, 2]
    return torch.sqrt(error.square() + 1e-8)


def track_lin_vel_xy_yaw_frame_rms(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    error = env.command_manager.get_command(command_name)[:, :2] - horizontal_root_velocity(env.scene[asset_cfg.name])[:, :2]
    return torch.sqrt(error.square().mean(dim=-1) + 1e-8)


def track_lin_vel_xy_rms(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root-mean-square linear velocity tracking error (base frame, xy).

    Ported from the chocolate AMP setup: an unnormalized L2-style penalty
    complementing the exponential tracking kernel.
    """
    asset = env.scene[asset_cfg.name]
    error = env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]
    return torch.sqrt(torch.mean(torch.square(error), dim=-1) + 1e-8)


def track_ang_vel_z_rms(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root-mean-square yaw velocity tracking error (base frame).

    Ported from the chocolate AMP setup.
    """
    asset = env.scene[asset_cfg.name]
    error = env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    return torch.sqrt(torch.square(error) + 1e-8)


class ControlStepJointAccL2(ManagerTermBase):
    """Penalize joint acceleration measured between consecutive control steps.

    Chocolate AMP differences velocities over its policy step and overwrites
    its history with the RSI velocity after each reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_joint_vel: torch.Tensor | None = None

    def reset(self, env_ids=None) -> None:
        # Reset events write RSI states before the reward manager is reset.
        joint_vel = self._env.scene["robot"].data.joint_vel.torch
        if self._previous_joint_vel is None:
            self._previous_joint_vel = joint_vel.clone()
        else:
            indexes = slice(None) if env_ids is None else env_ids
            self._previous_joint_vel[indexes] = joint_vel[indexes]

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        joint_vel = asset.data.joint_vel.torch
        if self._previous_joint_vel is None:
            self._previous_joint_vel = joint_vel.clone()
        joint_acc = (joint_vel[:, asset_cfg.joint_ids] - self._previous_joint_vel[:, asset_cfg.joint_ids]) / env.step_dt
        self._previous_joint_vel.copy_(joint_vel)
        return torch.sum(torch.square(joint_acc), dim=-1)
