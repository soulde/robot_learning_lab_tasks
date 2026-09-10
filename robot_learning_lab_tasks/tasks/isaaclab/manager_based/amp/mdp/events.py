# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def reset_amp_reference_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str | None = None,
    motion_file_pattern: str | None = None,
    motion_files: list[str] | None = None,
    joint_names: list[str] | None = None,
    reset_frames: dict[str, int | None] | int | None = None,
    rsi_curriculum: bool = False,
):
    """Reset environments into sampled expert motion frames (RSI).

    The root pose/velocity and joint states are taken from a uniformly
    sampled reference frame instead of the default standing pose.
    """
    if motion_dir is None:
        return
    robot: Articulation = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    if len(env_ids) == 0:
        return

    from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp.motion_loader import (
        get_amp_motion_library,
    )

    library = get_amp_motion_library(
        motion_dir, motion_file_pattern, motion_files, robot.device, reset_frames
    )
    # Chocolate-style RSI curriculum: fast motion pools join the resets only
    # as the velocity command curriculum widens the speed range.
    pool_probs = _rsi_pool_mix(env) if rsi_curriculum else None
    root_states, joint_pos, joint_vel = library.sample_reset_states(len(env_ids), pool_probs)

    # Place the sampled frame at the environment origin. Motions are
    # captured on flat ground, so the data root height is the height above
    # the terrain: it is added on top of the environment origin height,
    # which anchors the spawn onto rough terrain as well.
    root_states = root_states.clone()
    root_states[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root_states[:, 2] += env.scene.env_origins[env_ids, 2]
    robot.write_root_pose_to_sim(root_states[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)

    # Reorder motion columns into the articulation's joint order when the
    # NPZ joint order (joints.json contract) differs from the sim ordering.
    if joint_names is not None and list(joint_names) != list(robot.joint_names):
        reorder = [list(joint_names).index(name) for name in robot.joint_names]
        joint_pos = joint_pos[:, reorder]
        joint_vel = joint_vel[:, reorder]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _rsi_pool_mix(env: ManagerBasedEnv) -> dict[str, float]:
    """Speed-pool probabilities from the velocity command curriculum progress.

    jog grows to 25% by half progress, fast to 15% at full progress
    (chocolate rsi_curriculum mix).
    """
    term = env.command_manager.get_term("base_velocity")
    current = max(abs(term.cfg.ranges.lin_vel_x[0]), abs(term.cfg.ranges.lin_vel_x[1]))
    target = max(abs(env._original_vel_x[0]), abs(env._original_vel_x[1])) if hasattr(env, "_original_vel_x") else current
    if target <= 0:
        return {"base": 1.0, "jog": 0.0, "fast": 0.0}
    progress = min(1.0, max(0.0, (current - 0.1 * target) / max(0.9 * target, 1e-6)))
    jog = 0.25 * min(1.0, progress / 0.5)
    fast = 0.15 * min(1.0, max(0.0, (progress - 0.4) / 0.6))
    return {"base": 1.0 - jog - fast, "jog": jog, "fast": fast}
