"""Exercise RSI placement on CPU without starting the simulator."""

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


EVENTS = Path(__file__).parents[1] / "robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/events.py"


@pytest.fixture
def reset_fixture(monkeypatch):
    tree = ast.parse(EVENTS.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"reset_amp_reference_state", "_amp_reset_ground_height"}]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(EVENTS), "exec",
                 flags=__import__("__future__").annotations.compiler_flag), namespace)
    count = 4096
    frames = torch.zeros(count, 13)
    frames[:, :3] = torch.tensor([8.0, -5.0, 0.87])
    frames[:, 6] = 1.0
    frames[:, 7:13] = torch.arange(6)
    joints = torch.tensor([[1.0, 2.0]]).repeat(count, 1)
    poses = torch.full((count, 7), -99.0)
    velocities = torch.full((count, 6), -99.0)
    joint_states = torch.zeros(count, 2)
    robot = SimpleNamespace(
        device="cpu", joint_names=["b", "a"],
        write_root_pose_to_sim=lambda values, env_ids: poses.index_copy_(0, env_ids, values),
        write_root_link_velocity_to_sim_index=lambda root_velocity, env_ids: velocities.index_copy_(0, env_ids, root_velocity),
        write_joint_state_to_sim=lambda pos, vel, env_ids: joint_states.index_copy_(0, env_ids, pos),
    )
    origins = torch.tensor([4.0, 84.0, 2.0]).repeat(count, 1)
    scene = {"robot": robot}
    class Scene(dict):
        pass
    scene = Scene(scene)
    scene.num_envs = count
    scene.env_origins = origins
    scene.sensors = {"height_scanner": SimpleNamespace(
        cfg=SimpleNamespace(mesh_prim_paths=["/World/ground"]),
        device="cpu", meshes={("/World/ground", "cpu"): object()})}
    warp_utils = ModuleType("isaaclab.utils.warp")
    # Replace only the simulator's mesh query with an analytic sloped surface.
    def raycast(starts, directions, mesh):
        hits = starts.clone()
        hits[:, 2] = starts[:, 0] * 0.1
        return hits, None, None, None
    warp_utils.raycast_mesh = raycast
    monkeypatch.setitem(sys.modules, warp_utils.__name__, warp_utils)
    def sample_rsi(env_ids):
        return {
            "root_pos": frames[env_ids, :3],
            "root_quat": frames[env_ids, 3:7],
            "root_lin_vel": frames[env_ids, 7:10],
            "root_ang_vel": frames[env_ids, 10:13],
            "joint_pos": joints[env_ids],
            "joint_vel": joints[env_ids] * 2,
        }
    env = SimpleNamespace(
        scene=scene,
        device="cpu",
        _amp_rsi_sampler=sample_rsi,
        _amp_rsi_joint_layout={
            "joint_names": ("a", "b"),
            "body_names": ("base_link", "left_ankle_x_link", "right_ankle_x_link"),
        },
    )
    env.unwrapped = env
    return namespace["reset_amp_reference_state"], env, frames, poses, velocities, joint_states


def test_random_xy_uses_ground_at_new_position_and_preserves_reference(reset_fixture):
    reset, env, frames, poses, velocities, joints = reset_fixture
    before = frames.clone()
    torch.manual_seed(17)
    reset(env, None, asset_cfg=SimpleNamespace(name="robot"), terrain_clearance=0.0,
          xy_range=(-1.0, 1.0), terrain_height_sensor="height_scanner")
    offsets = poses[:, :2] - env.scene.env_origins[:, :2]
    assert bool((offsets.abs() <= 1.0).all())
    assert bool((offsets.std(dim=0) > 0.5).all())
    assert torch.unique(offsets, dim=0).shape[0] > 4000
    torch.testing.assert_close(poses[:, 2], 0.87 + poses[:, 0] * 0.1)
    torch.testing.assert_close(poses[:, 3:7], before[:, 3:7])
    torch.testing.assert_close(velocities, before[:, 7:13])
    torch.testing.assert_close(joints, torch.tensor([[2.0, 1.0]]).repeat(4096, 1))
    torch.testing.assert_close(frames, before)


def test_default_reset_keeps_origin_height_and_only_resets_selected_envs(reset_fixture):
    reset, env, _, poses, _, _ = reset_fixture
    reset(env, torch.tensor([2, 7]), asset_cfg=SimpleNamespace(name="robot"), terrain_clearance=0.0)
    torch.testing.assert_close(poses[[2, 7], :3], torch.tensor([[4.0, 84.0, 2.87]]).repeat(2, 1))
    assert bool((poses[0] == -99.0).all())


def test_ground_query_miss_fails_before_writing_invalid_pose(reset_fixture, monkeypatch):
    reset, env, _, poses, _, _ = reset_fixture
    def miss(starts, directions, mesh):
        return torch.full_like(starts, float("inf")), None, None, None
    monkeypatch.setattr(sys.modules["isaaclab.utils.warp"], "raycast_mesh", miss)
    with pytest.raises(RuntimeError, match="terrain"):
        reset(env, torch.tensor([2]), asset_cfg=SimpleNamespace(name="robot"), terrain_clearance=0.0,
              xy_range=(-1.0, 1.0),
              terrain_height_sensor="height_scanner")
    assert bool((poses == -99.0).all())


def test_reset_applies_rsl_precomputed_support_offsets(reset_fixture):
    reset, env, _, poses, velocities, _ = reset_fixture
    env.scene["robot"].joint_names = ["a", "b"]
    original_sampler = env._amp_rsi_sampler
    def sample_with_offsets(ids):
        reference = original_sampler(ids)
        reference["root_height_offset"] = torch.full((len(ids),), 0.05)
        reference["root_height_velocity_offset"] = torch.full((len(ids),), 0.5)
        return reference
    env._amp_rsi_sampler = sample_with_offsets
    reset(env, torch.tensor([2]), asset_cfg=SimpleNamespace(name="robot"),
          terrain_clearance=0.0)
    torch.testing.assert_close(poses[2, 2], torch.tensor(2.92))
    torch.testing.assert_close(velocities[2, 2], torch.tensor(2.5))
