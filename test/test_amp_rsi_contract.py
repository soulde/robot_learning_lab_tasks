"""CPU checks for reference link velocities and collision-aware RSI height."""
import ast
import re
from pathlib import Path
from types import SimpleNamespace
import torch

MDP = Path(__file__).parents[1] / 'robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp'

def test_dr02_stair_motions_are_excluded_from_amp_and_rsi():
    config = MDP.parent / 'config/dr02/flat_env_cfg.py'
    tree = ast.parse(config.read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'DR02_AMP_MOTION_EXCLUDES'
                              for target in node.targets))
    excludes = [re.compile(pattern) for pattern in ast.literal_eval(assignment.value)]
    stair_names = ['stairs_climbing_down_loop_R_001__A300.npz',
                   'stairs_climbing_jog_up_start_R_001__A300.npz']
    assert all(any(pattern.fullmatch(name) for pattern in excludes) for name in stair_names)

def test_amp_observations_use_link_velocity():
    tree = ast.parse((MDP/'observations.py').read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'amp_root_linear_velocity','amp_root_angular_velocity'}]
    ns = {}; exec(compile(ast.Module(body=funcs,type_ignores=[]),'obs','exec',flags=__import__('__future__').annotations.compiler_flag),ns)
    linear=torch.tensor([[1.,2.,3.]]);angular=torch.tensor([[4.,5.,6.]])
    env=SimpleNamespace(scene={'robot':SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=SimpleNamespace(torch=linear),root_link_ang_vel_w=SimpleNamespace(torch=angular)))})
    assert ns['amp_root_linear_velocity'](env) is linear
    assert ns['amp_root_angular_velocity'](env) is angular

def test_rsi_reset_consumes_rsl_rl_xyzw_sampler():
    tree = ast.parse((MDP/'events.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'reset_amp_reference_state')
    ns = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'events', 'exec',
                 flags=__import__('__future__').annotations.compiler_flag), ns)
    written = {}
    class Robot:
        device = 'cpu'
        joint_names = ['joint_b', 'joint_a']
        def write_root_pose_to_sim(self, pose, env_ids):
            written['pose'] = pose
        def write_root_link_velocity_to_sim_index(self, root_velocity, env_ids):
            written['velocity'] = root_velocity
        def write_joint_state_to_sim(self, pos, vel, env_ids):
            written['joint_pos'], written['joint_vel'] = pos, vel
    robot = Robot()
    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., 20., 1.]])
    scene = Scene(robot=robot)
    reference = {
        'root_pos': torch.tensor([[3., 4., .8]]),
        'root_quat': torch.tensor([[.1, .2, .3, .9]]),  # RSL-RL RSI contract is XYZW.
        'root_lin_vel': torch.tensor([[1., 2., 3.]]),
        'root_ang_vel': torch.tensor([[4., 5., 6.]]),
        'joint_pos': torch.tensor([[10., 20.]]),
        'joint_vel': torch.tensor([[30., 40.]]),
        'root_height_offset': torch.tensor([.12]),
        'root_height_velocity_offset': torch.tensor([.3]),
    }
    sampled = []
    def sampler(env_ids):
        sampled.append(env_ids.clone())
        return reference
    env = SimpleNamespace(device='cpu', scene=scene, _amp_rsi_sampler=sampler,
                          _amp_rsi_joint_layout={'joint_names': ('joint_a', 'joint_b')})
    env.unwrapped = env
    ns['reset_amp_reference_state'](env, torch.tensor([1]), asset_cfg=SimpleNamespace(name='robot'),
                                    terrain_clearance=0.0)
    torch.testing.assert_close(written['pose'], torch.tensor([[10., 20., 1.92, .1, .2, .3, .9]]))
    torch.testing.assert_close(written['velocity'], torch.tensor([[1., 2., 3.3, 4., 5., 6.]]))
    torch.testing.assert_close(written['joint_pos'], torch.tensor([[20., 10.]]))
    torch.testing.assert_close(written['joint_vel'], torch.tensor([[40., 30.]]))
    assert len(sampled) == 1 and sampled[0].tolist() == [1]
