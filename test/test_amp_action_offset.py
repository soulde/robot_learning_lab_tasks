"""Verify default-pose randomization respects reordered action joints on CPU."""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

EVENTS = Path(__file__).parents[1] / 'robot_learning_lab_tasks/tasks/isaaclab/manager_based/amp/mdp/events.py'


@pytest.mark.parametrize('selected_joints', [slice(None), [1, 3]])
@pytest.mark.parametrize('action_joints', [[2, 0, 3, 1], [3, 1]])
def test_randomized_offset_matches_action_order_and_preserves_other_envs(selected_joints, action_joints):
    tree = ast.parse(EVENTS.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'randomize_joint_default_pos')
    namespace = {'torch': torch, '_randomize_prop_by_op': lambda data, *args, **kwargs: data + 0.01}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(EVENTS), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    original = torch.tensor([[0., 0.765, 1.25, -0.765], [0., 0.765, 1.25, -0.765]])
    robot = SimpleNamespace(device='cpu', data=SimpleNamespace(default_joint_pos=original.clone()))
    action = SimpleNamespace(_joint_ids=action_joints, _offset=original[:, action_joints].clone())
    env = SimpleNamespace(scene={'robot': robot}, action_manager=SimpleNamespace(get_term=lambda _: action))
    namespace['randomize_joint_default_pos'](env, torch.tensor([1]),
        SimpleNamespace(name='robot', joint_ids=selected_joints), (-0.01, 0.01), 'add')
    expected = robot.data.default_joint_pos[:, action_joints]
    assert torch.allclose(action._offset, expected)
    assert torch.equal(robot.data.default_joint_pos[0], original[0])
