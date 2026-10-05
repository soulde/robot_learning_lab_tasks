"""Check torso-only resets and illegal-contact penalties without a simulator."""

from pathlib import Path
import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
import torch


@pytest.fixture(params=["flat", "rough"])
def contact_case(request, monkeypatch, tmp_path):
    from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.config.dr02.flat_env_cfg import (
        DeeproboticsDR02ProAMPFlatEnvCfg,
    )
    from robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.config.dr02.rough_env_cfg import (
        DeeproboticsDR02ProAMPRoughEnvCfg,
    )
    import robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.config.dr02.flat_env_cfg as flat_module

    # Config construction discovers file names only; it never loads these frames.
    (tmp_path / "walking_slow04_stageii.npz").touch()
    monkeypatch.setattr(flat_module, "_ROBOT_DATA_ROOT", tmp_path)

    cfg_type = DeeproboticsDR02ProAMPFlatEnvCfg if request.param == "flat" else DeeproboticsDR02ProAMPRoughEnvCfg
    cfg = cfg_type()
    term = cfg.terminations.torso_contact
    urdf = Path(__file__).parents[2] / "robot_learning_lab_zoo/robots/deeprobotics/dr02_pro_description/urdf/dr02_pro.urdf"
    names = [link.attrib["name"] for link in ET.parse(urdf).getroot().findall("link")]
    penalty = cfg.rewards.undesired_contacts
    for contact_term in (term, penalty):
        sensor_cfg = contact_term.params["sensor_cfg"]
        patterns = sensor_cfg.body_names
        if isinstance(patterns, str):
            patterns = [patterns]
        sensor_cfg.body_ids = [i for i, name in enumerate(names) if any(re.fullmatch(p, name) for p in patterns)]
    forces = torch.zeros(1, 1, len(names), 3)
    env = SimpleNamespace(
        scene=SimpleNamespace(sensors={"contact_forces": SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=forces,
                                 net_normal_forces_w_history=SimpleNamespace(torch=forces)))}),
        num_envs=1, device="cpu", step_dt=0.02, episode_length_buf=torch.tensor([100]),
    )
    return term, env, names, forces, penalty


def test_torso_contact_resets_on_first_step(contact_case):
    term, env, names, forces, _ = contact_case
    forces[0, 0, names.index("body"), 2] = 1.1
    assert term.func(env, **term.params).item() is True


@pytest.mark.parametrize("body", [
    "base_link", "waist_x_link", "waist_z_link", "neck_link", "head_link",
    "left_shoulder_y_link", "right_shoulder_x_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_x_link", "right_wrist_z_link",
    "left_hand_link", "right_hand_link",
])
def test_other_upper_body_contacts_do_not_reset(contact_case, body):
    term, env, names, forces, _ = contact_case
    forces[0, 0, names.index(body), 2] = 100.0
    assert term.func(env, **term.params).item() is False


@pytest.mark.parametrize("body", ["left_knee_link", "right_ankle_x_link", "left_hip_y_link"])
def test_leg_contacts_do_not_trigger_upper_body_reset(contact_case, body):
    term, env, names, forces, _ = contact_case
    forces[0, 0, names.index(body), 2] = 100.0
    assert term.func(env, **term.params).item() is False


def test_subthreshold_torso_contact_does_not_reset(contact_case):
    term, env, names, forces, _ = contact_case
    forces[0, 0, names.index("body"), 2] = 0.9
    assert term.func(env, **term.params).item() is False


def test_two_illegal_contacts_receive_doubled_penalty(contact_case):
    _, env, names, forces, penalty = contact_case
    forces[0, 0, names.index("body"), 2] = 11.0
    forces[0, 0, names.index("left_knee_link"), 2] = 11.0
    forces[0, 0, names.index("right_ankle_x_link"), 2] = 100.0
    forces[0, 0, names.index("waist_x_link"), 2] = 9.0
    assert (penalty.func(env, **penalty.params) * penalty.weight).item() == -20.0


@pytest.mark.parametrize("body", [
    "base_link", "body", "waist_x_link", "waist_z_link", "neck_link", "head_link",
    *[f"{side}_{part}_link" for side in ("left", "right")
      for part in ("shoulder_x", "shoulder_y", "shoulder_z", "elbow",
                   "wrist_x", "wrist_y", "wrist_z", "hand")],
])
def test_every_upper_body_link_receives_illegal_contact_penalty(contact_case, body):
    _, env, names, forces, penalty = contact_case
    forces[0, 0, names.index(body), 2] = 11.0
    assert (penalty.func(env, **penalty.params) * penalty.weight).item() == -10.0
