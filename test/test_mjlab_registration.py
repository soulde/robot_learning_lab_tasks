import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent

TWO_TERRAIN_ROBOTS = {
    "ANYmal-D",
    "Agibot-D1",
    "Booster-T1",
    "DDTRobot-Tita",
    "DR02",
    "DeepRobotics-Lite3",
    "Deeprobotics-M20",
    "FFTAI-GR1T1",
    "FFTAI-GR1T2",
    "HandStand-Unitree-A1",
    "MagicLab-Bot-Gen1",
    "MagicLab-Bot-Z1",
    "MagicLab-Dog",
    "MagicLab-Dog-W",
    "Openloong-Loong",
    "RoboParty-ATOM01",
    "RobotEra-Xbot",
    "Unitree-A1",
    "Unitree-B2",
    "Unitree-B2W",
    "Unitree-G1",
    "Unitree-Go2",
    "Unitree-Go2W",
    "Unitree-H1",
    "Zsibot-ZSL1",
    "Zsibot-ZSL1W",
}


def test_migrated_mjlab_tasks_register_complete_runtime_configs() -> None:
    pytest.importorskip("mjlab", exc_type=ImportError)
    dependency_paths = [
        SOURCE_ROOT / "rll_rl" / "src",
        SOURCE_ROOT / "robot_learning_lab_zoo",
        SOURCE_ROOT / "robot_learning_lab_datasets",
        ROOT,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in dependency_paths)
    env["MPLCONFIGDIR"] = "/tmp/robot-lab-matplotlib"
    script = """
import json
import robot_learning_lab_tasks.tasks.mjlab  # noqa: F401
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

task_ids = [task_id for task_id in list_tasks() if task_id.startswith("RobotLab-MJLab-")]
for task_id in task_ids:
    assert load_env_cfg(task_id) is not load_env_cfg(task_id)
    assert load_env_cfg(task_id, play=True) is not load_env_cfg(task_id, play=True)
    assert load_rl_cfg(task_id) is not load_rl_cfg(task_id)
    assert load_runner_cls(task_id) is not None
print(json.dumps(task_ids))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    actual = set(json.loads(result.stdout.strip().splitlines()[-1]))

    expected = {"RobotLab-MJLab-AMP-Flat-Unitree-G1", "RobotLab-MJLab-Velocity-Rough-Deeprobotics-DR02-Pro"}
    expected.update(
        f"RobotLab-MJLab-Velocity-{terrain}-{robot}" for terrain in ("Flat", "Rough") for robot in TWO_TERRAIN_ROBOTS
    )
    assert actual == expected
