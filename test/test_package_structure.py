import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "robot_learning_lab_tasks"
TASKS = PACKAGE / "tasks"


def test_backend_packages_are_isolated() -> None:
    assert (TASKS / "isaaclab" / "direct").is_dir()
    assert (TASKS / "isaaclab" / "manager_based").is_dir()
    assert (TASKS / "mjlab" / "__init__.py").is_file()
    assert (TASKS / "mjlab" / "velocity" / "rewards.py").is_file()
    for robot in ("unitree_a1", "unitree_go2", "unitree_g1", "deeprobotics_lite3", "dr02"):
        assert (TASKS / "mjlab" / "velocity" / robot / "env_cfgs.py").is_file()


def test_tasks_no_longer_depend_on_robot_lab() -> None:
    for source in PACKAGE.rglob("*.py"):
        contents = source.read_text(encoding="utf-8")
        assert "robot_lab.tasks" not in contents, source
        assert "robot_lab.assets" not in contents, source
        assert "mjlabplusplus" not in contents, source


def test_mjlab_extra_installs_runtime_registration_dependencies(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to validate package builds")

    dist_dir = tmp_path / "dist"
    cache_dir = tmp_path / "uv-cache"
    subprocess.run(
        [
            uv,
            "build",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--out-dir",
            str(dist_dir),
            str(ROOT),
        ],
        env={"PATH": os.environ["PATH"], "UV_CACHE_DIR": str(cache_dir)},
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(dist_dir.glob("robot_learning_lab_tasks-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read("robot_learning_lab_tasks-0.1.0.dist-info/METADATA").decode("utf-8")
        entry_points = archive.read("robot_learning_lab_tasks-0.1.0.dist-info/entry_points.txt").decode("utf-8")

    assert 'Requires-Dist: mjlab; extra == "mjlab"' in metadata
    assert 'Requires-Dist: robot_learning_lab_zoo[mjlab]; extra == "mjlab"' in metadata
    assert 'Requires-Dist: rll_rl[amp]; extra == "mjlab"' in metadata
    assert "robot_learning_lab_tasks = robot_learning_lab_tasks.tasks.mjlab" in entry_points
