from pathlib import Path


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
