import pytest

pytest.importorskip("pxr", reason="Isaac Sim application must be running")
gym = pytest.importorskip("gymnasium")


def test_dr02_pro_camera_tasks_are_registered() -> None:
    import robot_learning_lab_tasks.tasks.isaaclab.manager_based.locomotion.velocity.config.humanoid.deeprobotics_dr02_pro  # noqa: F401

    assert gym.spec("RobotLab-Isaac-Velocity-Rough-Deeprobotics-DR02-Pro-Camera-v0") is not None
    assert gym.spec("RobotLab-Isaac-Velocity-Flat-Deeprobotics-DR02-Pro-Camera-v0") is not None


def test_dr02_pro_camera_cfg_exposes_three_configurable_d435_streams() -> None:
    from robot_learning_lab_tasks.tasks.isaaclab.manager_based.locomotion.velocity.config.humanoid.deeprobotics_dr02_pro.camera_env_cfg import (
        DeeproboticsDR02ProCameraRoughEnvCfg,
    )

    cfg = DeeproboticsDR02ProCameraRoughEnvCfg()

    expected = {
        "d435_head": (0.11871, 0.0175, 0.5096),
        "d435_front": (0.12559, 0.0175, 0.078407),
        "d435_rear": (-0.11241, -0.0175, -0.037264),
    }
    for name, position in expected.items():
        camera = getattr(cfg.scene, name)
        assert camera.prim_path == f"{{ENV_REGEX_NS}}/Robot/body/{name}"
        assert camera.offset.pos == position
        assert camera.offset.convention == "world"
        assert camera.data_types == ["rgb", "distance_to_image_plane"]
        assert camera.width == 640
        assert camera.height == 480
        assert camera.update_period == 1.0 / 30.0
        assert camera.spawn.clipping_range == (0.1, 10.0)

    cfg.scene.d435_rear = None
    cfg.scene.d435_head.width = 848
    cfg.scene.d435_head.data_types = ["distance_to_image_plane"]
    assert cfg.scene.d435_rear is None
    assert cfg.scene.d435_head.width == 848
    assert cfg.scene.d435_head.data_types == ["distance_to_image_plane"]
