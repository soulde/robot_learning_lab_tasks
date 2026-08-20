def test_mjlab_registers_dex3_tasks_without_replacing_g1_tasks() -> None:
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from robot_learning_lab_zoo.assets.mjlab.unitree import (
        UNITREE_G1_29DOF_CFG,
        UNITREE_G1_29DOF_DEX3_BACKPACK_CFG,
        UNITREE_G1_29DOF_DEX3_CFG,
    )

    import robot_learning_lab_tasks.tasks.mjlab  # noqa: F401

    expected = {
        "RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3",
        "RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3",
        "RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3",
    }
    assert expected <= set(list_tasks())
    old_cfg = load_env_cfg("RobotLab-MJLab-Velocity-Flat-Unitree-G1")
    dex3_cfg = load_env_cfg("RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3")
    assert old_cfg.scene.entities["robot"].spec_fn is UNITREE_G1_29DOF_CFG.spec_fn
    assert dex3_cfg.scene.entities["robot"].spec_fn is UNITREE_G1_29DOF_DEX3_CFG.spec_fn
    assert all("_hand_" not in name for name in old_cfg.actions["joint_pos"].scale)
    assert any("_hand_" in name for name in dex3_cfg.actions["joint_pos"].scale)

    backpack_expected = {
        "RobotLab-MJLab-Velocity-Rough-Unitree-G1-Dex3-Backpack",
        "RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3-Backpack",
        "RobotLab-MJLab-AMP-Flat-Unitree-G1-Dex3-Backpack",
    }
    assert backpack_expected <= set(list_tasks())
    backpack_cfg = load_env_cfg("RobotLab-MJLab-Velocity-Flat-Unitree-G1-Dex3-Backpack")
    assert backpack_cfg.scene.entities["robot"].spec_fn is UNITREE_G1_29DOF_DEX3_BACKPACK_CFG.spec_fn
    assert backpack_cfg.actions["joint_pos"].scale == dex3_cfg.actions["joint_pos"].scale
