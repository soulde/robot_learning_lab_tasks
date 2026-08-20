# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeG1AMPFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "unitree_g1_amp_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()

        # AMP-specific configuration
        motion_dir = str(Path.home() / "GMR-private" / "retarget_data" / "unitree_g1" / "motions")

        # Key body names for AMP discriminator observations
        # Must match command.body_names for dimension consistency
        key_body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        # AMP algorithm parameters
        self.algorithm.class_name = "rsl_rl.algorithms:AMP"
        self.algorithm.motion_dir = motion_dir
        self.algorithm.key_body_names = key_body_names
        self.algorithm.task_reward_scale = 0.0
        self.algorithm.style_reward_scale = 1.0
        self.algorithm.discriminator_hidden_dims = [1024, 512]
        self.algorithm.discriminator_learning_rate = 5e-4
        self.algorithm.discriminator_batch_size = 4096
        self.algorithm.discriminator_updates = 4


@configclass
class UnitreeG1Dex3BackpackAMPFlatPPORunnerCfg(UnitreeG1AMPFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_g1_dex3_backpack_amp_flat"
