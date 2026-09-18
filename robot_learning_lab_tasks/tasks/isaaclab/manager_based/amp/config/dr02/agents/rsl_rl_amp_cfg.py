"""RSL-RL AMP configuration for Deeprobotics DR02 Pro."""

import json
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from ..flat_env_cfg import (
    DR02_AMP_KEY_BODY_NAMES,
    DR02_JOINT_NAMES,
    dr02_amp_body_names_path,
    dr02_amp_motion_dir,
    dr02_amp_motion_files,
)


@configclass
class DeeproboticsDR02ProAMPFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "deeprobotics_dr02_pro_amp_flat"
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.obs_groups = {
            "actor": ["policy"],
            "critic": ["critic"],
            "discriminator": ["amp"],
        }
        body_names_path = Path(dr02_amp_body_names_path())
        body_names = json.loads(body_names_path.read_text(encoding="utf-8"))["body_names"]
        self.algorithm.class_name = "rsl_rl.algorithms:AMP"
        self.algorithm.motion_dir = dr02_amp_motion_dir()
        # Shared with the env-side RSI library (single exclusion list).
        self.algorithm.motion_files = dr02_amp_motion_files()
        self.algorithm.body_names = body_names
        self.algorithm.key_body_names = list(DR02_AMP_KEY_BODY_NAMES)
        self.algorithm.joint_names = list(DR02_JOINT_NAMES)
        # Convex mixing (weights sum to 1), tuned for a steady-state 1:1
        # contribution balance: env task rewards are dt-scaled (~0.05/step)
        # while a converged style reward is ~0.7, so 0.93*0.05 = 0.07*0.7.
        self.algorithm.task_reward_scale = 0.93
        self.algorithm.style_reward_scale = 0.07
        self.algorithm.discriminator_hidden_dims = [1024, 512]
        self.algorithm.discriminator_learning_rate = 1e-4
        self.algorithm.discriminator_batch_size = 4096
        # Weaken the discriminator so the style reward can climb while the
        # policy is still far from the expert distribution.
        self.algorithm.discriminator_updates = 2
        self.algorithm.discriminator_gradient_penalty_scale = 10.0
        # Large discriminator replay buffer keeps older policy samples in the
        # expert/policy comparison.
        self.algorithm.amp_replay_buffer_size = 1_000_000
