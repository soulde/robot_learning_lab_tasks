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
    clip_actions = 100.0
    experiment_name = "deeprobotics_dr02_pro_amp_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
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
        # Looser KL and lower entropy, synced with the chocolate AMP setup:
        # keeps the adaptive learning rate from collapsing when the terrain
        # difficulty shifts.
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
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
        self.algorithm.amp_include_root_height = False
        # BeyondMimic/SOMA exports in this motion directory use MuJoCo/GMR WXYZ.
        self.algorithm.motion_quaternion_format = "wxyz"
        # Chocolate combines rewards as:
        #   0.8 * task_reward + 0.2 * (0.2 * style_reward)
        # Its discriminator reward is bounded in [0, 1], so the equivalent
        # style coefficient in this fork is 0.04, not 0.16.
        # Exact Chocolate AmpPose mixing:
        #   0.8 * task_reward + 0.2 * (0.2 * bounded_style_reward)
        self.algorithm.task_reward_scale = 0.8
        self.algorithm.style_reward_scale = 0.04
        # Chocolate discriminator geometry; discriminator updates share the
        # policy's Adam optimizer and KL-adaptive learning rate. Its 20 updates
        # match the 5 epochs x 4 minibatches below.
        self.algorithm.discriminator_hidden_dims = [512, 256]
        self.algorithm.discriminator_activation = "lrelu"
        self.algorithm.discriminator_learning_rate = 5.0e-4
        self.algorithm.discriminator_batch_size = 4096
        self.algorithm.discriminator_updates = 20
        self.algorithm.discriminator_loss_scale = 1.0
        self.algorithm.discriminator_logit_regularization_scale = 0.0
        self.algorithm.discriminator_gradient_penalty_scale = 10.0
        self.algorithm.discriminator_weight_decay_scale = 0.0
        # Large discriminator replay buffer keeps older policy samples in the
        # expert/policy comparison.
        self.algorithm.amp_replay_buffer_size = 1_000_000
