"""Training configuration for Unitree G1 AMP."""

from pathlib import Path

from rll_rl import AMPRunnerCfg

_G1_MOTION_GLOB = (
    str(Path.home() / "GMR-private/retarget_data/unitree_g1/motions/dance*.pkl"),
)


def unitree_g1_amp_runner_cfg() -> AMPRunnerCfg:
    cfg = AMPRunnerCfg(max_iterations=30_000, save_interval=500)
    cfg.obs_groups = {
        "actor": ("actor",),
        "critic": ("critic",),
        "amp": ("amp",),
    }
    cfg.amp.motion_files = _G1_MOTION_GLOB
    cfg.actor_critic.actor_hidden_dims = (512, 256, 128)
    cfg.actor_critic.critic_hidden_dims = (512, 256, 128)
    cfg.algorithm.learning_rate = 5.0e-4
    cfg.algorithm.num_mini_batches = 4
    cfg.discriminator.hidden_dims = (1024, 512)
    return cfg


def unitree_g1_dex3_backpack_amp_runner_cfg() -> AMPRunnerCfg:
    """Create an AMP runner configuration with an isolated backpack log namespace."""
    cfg = unitree_g1_amp_runner_cfg()
    cfg.experiment_name = "g1_dex3_backpack_amp_flat"
    return cfg
