"""Deeprobotics DR02 AMP task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsDR02ProAMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:DeeproboticsDR02ProAMPFlatPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-AMP-Flat-Deeprobotics-DR02-Pro-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:DeeproboticsDR02ProAMPRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:DeeproboticsDR02ProAMPFlatPPORunnerCfg",
    },
)


gym.register(
    id="RobotLab-Isaac-AMP-Rough-Deeprobotics-DR02-Pro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsDR02ProAMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:DeeproboticsDR02ProAMPFlatPPORunnerCfg",
    },
)
