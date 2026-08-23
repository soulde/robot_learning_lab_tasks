# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsDR02ProRoughEnvCfg


@configclass
class DeeproboticsDR02ProFlatEnvCfg(DeeproboticsDR02ProRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "DeeproboticsDR02ProFlatEnvCfg":
            self.disable_zero_weight_rewards()
