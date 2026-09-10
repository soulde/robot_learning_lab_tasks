"""Fall-recovery AMP environment configuration for Deeprobotics DR02 Pro.

Harder variant of the flat task: strong external pushes can knock the robot
over and a fallen robot gets a grace period to stand back up before the
episode ends. Intended to be trained with fall/push recovery reference
motions in the expert dataset.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

import robot_learning_lab_tasks.tasks.isaaclab.manager_based.amp.mdp as mdp

from .flat_env_cfg import DeeproboticsDR02ProAMPFlatEnvCfg


@configclass
class DeeproboticsDR02ProAMPRecoveryEnvCfg(DeeproboticsDR02ProAMPFlatEnvCfg):
    """DR02 Pro AMP flat terrain with fall-recovery training features."""

    def __post_init__(self):
        super().__post_init__()
        # A fallen robot gets 5 s of trunk contact to recover (e.g. stand
        # back up) before the episode terminates.
        self.terminations.torso_contact.params["grace_time"] = 5.0
        # Rare strong push with fully randomized direction and magnitude
        # that can knock the robot over.
        self.events.randomize_knockdown_push = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(20.0, 30.0),
            params={
                "velocity_range": {
                    "x": (-3.0, 3.0),
                    "y": (-3.0, 3.0),
                    "z": (-0.5, 0.5),
                    "roll": (-1.5, 1.5),
                    "pitch": (-1.5, 1.5),
                    "yaw": (-2.0, 2.0),
                }
            },
        )
        # Recovery reference motions start from fallen states, so resets
        # sample every frame of the dataset.
        self.events.reset_amp_reference_state.params["reset_frames"] = {"default": None}
