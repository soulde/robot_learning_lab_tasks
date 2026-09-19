"""Locate the per-robot retargeted motion dataset directory.

The AMP motion datasets (discriminator input, RSI reset library) live in a
data directory outside the repo. Its location varies per host and differs
inside containers. Resolve it from the ``RLL_MOTION_DATA_DIR`` environment
variable when set; the variable points directly at the robot-level directory
that holds the robot config (``bodies.json``) and the motion data together,
so users keep config and data in one directory. Falls back to the historical
DR02 default layout under the user home directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def motion_data_root() -> Path:
    """Robot-level root of the retargeted motion data (env-overridable).

    The directory holds the robot config (``bodies.json``) and the motion
    npz files together.
    """
    return Path(os.environ.get("RLL_MOTION_DATA_DIR", str(Path.home() / "GMR-private" / "retarget_data" / "dr02")))
