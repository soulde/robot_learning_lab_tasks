"""Locate the retargeted motion dataset root.

The AMP motion datasets (discriminator input, RSI reset library) live in a
data directory outside the repo. Its location varies per host and differs
inside containers. Resolve it from the ``RLL_MOTION_DATA_ROOT`` environment
variable when set, falling back to the historical default layout under the
user home directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def motion_data_root() -> Path:
    """Root of the retargeted motion data tree (env-overridable).

    Contains per-robot subdirectories such as ``dr02/`` and ``unitree_g1/``.
    """
    return Path(os.environ.get("RLL_MOTION_DATA_ROOT", str(Path.home() / "GMR-private" / "retarget_data")))
