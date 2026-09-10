"""Reference motion library for AMP reference state initialization (RSI).

Loads BeyondMimic-format NPZ motions (see deploy_std retarget_motion_standard)
so environments can reset into expert motion frames for AMP training.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch


class AmpMotionLibrary:
    """Concatenated expert motion frames sampled uniformly for resets."""

    def __init__(
        self,
        motion_dir: str,
        motion_file_pattern: str | None = None,
        motion_files: list[str] | None = None,
        device: str | torch.device = "cpu",
        reset_frames: dict[str, int | None] | int | None = None,
    ) -> None:
        discovered = sorted(glob.glob(os.path.join(motion_dir, "*.npz")))
        if not discovered:
            raise ValueError(f"AMP motion library found no NPZ files in {motion_dir}")
        if motion_file_pattern is not None and motion_files is not None:
            raise ValueError("motion_file_pattern and motion_files are mutually exclusive")
        if motion_files is not None:
            available = {os.path.basename(path) for path in discovered}
            missing = sorted(set(motion_files) - available)
            if missing:
                raise ValueError(f"AMP motion files not found in {motion_dir}: {missing}")
            selected = set(motion_files)
            paths = [path for path in discovered if os.path.basename(path) in selected]
        elif motion_file_pattern is not None:
            pattern = re.compile(motion_file_pattern)
            paths = [path for path in discovered if pattern.fullmatch(os.path.basename(path))]
            if not paths:
                raise ValueError(
                    f"AMP motion_file_pattern {motion_file_pattern!r} matched no NPZ files in {motion_dir}"
                )
        else:
            paths = discovered
        print(f"[AMP RSI] Loading {len(paths)} of {len(discovered)} NPZ files from {motion_dir}")

        joint_pos, joint_vel = [], []
        root_pos, root_quat = [], []
        root_lin_vel, root_ang_vel = [], []
        motion_starts = []
        selected_names = []
        resolved_windows = []
        reset_frames_spec = _resolve_reset_frames_spec(reset_frames)
        for path in paths:
            window = next(
                (n for pattern, n in reset_frames_spec if pattern.fullmatch(os.path.basename(path))),
                reset_frames_spec[-1][1],
            )
            resolved_windows.append(window)
            selected_names.append(os.path.basename(path))
            data = np.load(path, allow_pickle=True)
            for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                if key not in data:
                    raise ValueError(f"AMP motion {path} is missing required field {key!r}")
            # Body 0 is the free root body per the deploy_std bodies.json contract.
            root_pos.append(data["body_pos_w"][:, 0, :])
            root_quat.append(data["body_quat_w"][:, 0, :])
            root_lin_vel.append(data["body_lin_vel_w"][:, 0, :])
            root_ang_vel.append(data["body_ang_vel_w"][:, 0, :])
            joint_pos.append(data["joint_pos"])
            joint_vel.append(data["joint_vel"])
            motion_starts.append(sum(len(item) for item in joint_pos) - len(joint_pos[-1]))
        self.device = torch.device(device)
        self.joint_pos = torch.tensor(np.concatenate(joint_pos), dtype=torch.float32, device=self.device)
        self.joint_vel = torch.tensor(np.concatenate(joint_vel), dtype=torch.float32, device=self.device)
        # Root state layout: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
        self.root_states = torch.zeros(len(self.joint_pos), 13, dtype=torch.float32, device=self.device)
        self.root_states[:, 0:3] = torch.tensor(np.concatenate(root_pos), dtype=torch.float32, device=self.device)
        # body_quat_w is WXYZ per the GMR export (MuJoCo xquat), matching Isaac Lab.
        self.root_states[:, 3:7] = torch.tensor(np.concatenate(root_quat), dtype=torch.float32, device=self.device)
        self.root_states[:, 7:10] = torch.tensor(np.concatenate(root_lin_vel), dtype=torch.float32, device=self.device)
        self.root_states[:, 10:13] = torch.tensor(np.concatenate(root_ang_vel), dtype=torch.float32, device=self.device)
        # Build the reset pool per motion from its resolved window; a window
        # larger than the motion keeps the whole motion.
        # Speed-pool classification for the RSI curriculum (chocolate-style):
        # fast motions join resets only as the velocity command curriculum
        # progresses. Motions matching no pool regex use "base".
        pools = [_classify_rsi_pool(name) for name in selected_names]
        reset_pools: dict[str, list[int]] = {}
        for start, length, window, pool in zip(
            motion_starts, (len(item) for item in joint_pos), resolved_windows, pools
        ):
            stop = start + length if not window else start + min(window, length)
            reset_pools.setdefault(pool, []).extend(range(start, stop))
        self.reset_pools = {
            pool: torch.tensor(indices, dtype=torch.long, device=self.device)
            for pool, indices in reset_pools.items()
        }
        all_indices = sorted(i for ids in reset_pools.values() for i in ids)
        self.reset_frame_indices = torch.tensor(all_indices, dtype=torch.long, device=self.device)
        print(
            f"[AMP RSI] Loaded {len(self.joint_pos)} reference frames;"
            f" resets sample {len(self.reset_frame_indices)} of them (reset_frames={reset_frames})"
        )

    def __len__(self) -> int:
        return len(self.joint_pos)

    def sample_reset_states(
        self, count: int, pool_probs: dict[str, float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paired (root_states, joint_pos, joint_vel) from the same frames.

        ``pool_probs`` optionally weights speed pools (chocolate RSI
        curriculum); the pool assignment comes from the motion filename.
        """
        indices = self._sample_indices(count, pool_probs)
        return self.root_states[indices], self.joint_pos[indices], self.joint_vel[indices]

    def _sample_indices(self, count: int, pool_probs: dict[str, float] | None = None) -> torch.Tensor:
        if pool_probs:
            pools = [p for p, prob in pool_probs.items() if prob > 0 and p in self.reset_pools]
            probs = torch.tensor([pool_probs[p] for p in pools], dtype=torch.float, device=self.device)
            choices = torch.multinomial(probs, count, replacement=True)
            per_pool = [
                self.reset_pools[pool][
                    torch.randint(0, len(self.reset_pools[pool]), (int((choices == i).sum()),), device=self.device)
                ]
                for i, pool in enumerate(pools)
            ]
            return torch.cat(per_pool)
        return self.reset_frame_indices[
            torch.randint(0, len(self.reset_frame_indices), (count,), device=self.device)
        ]


def _classify_rsi_pool(name: str) -> str:
    """Speed pool for an RSI reset: base / jog / fast."""
    import re

    if re.search(r"fast|run|sprint", name, re.IGNORECASE):
        return "fast"
    if re.search(r"jog", name, re.IGNORECASE):
        return "jog"
    return "base"


_LIBRARY_CACHE: dict[tuple, AmpMotionLibrary] = {}


def get_amp_motion_library(
    motion_dir: str,
    motion_file_pattern: str | None = None,
    motion_files: list[str] | None = None,
    device: str | torch.device = "cpu",
    reset_frames: dict[str, int | None] | int | None = None,
) -> AmpMotionLibrary:
    """Return a cached library so every reset term shares one instance."""
    key = (
        motion_dir,
        motion_file_pattern,
        tuple(motion_files or ()),
        str(device),
        tuple(sorted((k, str(v)) for k, v in (reset_frames or {}).items()))
        if isinstance(reset_frames, dict)
        else str(reset_frames),
    )
    if key not in _LIBRARY_CACHE:
        _LIBRARY_CACHE[key] = AmpMotionLibrary(motion_dir, motion_file_pattern, motion_files, device, reset_frames)
    return _LIBRARY_CACHE[key]


def _resolve_reset_frames_spec(
    reset_frames: dict[str, int | None] | int | None,
) -> list[tuple[re.Pattern, int | None]]:
    """Turn the reset window config into [(compiled pattern, window), ...].

    The last entry is the default ("" matches every basename). A window of
    None or 0 samples the whole motion.
    """
    spec: dict = reset_frames if isinstance(reset_frames, dict) else {"default": reset_frames}
    default = spec.get("default")
    patterns = [(re.compile(pattern), window) for pattern, window in spec.items() if pattern != "default"]
    patterns.append((re.compile(r""), default))
    return patterns
