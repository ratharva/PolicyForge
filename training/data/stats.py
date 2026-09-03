"""Approximate normalization statistics (mean/std per feature), sampled from a
subset of the built Ray Dataset -- not a full pass, which isn't needed for
stable mean/std estimates.

Image stats are computed over raw [0, 255] float32 pixel values, NOT rescaled
to [0, 1] -- this must match training/vendor/util.py's NumpyToTorchCollate, which
widens uint8 images to float32 with no /255 rescale.

Call this on the dataset BEFORE any HWC->CHW transpose stage (see
training/data/ray_dataset.py's build_lerobot_v3_dataset vs.
transpose_for_training) -- the transpose below assumes HWC input.
"""
from __future__ import annotations

import numpy as np

from training.config import DataConfig


def compute_dataset_stats(ds, cfg: DataConfig, n_samples: int = 500) -> dict:
    print(f"Sampling {n_samples} rows to compute normalization stats ...")
    rows = ds.take(n_samples)
    if not rows:
        raise ValueError("dataset produced zero rows -- can't compute normalization stats")

    states = np.stack([r["observation.state"] for r in rows])  # (N, state_dim)
    actions = np.stack([r["action"] for r in rows])  # (N, chunk_size, action_dim)
    is_pad = np.stack([r["action_is_pad"] for r in rows])  # (N, chunk_size)
    valid_actions = actions[~is_pad]  # (M, action_dim) -- excludes padded chunk tail

    stats: dict = {
        "observation.state": {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
        },
        "action": {
            "mean": valid_actions.mean(axis=0).tolist(),
            "std": valid_actions.std(axis=0).tolist(),
        },
    }

    for cam_key in cfg.robot.camera_keys:
        col = f"observation.images.{cam_key}"
        imgs = np.stack([r[col] for r in rows]).astype(np.float32)  # (N,H,W,3), range [0,255]
        imgs = imgs.transpose(0, 3, 1, 2)  # (N,3,H,W)
        stats[col] = {
            "mean": imgs.mean(axis=(0, 2, 3)).reshape(3, 1, 1).tolist(),
            "std": imgs.std(axis=(0, 2, 3)).reshape(3, 1, 1).tolist(),
        }

    print(f"  computed stats for: {list(stats.keys())}")
    return stats
