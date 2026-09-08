"""Per-camera image normalization -- owns 100% of VISUAL-feature scaling,
independently of lerobot's own per-FeatureType NormalizerProcessorStep.

lerobot's normalization_mapping is one mode per FeatureType (VISUAL/STATE/
ACTION), and its native modes (MEAN_STD, MIN_MAX -> [-1,1] from *dataset*
min/max, QUANTILES, IDENTITY) don't cover "divide by 255", "depth", or
"log" at all -- so every training/model/{act,molmoact2,pi05}.py build_*_config
sets normalization_mapping["VISUAL"] = "IDENTITY" and this module's
apply_image_normalization() is called once in training/train_loop.py,
before the policy's own preprocessor(batch), to do all image scaling
itself. STATE/ACTION normalization is untouched -- still lerobot's own
MEAN_STD pipeline.
"""
from __future__ import annotations

import numpy as np
import torch

MODES = ("mean_std", "unit01", "unit_pm1", "depth", "log")
IMAGE_KEY_PREFIX = "observation.images."


def _require_max(cam: str, mode: str, max_values: dict[str, float]) -> float:
    if cam not in max_values:
        raise ValueError(
            f"camera {cam!r} uses image-normalization mode {mode!r}, which needs an explicit "
            f"max value (--image-normalization-max {cam}=<value>) -- no guessed default, since "
            f"it depends on the real sensor's value range (e.g. max depth in millimeters)."
        )
    value = max_values[cam]
    if value <= 0:
        # 0 divides by zero (depth/log); negative gives inverted clipping/NaN logs.
        raise ValueError(f"image-normalization max for camera {cam!r} must be positive, got {value}")
    return value


def _normalize_one(x, mode: str, cam: str, stats: dict, key: str, max_values: dict[str, float]):
    """x is either a torch.Tensor (training/train_loop.py's batches, already
    collated) or a np.ndarray (the Ray-Data-side --molmoact2-offload-tokenization
    path, still numpy at this stage) -- both are supported so the same
    normalization actually applies on either path."""
    is_torch = isinstance(x, torch.Tensor)
    if mode == "mean_std":
        s = stats[key]
        if is_torch:
            mean = torch.as_tensor(s["mean"], dtype=x.dtype, device=x.device)
            std = torch.as_tensor(s["std"], dtype=x.dtype, device=x.device)
        else:
            mean = np.asarray(s["mean"], dtype=x.dtype)
            std = np.asarray(s["std"], dtype=x.dtype)
        return (x - mean) / std
    if mode == "unit01":
        return x / 255.0
    if mode == "unit_pm1":
        return x / 127.5 - 1.0
    if mode == "depth":
        m = _require_max(cam, mode, max_values)
        clipped = torch.clamp(x, min=0.0, max=m) if is_torch else np.clip(x, 0.0, m)
        return clipped / m
    if mode == "log":
        m = _require_max(cam, mode, max_values)
        if is_torch:
            clipped = torch.clamp(x, min=0.0)
            log_max = torch.log1p(torch.as_tensor(float(m), dtype=x.dtype, device=x.device))
            return torch.log1p(clipped) / log_max
        clipped = np.clip(x, 0.0, None)
        return np.log1p(clipped) / np.log1p(m)
    raise ValueError(f"unknown image_normalization mode {mode!r} for camera {cam!r} -- must be one of {MODES}")


def _replicate_to_3_channels(x):
    """Any single-channel observation.images.* array/tensor (a depth
    camera, in practice -- RGB is already 3 channels and untouched) gets
    replicated to 3 channels here, so it flows through the exact same
    3-channel VISUAL path pretrained vision backbones expect (ACT's
    ResNet18, MolmoAct2's/PI05's vision encoders) instead of needing every
    backbone to accept 1 channel. Runs AFTER normalization, on already-CHW
    (or batched NCHW) arrays -- see training/train_loop.py's call order --
    so channel is axis -3. Generic on channel count, not depth-specific."""
    channel_axis = -3
    if x.ndim < 3 or x.shape[channel_axis] != 1:
        return x
    reps = [1] * x.ndim
    reps[channel_axis] = 3
    if isinstance(x, torch.Tensor):
        return x.repeat(*reps)
    return np.tile(x, reps)


def visual_input_features(data_cfg) -> dict:
    """PolicyFeature entries for every camera in data_cfg.robot -- RGB
    (camera_keys) and depth (depth_camera_keys) alike, all declared
    3-channel: apply_image_normalization replicates any single-channel
    camera (depth) to 3 channels (see _replicate_to_3_channels) before the
    policy ever sees it, so the declared shape must match what the policy
    actually receives, not depth's native 1-channel form. Shared by every
    training/model/{act,molmoact2,pi05}.py build_*_config."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    h, w = data_cfg.image_size
    keys = list(data_cfg.robot.camera_keys) + [f"depth_{k}" for k in data_cfg.robot.depth_camera_keys]
    return {
        f"observation.images.{k}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for k in keys
    }


def apply_image_normalization(
    batch: dict, camera_modes: dict[str, str], default_mode: str, stats: dict, max_values: dict[str, float],
) -> dict:
    """Scales every `observation.images.<camera>` array/tensor in `batch`
    according to `camera_modes` (camera key -> mode; a camera missing from
    it uses `default_mode`). `stats` is training/data/stats.py's
    compute_dataset_stats output (mean/std per image key) -- only consulted
    for cameras using "mean_std", which replicates today's default
    behavior exactly. `max_values` is a required per-camera clip/scale
    ceiling for "depth"/"log" cameras. Also replicates any single-channel
    camera (depth) to 3 channels -- see _replicate_to_3_channels."""
    out = dict(batch)
    for key, x in batch.items():
        if not key.startswith(IMAGE_KEY_PREFIX):
            continue
        cam = key[len(IMAGE_KEY_PREFIX):]
        mode = camera_modes.get(cam, default_mode)
        out[key] = _replicate_to_3_channels(_normalize_one(x, mode, cam, stats, key, max_values))
    return out
