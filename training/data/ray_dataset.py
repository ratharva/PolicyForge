"""Ray Data pipeline for training.

Two ways to get a training-ready Ray Dataset:

  build_dataset_direct()      -- local episode.mcap files -> per-tick,
                                  action-chunked rows via our own flat_map.
                                  Kept for quick debugging; NOT what train.py
                                  uses by default anymore.

  build_lerobot_v3_dataset()  -- reads an already-converted LeRobot v3 root
                                  (see training/data/convert.py) via
                                  training/vendor/lerobot_datasource.py's
                                  LeRobotDatasource. This is train.py's
                                  default path.

The v3 path needs a HWC->CHW transpose stage (transpose_for_training) applied
AFTER any stats sampling, since training/vendor/util.py's NumpyToTorchCollate
(used in train_loop.py) assumes images already arrive CHW.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import ray

from training.config import DataConfig
from training.data.decode import read_raw_episode
from training.data.episode import align_episode, chunk_actions
from training.data.video_decode import decode_camera_stream


def _set_name(ds: "ray.data.Dataset", name: str | None) -> "ray.data.Dataset":
    if not name:
        return ds
    # set_name()'s return value isn't consistently documented across Ray
    # versions -- fall back to the pre-call reference if it returns None.
    result = ds.set_name(name)
    return result if result is not None else ds


# ---------------------------------------------------------------------------
# Direct MCAP path (debug/fallback -- not train.py's default)
# ---------------------------------------------------------------------------


def _decode_episode_row(row: dict[str, Any], cfg: DataConfig, chunk_size: int) -> list[dict[str, Any]]:
    mcap_path = row["path"]
    task = row["task"]
    try:
        raw = read_raw_episode(mcap_path, cfg)
        for cam_key in list(raw.cameras):
            raw.cameras[cam_key] = decode_camera_stream(raw.cameras[cam_key], cfg.image_size)
        aligned = align_episode(raw, cfg)
    except ValueError as e:
        print(f"WARNING: skipping {mcap_path} ({task}): {e}")
        return []

    action_chunks, is_pad = chunk_actions(aligned["action"], chunk_size)
    t = aligned["state"].shape[0]
    out = []
    for i in range(t):
        item = {
            "task": task,
            "observation.state": aligned["state"][i].astype(np.float32),
            "action": action_chunks[i].astype(np.float32),
            "action_is_pad": is_pad[i],
        }
        for cam_key in cfg.robot.camera_keys:
            item[f"observation.images.{cam_key}"] = aligned[f"image.{cam_key}"][i]  # uint8 HWC
        out.append(item)
    return out


def build_dataset_direct(
    local_by_task: dict[str, list[str]], cfg: DataConfig, chunk_size: int, name: str | None = None,
) -> "ray.data.Dataset":
    """Decodes MCAP directly, no LeRobot v3 conversion. Rows come out HWC
    uint8 (NOT transposed) -- if wiring this into train_loop.py, either add a
    transpose stage (see transpose_for_training below) or use a collate that
    permutes on the way to the GPU, not vendor/util.py's NumpyToTorchCollate as-is."""
    rows = [
        {"path": p, "task": task}
        for task, paths in local_by_task.items()
        for p in paths
    ]
    ds = ray.data.from_items(rows)
    ds = ds.flat_map(_decode_episode_row, fn_args=(cfg, chunk_size))
    return _set_name(ds, name)


# ---------------------------------------------------------------------------
# LeRobot v3 path (train.py's default)
# ---------------------------------------------------------------------------


def _rename_columns(row: dict, rename: dict[str, str]) -> dict:
    return {rename.get(k, k): v for k, v in row.items()}


def build_lerobot_v3_dataset(v3_root: str, chunk_size: int) -> "ray.data.Dataset":
    """Raw LeRobotDatasource read, renamed to observation.images.<key>: HWC
    uint8 images, action already chunked + padded by the datasource itself
    (action_chunk_size=chunk_size). Sample THIS (pre-transpose) for
    normalization stats -- see training/data/stats.py and train.py's call order.

    lerobot_v3_writer.py names video features with the short camera key (e.g.
    "top") to match meta/episodes' `videos/{key}/...` columns, which
    LeRobotDatasource requires -- renamed here to what the rest of the
    training pipeline (NumpyToTorchCollate, ACTConfig) expects.
    """
    from training.vendor.lerobot_datasource import LeRobotDatasource

    source = LeRobotDatasource(v3_root, action_chunk_size=chunk_size)
    rename = {k: f"observation.images.{k}" for k in source.meta.video_keys}
    return ray.data.read_datasource(source).map(_rename_columns, fn_args=(rename,))


def _transpose_images(batch: dict, image_keys: list[str]) -> dict:
    out = dict(batch)
    for key in image_keys:
        out[key] = np.transpose(np.stack(list(batch[key])), (0, 3, 1, 2))
    return out


def transpose_for_training(
    ds: "ray.data.Dataset", camera_keys: tuple[str, ...], name: str | None = None,
) -> "ray.data.Dataset":
    """HWC -> CHW, matching what training/vendor/util.py's NumpyToTorchCollate assumes."""
    image_keys = [f"observation.images.{k}" for k in camera_keys]
    ds = ds.map_batches(_transpose_images, batch_size=32, fn_args=(image_keys,))
    return _set_name(ds, name)


# ---------------------------------------------------------------------------
# MolmoAct2 Ray Data preprocessing offload (optional -- --molmoact2-offload-tokenization)
# ---------------------------------------------------------------------------
# Runs MolmoAct2's real HF tokenizer + image-processor preprocessing
# (make_molmoact2_pre_post_processors) as a Ray Data map_batches stage
# instead of inline in train_loop.py, so it scales across the cluster's CPU
# workers independently of GPU worker count. Moves the entire
# preprocessor(batch) call (normalization + tokenization together) since
# make_molmoact2_pre_post_processors returns one bundled pipeline object,
# not separable sub-steps.
#
# UNVERIFIED, flagged not hidden (see training/README.md): whether HF
# tokenizer/processor output round-trips correctly through Ray Data's
# Arrow-backed batch representation and then into
# training/vendor/util.py's NumpyToTorchCollate is not confirmed -- construct a
# real batch and compare tensor shapes/dtypes against the non-offloaded path
# before trusting this for a real run.


class _MolmoAct2PreprocessStage:
    """Ray Data map_batches actor class: builds the MolmoAct2 preprocessor
    ONCE per actor (loads the HF tokenizer/image processor from
    checkpoint_path), then applies it to every batch that flows through.
    Constructed from picklable inputs only -- HF processor objects aren't
    reliably picklable across a Ray Data actor boundary."""

    def __init__(self, data_cfg, overrides, train_cfg, dataset_stats: dict):
        from training.model.molmoact2 import build_molmoact2_config

        cfg = build_molmoact2_config(data_cfg, overrides, train_cfg, device="cpu")
        from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors

        self.preprocessor, _ = make_molmoact2_pre_post_processors(cfg, dataset_stats=dataset_stats)

    def __call__(self, batch: dict) -> dict:
        return self.preprocessor(batch)


def offload_molmoact2_preprocessing(
    ds: "ray.data.Dataset", data_cfg, overrides, train_cfg, dataset_stats: dict,
    concurrency: int, name: str | None = None,
) -> "ray.data.Dataset":
    """Runs the full MolmoAct2 preprocessor as a Ray Data map_batches stage
    (an actor pool of size `concurrency`) instead of inline in
    train_loop.py -- see module comment above. Caller (train.py) derives
    `concurrency` from live cluster CPU count, not a hardcoded number."""
    ds = ds.map_batches(
        _MolmoAct2PreprocessStage,
        concurrency=concurrency,
        fn_constructor_kwargs={
            "data_cfg": data_cfg, "overrides": overrides,
            "train_cfg": train_cfg, "dataset_stats": dataset_stats,
        },
    )
    return _set_name(ds, name)
