"""Opt-in W&B support (training/train_loop.py's --wandb path): configurable
metric filtering, and episode-preview GIFs sampled from real recorded
data. `filter_metrics` is standalone (no Ray/torch coupling, mirrors
training/perf_logging.py's style); `sample_episode_frames` reuses
training/vendor/lerobot_datasource.py's already-built
LeRobotDatasourceMetadata (constructed read-only -- CLAUDE.md's own
caution about vendor/ applies) rather than guessing at the v3 root's
layout.
"""
from __future__ import annotations

import fnmatch

import av
import numpy as np


def _matches_any(key: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(key, p) for p in patterns)


def filter_metrics(d: dict, include: list[str] | None, exclude: list[str]) -> dict:
    """`include=None` means "everything" (still subject to `exclude`);
    both support fnmatch glob patterns (e.g. "perf/*"), not just exact key
    names."""
    out = {}
    for k, v in d.items():
        if include is not None and not _matches_any(k, include):
            continue
        if exclude and _matches_any(k, exclude):
            continue
        out[k] = v
    return out


# --- Named metric groups (--wandb-metric-groups) -- friendly names for the
# glob patterns a user would otherwise have to already know exist. Purely
# a convenience layer over filter_metrics above: a group name expands to
# its glob list, which gets merged into the SAME --wandb-metrics allowlist
# --wandb-exclude-metrics still applies on top of, unchanged.
METRIC_GROUPS: dict[str, tuple[str, list[str]]] = {
    "core": (
        "loss curves at every cadence (train/window/epoch/val/test)",
        ["train/*", "window/*", "epoch/*", "val/*", "test/*"],
    ),
    "perf": ("training/perf_logging.py's --log-perf-metrics output", ["perf/*"]),
    "media": ("episode-preview + predicted-frames GIFs", ["gif/*", "val_gif/*"]),
}


def expand_metric_groups(names: list[str]) -> list[str]:
    """Expands group names (e.g. ["core", "perf"]) into their real glob
    patterns, deduped. Raises ValueError naming the real available groups
    on an unknown name -- never silently ignored."""
    unknown = [n for n in names if n not in METRIC_GROUPS]
    if unknown:
        raise ValueError(f"unknown metric group(s) {unknown} -- available: {sorted(METRIC_GROUPS)}")
    patterns: list[str] = []
    for name in names:
        patterns.extend(METRIC_GROUPS[name][1])
    return list(dict.fromkeys(patterns))


# Every concrete metric name/prefix this pipeline actually emits today --
# one documented place to look instead of grepping train_loop.py/
# perf_logging.py. Kept in sync by hand (these are real, hardcoded keys at
# their call sites, not derived) -- see --list-wandb-metrics in train.py.
# A custom/future PolicyAdapter.extra_metrics or .predict_frames
# implementation adds MORE keys under train/window/epoch/val/test or
# val_gif/ respectively that can't be listed here in advance -- see
# training/model/registry.py.
KNOWN_METRICS: list[str] = [
    *(
        f"{prefix}/{name}"
        for prefix in ("train", "window", "epoch", "val", "test")
        for name in ("loss", "l1_loss", "kld_loss")
    ),
    "train/lr_group0", "train/lr_group1  (per optimizer param group -- ACT has 2, MolmoAct2 has 4, PI05 is flat)",
    "perf/data_wait_s", "perf/preprocess_s", "perf/compute_s", "perf/optimizer_step_s",
    "perf/optimizer_step_s_avg", "perf/io_bound_fraction", "perf/samples_per_sec",
    "perf/checkpoint_save_s", "perf/effective_batch_size",
    "perf/gpu_util_pct", "perf/gpu_mem_util_pct  (need nvidia-ml-py installed)",
    "perf/vram_allocated_mb", "perf/vram_reserved_mb", "perf/vram_peak_mb",
    "gif/<camera>  (one per --wandb-gif-cameras entry)",
    "val_gif/<name>  (only if a policy implements PolicyAdapter.predict_frames)",
]


def format_metrics_catalog() -> str:
    """Human-readable listing for --list-wandb-metrics -- every named
    group (with its real glob patterns) plus every concrete metric name
    this pipeline can emit today."""
    lines = ["Metric groups (--wandb-metric-groups NAME [NAME ...]):", ""]
    for name, (description, patterns) in METRIC_GROUPS.items():
        lines.append(f"  {name:8s} {description}")
        lines.append(f"           -> {', '.join(patterns)}")
    lines += [
        "",
        "Known metric names/prefixes (--wandb-metrics / --wandb-exclude-metrics, fnmatch globs OK):",
        "",
    ]
    lines += [f"  {m}" for m in KNOWN_METRICS]
    return "\n".join(lines)


def sample_episode_frames(
    v3_root: str, episode_index: int, camera_key: str, n_frames: int,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decodes up to `n_frames` consecutive frames from ONE real recorded
    episode's own video, for `camera_key` -- reads directly from the
    converted v3 root's video files, bypassing Ray Data/the shuffled
    training-batch iterator entirely (the same "separate, purpose-built
    read path alongside the main pipeline" training/data/stats.py's own
    sampling already uses). The real video path + seek offset (`chunk`/
    `file_index`/`from_timestamp`) is resolved from
    `LeRobotDatasourceMetadata.episodes` and `.video_path_template` -- NOT
    a hardcoded "one file per episode" path assumption, since a dataset
    built by lerobot's own official v3 migration (droid) can pack multiple
    episodes into one video file, unlike this project's own writer's
    convention. `image_size`, if given, resizes each frame the same way
    `_decode_video_bytes` (training/data_prep/strategies/agibot_hdf5.py)
    does. Returns `(T, H, W, 3)` uint8, `T <= min(n_frames, episode length)`.
    Raises `ValueError` if `camera_key` isn't
    one of this dataset's real video keys (e.g. a depth camera, which
    isn't video-encoded -- see training/data_prep/lerobot_v3_writer.py)."""
    from training.vendor.lerobot_datasource import LeRobotDatasourceMetadata

    meta = LeRobotDatasourceMetadata(v3_root)
    if camera_key not in meta.video_keys:
        raise ValueError(
            f"{camera_key!r} isn't one of {v3_root!r}'s real video keys {meta.video_keys} -- "
            "sample_episode_frames only supports RGB (video-encoded) cameras."
        )

    ep_indices = meta.episodes.column("episode_index").to_pylist()
    try:
        row = ep_indices.index(episode_index)
    except ValueError:
        raise ValueError(f"episode_index {episode_index} not found in {v3_root!r}'s meta/episodes") from None

    chunk = meta.episodes.column(f"videos/{camera_key}/chunk_index")[row].as_py()
    file_index = meta.episodes.column(f"videos/{camera_key}/file_index")[row].as_py()
    from_ts = meta.episodes.column(f"videos/{camera_key}/from_timestamp")[row].as_py()
    # Bounds reading to this episode's own frames, for video files packing multiple episodes.
    episode_length = meta.episodes.column("length")[row].as_py()
    rel_path = meta.video_path_template.format(video_key=camera_key, chunk_index=chunk, file_index=file_index)
    path = f"{meta.fs_root}/{rel_path}"

    is_local = "://" not in v3_root
    if is_local:
        container = av.open(path)
    else:
        import fsspec

        fs, _ = fsspec.core.url_to_fs(v3_root)
        container = av.open(fs.open(path, "rb"))

    frames: list[np.ndarray] = []
    try:
        stream = container.streams.video[0]
        if from_ts > 0:
            container.seek(int(from_ts / stream.time_base), stream=stream)
        max_frames = min(n_frames, episode_length)
        for packet in container.demux(video=0):
            for frame in packet.decode():
                # Seek can land before from_ts (nearest keyframe) -- skip pre-roll frames.
                if frame.time is not None and frame.time < from_ts:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                if image_size is not None and arr.shape[:2] != image_size:
                    h, w = image_size
                    arr = frame.reformat(width=w, height=h).to_ndarray(format="rgb24")
                frames.append(arr)
                if len(frames) >= max_frames:
                    break
            if len(frames) >= max_frames:
                break
    finally:
        container.close()
    return np.stack(frames)
