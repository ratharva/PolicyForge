"""Write decoded, aligned episodes into a LeRobot v3 dataset root that
training/vendor/lerobot_datasource.py can read directly. Format-agnostic --
takes plain dict[str, np.ndarray] + RobotSchema, doesn't care which
ingestion strategy (mcap, hf_lerobot_mirror, agibot_hdf5) produced them.

The root can be local, s3://, or gs:// -- every read/write of a small file
(parquet, json) here goes through fsspec (training/data_prep/source.py's
open_fs()). Video encoding is the exception -- see _write_episode_video's
docstring for why that still goes to a local temp file first. Only local
roots have been run against real data.

One dedicated video file per (episode, camera) and one data parquet file
per episode -- simpler and safer to write correctly than LeRobot v3's own
multi-episode-per-file packing, at the cost of more/smaller files.

Two-phase write, because the data parquet's `index` column must hold GLOBAL
row numbers (cumulative across all prior episodes -- lerobot_datasource.py's
read-task filtering depends on this), which aren't known until every
episode's post-alignment length is known:
  Phase 1 (write_episode, parallel via Ray in convert.py): decode+align+encode
    video, write the data parquet with a LOCAL 0-based placeholder index.
  Phase 2 (finalize_dataset, sequential on the driver, cheap): patch each
    episode's index to its real global offset, then write meta/info.json,
    meta/tasks.parquet, meta/episodes/*.parquet.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training.common.robots import RobotSchema
from training.data_prep.source import open_fs

DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4"


def _is_local_root(root: str) -> bool:
    return "://" not in root


@dataclass
class EpisodeRecord:
    episode_index: int
    task: str
    length: int
    # Depth camera_key -> real (H, W, C) shape / numpy dtype name this
    # episode's depth arrays actually had -- depth keeps its native sensor
    # resolution (unlike RGB cameras, which are resized to image_size), so
    # finalize_dataset reads this off a real record instead of guessing.
    # Empty for every dataset without depth_camera_keys.
    depth_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    depth_dtypes: dict[str, str] = field(default_factory=dict)


# Not part of the LeRobot v3 schema -- lets this project detect a stale
# conversion. Without it, "does meta/info.json exist" is the only check,
# which says nothing about whether THIS dataset was built for the
# tasks/episode-cap/schema/dataset_source currently requested.
CONVERSION_PARAMS_FILENAME = "conversion_params.json"


def compute_conversion_params(
    tasks: list[str], max_episodes_per_task: int, robot: RobotSchema,
    image_size: tuple[int, int], dataset_source: str, revision: str | None = None,
) -> dict:
    """`tasks` are raw CLI substrings, not resolved task directory names --
    comparable both before discovery (train.py's is_prepared() gate) and
    after (convert.py's inner check) without re-resolving names either time.
    `dataset_source` lets train.py resolve the right RobotSchema back out of
    conversion_params.json without a --dataset-source flag in the common
    case -- see training/data_prep/strategies/registry.py. `revision`
    records the pinned source-repo commit this conversion actually used, so
    a later pin change correctly reads as "stale, reconvert"."""
    return {
        "dataset_source": dataset_source,
        "tasks": sorted(tasks),
        "max_episodes_per_task": max_episodes_per_task,
        "tick_fps": robot.tick_fps,
        "image_size": list(image_size),
        "camera_keys": list(robot.camera_keys),
        "revision": revision,
    }


def write_conversion_params(root: str, params: dict) -> None:
    fs, fs_root = open_fs(root)
    fs.makedirs(f"{fs_root}/meta", exist_ok=True)
    with fs.open(f"{fs_root}/meta/{CONVERSION_PARAMS_FILENAME}", "w") as f:
        json.dump(params, f, indent=2)


def read_conversion_params(root: str) -> dict | None:
    fs, fs_root = open_fs(root)
    path = f"{fs_root}/meta/{CONVERSION_PARAMS_FILENAME}"
    if not fs.exists(path):
        return None
    with fs.open(path, "r") as f:
        return json.load(f)


def dataset_exists(root: str) -> bool:
    """True if meta/info.json exists at root. Doesn't check whether it
    matches a given request -- see prepare.py's is_prepared() for that."""
    fs, fs_root = open_fs(root)
    return fs.exists(f"{fs_root}/meta/info.json")


def _encode_video(local_path: str, frames: np.ndarray, fps: float) -> None:
    """frames: (T, H, W, 3) uint8 RGB -> a real mp4 container (not a raw
    elementary stream), always written to a real local path -- see
    _write_episode_video's docstring for why."""
    container = av.open(local_path, mode="w")
    h, w = frames.shape[1], frames.shape[2]
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height = w, h
    stream.pix_fmt = "yuv420p"
    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():  # flush
        container.mux(packet)
    container.close()


def _write_episode_video(
    fs, fs_root: str, is_local: bool, episode_index: int, camera_key: str,
    frames: np.ndarray, fps: float,
) -> None:
    """Encoding always goes to a real local path first, even for a remote
    root: standard mp4 muxing patches byte offsets into an already-written
    header once the final size is known, which needs a seekable output, and
    fsspec's remote write handles (s3fs, gcsfs) are forward-only upload
    streams. Parquet/json elsewhere don't have this issue (sequential
    forward writes only), so they go straight through fs.open()."""
    rel_path = VIDEO_PATH_TEMPLATE.format(video_key=camera_key, chunk_index=0, file_index=episode_index)
    abs_path = f"{fs_root}/{rel_path}"

    if is_local:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        _encode_video(abs_path, frames, fps)
        return

    fs.makedirs(os.path.dirname(abs_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        _encode_video(tmp_path, frames, fps)
        fs.put(tmp_path, abs_path)
    finally:
        os.remove(tmp_path)


def _write_episode_data(
    fs, fs_root: str, episode_index: int, task_index: int,
    state: np.ndarray, action: np.ndarray, fps: float,
    depth: dict[str, np.ndarray] | None = None,
) -> None:
    """state/action: (T, dim) float32. `index` is a LOCAL 0-based placeholder
    here -- finalize_dataset() patches it to the real global offset.

    depth: {camera_key: (T,H,W,C) array}, written flattened to (T, H*W*C)
    per-row -- same "list of fixed-length 1D arrays" pyarrow convention
    state/action already use, just longer rows. NOT video-encoded (depth
    isn't RGB -- see training/data_prep/strategies/agibot_hdf5.py); the
    real (H,W,C)/dtype needed to reshape this back on read is recorded by
    the caller (write_episode) into the returned EpisodeRecord, since this
    flat column alone doesn't carry it."""
    t = state.shape[0]
    columns = {
        "index": pa.array(np.arange(t), type=pa.int64()),
        "episode_index": pa.array([episode_index] * t, type=pa.int64()),
        "frame_index": pa.array(np.arange(t), type=pa.int64()),
        "timestamp": pa.array(np.arange(t) / fps, type=pa.float64()),
        "task_index": pa.array([task_index] * t, type=pa.int64()),
        "observation.state": pa.array(list(state.astype(np.float32))),
        "action": pa.array(list(action.astype(np.float32))),
    }
    for cam_key, arr in (depth or {}).items():
        columns[f"depth.{cam_key}"] = pa.array(list(arr.reshape(t, -1)))
    table = pa.table(columns)
    rel_path = DATA_PATH_TEMPLATE.format(chunk_index=0, file_index=episode_index)
    abs_path = f"{fs_root}/{rel_path}"
    fs.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with fs.open(abs_path, "wb") as f:
        pq.write_table(table, f)


def write_episode(
    root: str, episode_index: int, task: str, task_index: int,
    aligned: dict, robot: RobotSchema,
) -> EpisodeRecord:
    """aligned: a dict shaped like whatever an ingestion strategy's
    decode_and_align() returns (keys "state", "action",
    "image.<camera_key>", and "depth.<camera_key>" for
    robot.depth_camera_keys) -- see training/data_prep/strategies/."""
    fs, fs_root = open_fs(root)
    is_local = _is_local_root(root)
    state, action = aligned["state"], aligned["action"]
    depth = {cam_key: aligned[f"depth.{cam_key}"] for cam_key in robot.depth_camera_keys}
    _write_episode_data(fs, fs_root, episode_index, task_index, state, action, robot.tick_fps, depth=depth)
    for cam_key in robot.camera_keys:
        _write_episode_video(
            fs, fs_root, is_local, episode_index, cam_key, aligned[f"image.{cam_key}"], robot.tick_fps,
        )
    return EpisodeRecord(
        episode_index=episode_index, task=task, length=state.shape[0],
        depth_shapes={k: tuple(v.shape[1:]) for k, v in depth.items()},
        depth_dtypes={k: str(v.dtype) for k, v in depth.items()},
    )


def finalize_dataset(
    root: str, records: list[EpisodeRecord], robot: RobotSchema,
    image_size: tuple[int, int], task_to_index: dict[str, int],
) -> None:
    """task_to_index MUST be stable across incremental conversion runs, not
    recomputed here by sorting -- each episode's `task_index` column is
    written once into its data parquet at write_episode() time and never
    touched again. Resorting it fresh every run could silently reassign a
    DIFFERENT task_index to an unrelated existing task, corrupting every
    previously-converted episode's task label.
    """
    fs, fs_root = open_fs(root)
    records = sorted(records, key=lambda r: r.episode_index)
    lengths = [r.length for r in records]
    global_from = np.cumsum([0] + lengths[:-1])
    total_frames = int(sum(lengths))

    tasks = sorted({r.task for r in records})

    # --- phase 2a: patch each data parquet's `index` to its real global offset ---
    for rec, offset in zip(records, global_from):
        rel_path = DATA_PATH_TEMPLATE.format(chunk_index=0, file_index=rec.episode_index)
        abs_path = f"{fs_root}/{rel_path}"
        with fs.open(abs_path, "rb") as f:
            table = pq.read_table(f)
        # frame_index is a stable 0-based local position (written once, never
        # touched again). `index` is what gets patched here -- for a REUSED
        # episode it may already hold a prior run's global offset, so using
        # it as the local baseline again would double-apply the offset.
        local_index = table.column("frame_index").to_numpy()
        table = table.set_column(
            table.schema.get_field_index("index"), "index",
            pa.array(local_index + int(offset), type=pa.int64()),
        )
        with fs.open(abs_path, "wb") as f:
            pq.write_table(table, f)

    # --- meta/tasks.parquet ---
    fs.makedirs(f"{fs_root}/meta", exist_ok=True)
    with fs.open(f"{fs_root}/meta/tasks.parquet", "wb") as f:
        pq.write_table(
            pa.table({"task_index": list(task_to_index.values()), "task": list(task_to_index.keys())}), f,
        )
    assert set(tasks) <= set(task_to_index), (
        f"task_to_index is missing task(s) {set(tasks) - set(task_to_index)} present in records "
        f"-- caller must extend it BEFORE calling finalize_dataset, not after"
    )

    # --- meta/episodes/chunk-000/file-000.parquet ---
    ep_cols = {
        "episode_index": [r.episode_index for r in records],
        "length": lengths,
        "data/chunk_index": [0] * len(records),
        "data/file_index": [r.episode_index for r in records],
    }
    for cam_key in robot.camera_keys:
        ep_cols[f"videos/{cam_key}/chunk_index"] = [0] * len(records)
        ep_cols[f"videos/{cam_key}/file_index"] = [r.episode_index for r in records]
        # Every episode has its own dedicated video file, so it always
        # starts at the beginning of its file.
        ep_cols[f"videos/{cam_key}/from_timestamp"] = [0.0] * len(records)
    fs.makedirs(f"{fs_root}/meta/episodes/chunk-000", exist_ok=True)
    with fs.open(f"{fs_root}/meta/episodes/chunk-000/file-000.parquet", "wb") as f:
        pq.write_table(pa.table(ep_cols), f)

    # --- meta/info.json ---
    h, w = image_size
    features = {
        "observation.state": {"dtype": "float32", "shape": [robot.state_dim]},
        "action": {"dtype": "float32", "shape": [robot.action_dim]},
    }
    for cam_key in robot.camera_keys:
        # Short key (e.g. "top") must match the videos/{cam_key}/... columns
        # above and VIDEO_PATH_TEMPLATE's video_key= -- lerobot_datasource.py
        # derives video_keys from these feature names. Renamed to
        # "observation.images.<cam_key>" at read time -- see
        # ray_dataset.py's build_lerobot_v3_dataset.
        features[cam_key] = {"dtype": "video", "shape": [h, w, 3]}
    for cam_key in robot.depth_camera_keys:
        # Real shape/dtype from whatever a real written episode actually
        # produced -- depth keeps its native sensor resolution (unlike RGB
        # cameras, which are resized to image_size), so this is read off a
        # record, not guessed. Deliberately NOT "dtype": "video" -- that's
        # what makes lerobot_datasource.py's video_keys derivation skip
        # this column and fall through to its generic non-video passthrough
        # (this column is a flat (T, H*W*C) parquet array, not muxed mp4).
        shaped = next((r for r in records if cam_key in r.depth_shapes), None)
        if shaped is None:
            print(f"WARNING: no episode recorded a shape for depth camera {cam_key!r} -- "
                  f"every episode with it must have failed or been skipped; omitting it from info.json")
            continue
        features[f"depth.{cam_key}"] = {
            "dtype": shaped.depth_dtypes[cam_key], "shape": list(shaped.depth_shapes[cam_key]),
        }
    info = {
        "total_frames": total_frames,
        "total_episodes": len(records),
        "fps": robot.tick_fps,
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    with fs.open(f"{fs_root}/meta/info.json", "w") as f:
        json.dump(info, f, indent=2)

    # meta/stats.json must exist and be valid JSON (LeRobotDatasourceMetadata
    # loads it unconditionally) but its content isn't consumed by this
    # pipeline -- training/data_prep/stats.py computes real normalization
    # stats by sampling the built Ray Dataset directly instead.
    with fs.open(f"{fs_root}/meta/stats.json", "w") as f:
        json.dump({}, f)

    print(
        f"finalized LeRobot v3 dataset at {root}: {len(records)} episodes, "
        f"{total_frames} frames, {len(tasks)} tasks"
    )


# ---------------------------------------------------------------------------
# Incremental conversion support
# ---------------------------------------------------------------------------
# Not part of the LeRobot v3 schema. Lets convert.py add newly requested
# episodes without re-streaming/re-decoding ones already converted, and
# (together with wipe_dataset) avoid leaving orphaned video/parquet files
# behind when episodes are removed from the request instead of added.

EPISODE_MANIFEST_FILENAME = "episode_manifest.json"


def read_episode_manifest(root: str) -> list[dict]:
    """[{"rel_path", "task", "episode_index", "task_index", "length"}, ...],
    or [] if this dataset predates the manifest (treated as "nothing to
    reuse" by convert.py -- safe, just means a full reconvert once)."""
    fs, fs_root = open_fs(root)
    path = f"{fs_root}/meta/{EPISODE_MANIFEST_FILENAME}"
    if not fs.exists(path):
        return []
    with fs.open(path, "r") as f:
        return json.load(f)


def write_episode_manifest(root: str, entries: list[dict]) -> None:
    fs, fs_root = open_fs(root)
    fs.makedirs(f"{fs_root}/meta", exist_ok=True)
    with fs.open(f"{fs_root}/meta/{EPISODE_MANIFEST_FILENAME}", "w") as f:
        json.dump(entries, f, indent=2)


def wipe_dataset(root: str) -> None:
    """Used when episodes need to be REMOVED from an existing conversion.
    LeRobot v3 requires contiguous 0-based episode_index
    (lerobot_datasource.py raises otherwise), so removing one in place
    would require renumbering and renaming every subsequent episode's
    files -- not implemented. A full wipe + rebuild is simpler and correct.

    Refuses to wipe root/home-shaped paths or anything missing
    meta/conversion_params.json -- that file (not meta/info.json, which any
    LeRobot v3 dataset has regardless of who wrote it) is PolicyForge's own
    marker."""
    fs, fs_root = open_fs(root)
    if "://" not in root:
        resolved = os.path.realpath(root)
        if resolved in ("/", os.path.expanduser("~")) or resolved == os.path.dirname(resolved):
            raise ValueError(f"refusing to wipe {root!r} -- resolves to {resolved!r}, looks like a real root/home directory")
    if fs.exists(fs_root) and not fs.exists(f"{fs_root}/meta/{CONVERSION_PARAMS_FILENAME}"):
        raise ValueError(
            f"refusing to wipe {root!r} -- no meta/{CONVERSION_PARAMS_FILENAME} found there, "
            f"so this doesn't look like a dataset this tool created"
        )
    if fs.exists(fs_root):
        fs.rm(fs_root, recursive=True)
