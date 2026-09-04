"""The "agibot_hdf5" ingestion strategy: AgiBot World Alpha's custom format
-- per-task tar archives of HDF5 proprioception + per-task tar archives of
video (already-muxed mp4 containers, confirmed real -- av.open() works
directly, NOT training/data_prep/video_decode.py's raw-elementary-stream
decoder).

Real, gated-access-verified facts this module relies on (see
training/data_prep/schemas/agibot_alpha.yaml's comment for the full
sourcing): a proprio_stats.h5's `state/<path>`/`action/<path>` datasets and
its `timestamp` dataset are already ALL the same length T and already
positionally aligned -- confirmed by comparing a real episode's video frame
count (av-decoded) against its proprio row count: both 1136, exact match.
No floor-align step is needed here, unlike training/data_prep/strategies/mcap.py.

Real access pattern: neither observations/ nor proprio_stats/ tars have an
index (plain POSIX tar) -- a member can only be reached by streaming
sequentially from the start of its tar. Each task's observations/ tars are
split into several shards named "<lo>-<hi>.tar" by episode-id range;
proprio_stats/ is (today, for Alpha) a single shard covering the whole
dataset, but this module treats it generically as "however many shards
exist under that prefix", not hardcoded to one.

An earlier version of this module scanned shards live over HTTP (Range
requests to read headers without downloading each member's full data,
skipping forward via a computed offset). Measured for real against this
dataset, that was latency-bound, not bandwidth-bound (~2-2.7 headers/s per
connection regardless of parallelism, since each header read is its own
network round trip) and impractically slow even parallelized across many
connections (and real HF rate-limiting -- HTTP 429 -- kicks in well before
enough connections would fix it). This version instead downloads each
needed shard to local disk ONCE (training/data_prep/source.py's
download_one, which wraps huggingface_hub's own resumable, bandwidth-bound
bulk download -- proven infrastructure, already used by the mcap strategy)
and then scans the LOCAL copy with a plain single-pass tarfile scan --
skipping a non-wanted member there is a fast fseek, not a round trip, so
none of the old resync/parallel-region complexity is needed once the file
is local. Observations shards are downloaded per-task (download_task_observation_shards);
proprio_stats is downloaded as its own separate step
(download_proprio_shards), deliberately NOT tied to any one task, since
it's one archive shared across the whole dataset -- download_one's own
caching means a second task (or a later run) pays nothing extra once a
shard is already local. Real shard sizes checked directly against the repo
before choosing this approach: a single task's observations shards can
total 50-300GB, and proprio_stats is ~48GB dataset-wide -- both must
actually fit on local disk before this path is used for real.

Unlike the mcap strategy, this one does NOT go through
training/data_prep/convert.py's convert_to_lerobot_v3 -- that orchestration
assumes each episode is independently, cheaply fetchable (true for MCAP,
one file per episode; false here, since many episodes share one un-indexed
tar). Instead this module has its own prepare(), which:
  1. discovers real task names/episode ids (list_tasks/select_task_episodes)
  2. downloads every observations shard for each requested task, and every
     proprio_stats shard (once, shared across tasks), to local disk
  3. resolves which local shard covers each requested episode
     (resolve_local_shards) and scans each needed local shard EXACTLY ONCE
     (_scan_local_shard, Ray-parallel across shards), extracting every
     requested episode encountered along the way to local temp files -- not
     one open-and-scan per episode, which would re-pay the scan cost for
     everything before each episode's target
  4. transforms each episode's extracted files (real, already-verified
     _read_proprio_components/_decode_video_bytes) and writes it
     (write_episode, unchanged/format-agnostic)
Reuses plan_incremental_conversion (training/data_prep/incremental.py) for
manifest diffing/task_index-episode_index stability -- the same logic
convert_to_lerobot_v3 uses, not reimplemented here.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass

import av
import h5py
import numpy as np
import ray

from training.common.robots import RobotSchema
from training.data_prep.config import ConvertConfig
from training.data_prep.incremental import plan_incremental_conversion
from training.data_prep.lerobot_v3_writer import (
    EpisodeRecord,
    compute_conversion_params,
    dataset_exists,
    finalize_dataset,
    read_conversion_params,
    write_conversion_params,
    write_episode,
    write_episode_manifest,
)
from training.data_prep.source import download_one, parse_hf_uri


@dataclass
class AgiBotHdf5IngestionConfig:
    observations_tar_prefix: str
    proprio_stats_tar_prefix: str
    task_info_dir: str
    state_hdf5_paths: dict[str, str]
    action_hdf5_paths: dict[str, str]
    video_filename_template: str


def build_ingestion_config(spec: dict) -> AgiBotHdf5IngestionConfig:
    a = spec["agibot_hdf5"]
    return AgiBotHdf5IngestionConfig(
        observations_tar_prefix=a["observations_tar_prefix"],
        proprio_stats_tar_prefix=a["proprio_stats_tar_prefix"],
        task_info_dir=a["task_info_dir"],
        state_hdf5_paths=dict(a["state_hdf5_paths"]),
        action_hdf5_paths=dict(a["action_hdf5_paths"]),
        video_filename_template=a["video_filename_template"],
    )


# ---------------------------------------------------------------------------
# Phase 1: discover real tasks/episodes, resolve which tar shard covers each
# ---------------------------------------------------------------------------


def _load_task_info(repo_id: str, rel_path: str, token: str | None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id, rel_path, repo_type="dataset", token=token)
    with open(local_path) as f:
        return json.load(f)


def list_tasks(source_uri: str, agibot_cfg: AgiBotHdf5IngestionConfig, token: str | None) -> dict[str, str]:
    """{task_name: task_id} for every real task_info/task_<id>.json in the
    repo -- mirrors discover.py's role for MCAP, AgiBot-shaped."""
    from huggingface_hub import HfApi

    _, repo_id = parse_hf_uri(source_uri)
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    prefix = f"{agibot_cfg.task_info_dir}/task_"
    task_files = [f for f in files if f.startswith(prefix) and f.endswith(".json")]

    names_to_ids: dict[str, str] = {}
    for rel_path in task_files:
        task_id = rel_path[len(prefix):-len(".json")]
        episodes = _load_task_info(repo_id, rel_path, token)
        if episodes:
            names_to_ids[episodes[0]["task_name"]] = task_id
    return names_to_ids


def select_task_episodes(
    source_uri: str, agibot_cfg: AgiBotHdf5IngestionConfig,
    requested_tasks: list[str], max_episodes_per_task: int, token: str | None,
) -> dict[str, tuple[str, list[str]]]:
    """{task_name: (task_id, [episode_id, ...])} -- same substring-matching
    convention discover.py's select_episodes() uses for MCAP task names."""
    _, repo_id = parse_hf_uri(source_uri)
    names_to_ids = list_tasks(source_uri, agibot_cfg, token)

    selected: dict[str, tuple[str, list[str]]] = {}
    for requested in requested_tasks:
        matches = [name for name in names_to_ids if requested in name]
        if not matches:
            raise ValueError(
                f"No task name contains {requested!r}. Real task names: {sorted(names_to_ids)}"
            )
        if len(matches) > 1:
            print(f"  WARNING: {requested!r} matches {len(matches)} tasks {matches}; using {matches[0]!r}")
        task_name = matches[0]
        task_id = names_to_ids[task_name]
        episodes = _load_task_info(repo_id, f"{agibot_cfg.task_info_dir}/task_{task_id}.json", token)
        eids = sorted({str(e["episode_id"]) for e in episodes}, key=int)[:max_episodes_per_task]
        print(f"  {task_name}: using {len(eids)} episodes")
        selected[task_name] = (task_id, eids)
    return selected


def _list_tar_shards(source_uri: str, prefix: str, token: str | None) -> list[tuple[int, int, str]]:
    """[(lo, hi, rel_path), ...] for every real "<prefix>/<lo>-<hi>.tar" in
    the repo. rel_path (not a resolved URL) is what download_one needs."""
    from huggingface_hub import HfApi

    _, repo_id = parse_hf_uri(source_uri)
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    shards = []
    for f in files:
        if not (f.startswith(prefix + "/") and f.endswith(".tar")):
            continue
        name = f.rsplit("/", 1)[-1][:-len(".tar")]
        try:
            lo, hi = (int(x) for x in name.split("-"))
        except ValueError:
            continue
        shards.append((lo, hi, f))
    return shards


def list_observation_shards(
    source_uri: str, task_id: str, agibot_cfg: AgiBotHdf5IngestionConfig, token: str | None,
) -> list[tuple[int, int, str]]:
    """[(lo, hi, rel_path), ...] for every real observations shard belonging
    to task_id -- task-scoped, unlike proprio_stats (see list_proprio_shards)."""
    return _list_tar_shards(source_uri, f"{agibot_cfg.observations_tar_prefix}/{task_id}", token)


def list_proprio_shards(
    source_uri: str, agibot_cfg: AgiBotHdf5IngestionConfig, token: str | None,
) -> list[tuple[int, int, str]]:
    """[(lo, hi, rel_path), ...] for every real proprio_stats shard --
    dataset-wide (today, one ~48GB tar covering every task), not per-task."""
    return _list_tar_shards(source_uri, agibot_cfg.proprio_stats_tar_prefix, token)


def download_task_observation_shards(
    source_uri: str, task_id: str, agibot_cfg: AgiBotHdf5IngestionConfig, token: str | None,
    cache_dir: str | None = None,
) -> list[tuple[int, int, str]]:
    """Downloads EVERY observations shard belonging to task_id to local disk
    (via download_one -- huggingface_hub's own resumable, bandwidth-bound
    bulk download, not a per-header network round trip) and returns
    [(lo, hi, local_path), ...]. Deliberately whole-task, not
    per-episode-subset: once a task's shards are local, converting a
    different episode subset from the same task later needs no re-download
    (download_one's cache makes a repeat call for the same file a no-op)."""
    shards = list_observation_shards(source_uri, task_id, agibot_cfg, token)
    return [(lo, hi, download_one(source_uri, rel_path, token, cache_dir)) for lo, hi, rel_path in shards]


def download_proprio_shards(
    source_uri: str, agibot_cfg: AgiBotHdf5IngestionConfig, token: str | None,
    cache_dir: str | None = None,
) -> list[tuple[int, int, str]]:
    """Downloads every real proprio_stats shard to local disk. Kept as its
    own function, independent of any task_id -- proprio_stats is one shared
    archive across the WHOLE dataset (today: a single ~48GB tar), not a
    per-task resource, so this is called once per prepare() run regardless
    of how many tasks are being converted; download_one's cache means a
    second task in the same run (or a later run) pays nothing extra for an
    already-downloaded shard."""
    shards = list_proprio_shards(source_uri, agibot_cfg, token)
    return [(lo, hi, download_one(source_uri, rel_path, token, cache_dir)) for lo, hi, rel_path in shards]


def resolve_local_shards(
    episode_ids: list[str],
    obs_shards: list[tuple[int, int, str]],
    proprio_shards: list[tuple[int, int, str]],
) -> dict[str, tuple[str, str]]:
    """{episode_id: (local_observations_tar_path, local_proprio_tar_path)},
    matching each episode id against the already-downloaded shards' real
    filename ranges. Raises if any requested episode isn't covered by any
    downloaded shard, rather than silently skipping it."""
    def _find(shards: list[tuple[int, int, str]], eid_int: int) -> str | None:
        for lo, hi, path in shards:
            if lo <= eid_int <= hi:
                return path
        return None

    result = {}
    for eid in episode_ids:
        eid_int = int(eid)
        obs_path = _find(obs_shards, eid_int)
        proprio_path = _find(proprio_shards, eid_int)
        if obs_path is None or proprio_path is None:
            raise ValueError(
                f"episode {eid} not covered by any downloaded tar shard "
                f"-- observations={obs_path}, proprio_stats={proprio_path}"
            )
        result[eid] = (obs_path, proprio_path)
    return result


# ---------------------------------------------------------------------------
# Phase 2: scan each needed LOCAL shard exactly once, extract requested
# members. Shards are downloaded whole first (download_task_observation_shards/
# download_proprio_shards) -- see the module docstring for why that replaced
# an earlier live-HTTP-streaming approach.
# ---------------------------------------------------------------------------


def _extract_from_local_tar(tar_path: str, wanted_names: set[str], out_dir: str) -> dict[str, str]:
    """Single-pass scan of an already-local tar, extracting every member
    whose name is in wanted_names as it's encountered and stopping once all
    have been found. Skipping a non-wanted member is a real fseek (tarfile's
    own next()), not a network round trip, so -- unlike the old live-HF
    version -- no resync/parallel-region machinery is needed for this to be
    fast. Returns {member_name: local_path}; members not found are simply
    absent from the result."""
    found: dict[str, str] = {}
    remaining = set(wanted_names)
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not remaining:
                break
            if member.isfile() and member.name in remaining:
                local_path = os.path.join(out_dir, member.name.replace("/", "__"))
                with open(local_path, "wb") as f:
                    f.write(tar.extractfile(member).read())
                found[member.name] = local_path
                remaining.discard(member.name)
    if remaining:
        print(f"WARNING: {len(remaining)} member(s) not found in {tar_path}: "
              f"{sorted(remaining)[:5]}" + (" ..." if len(remaining) > 5 else ""))
    return found


@ray.remote
def _scan_local_shard(tar_path: str, wanted_names: list[str], out_dir: str) -> dict[str, str]:
    return _extract_from_local_tar(tar_path, set(wanted_names), out_dir)


# ---------------------------------------------------------------------------
# Transform (real, verified against real downloaded data -- see
# docs/customizing-datasets.md's "agibot_alpha" section)
# ---------------------------------------------------------------------------


def _read_proprio_components(
    h5file: h5py.File, group_prefix: str, hdf5_paths: dict[str, str],
) -> np.ndarray:
    """Concatenates components in the order given by hdf5_paths (a dict,
    insertion order preserved) into one (T, dim) array -- same "concatenate
    named components in order" convention strategies/mcap.py's
    align_episode uses, just with real HDF5 paths instead of MCAP topic
    names, and no floor-align needed (see module docstring)."""
    parts = []
    for path in hdf5_paths.values():
        arr = h5file[f"{group_prefix}/{path}"][()]
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)  # e.g. (T, 2, 4) -> (T, 8)
        parts.append(arr.astype(np.float64))
    return np.concatenate(parts, axis=1)


def _decode_video_bytes(data: bytes, image_size: tuple[int, int]) -> np.ndarray:
    container = av.open(io.BytesIO(data))
    h, w = image_size
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        if arr.shape[:2] != (h, w):
            av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            arr = av_frame.reformat(width=w, height=h).to_ndarray(format="rgb24")
        frames.append(arr)
    return np.stack(frames)


def _transform_episode(
    proprio_path: str, video_paths: dict[str, str], robot: RobotSchema,
    agibot_cfg: AgiBotHdf5IngestionConfig, image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    with h5py.File(proprio_path, "r") as h5file:
        state = _read_proprio_components(h5file, "state", agibot_cfg.state_hdf5_paths)
        action = _read_proprio_components(h5file, "action", agibot_cfg.action_hdf5_paths)
    aligned: dict[str, np.ndarray] = {"state": state, "action": action}
    for cam_key in robot.camera_keys:
        with open(video_paths[cam_key], "rb") as f:
            aligned[f"image.{cam_key}"] = _decode_video_bytes(f.read(), image_size)
    return aligned


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def prepare(
    robot: RobotSchema, agibot_cfg: AgiBotHdf5IngestionConfig, v3_root: str,
    tasks: list[str], max_episodes_per_task: int, image_size: tuple[int, int],
    dataset_source: str, source_uri: str, token: str | None,
    force: bool = False,
) -> str:
    exists = dataset_exists(v3_root)
    expected_params = compute_conversion_params(tasks, max_episodes_per_task, robot, image_size, dataset_source)
    if exists and not force:
        stored = read_conversion_params(v3_root)
        if stored == expected_params:
            print(f"LeRobot v3 dataset already exists at {v3_root}, matching this request "
                  f"-- skipping conversion (pass --reconvert to rebuild it anyway)")
            return v3_root

    print("\n=== discover real tasks/episodes ===")
    task_episodes = select_task_episodes(source_uri, agibot_cfg, tasks, max_episodes_per_task, token)
    # {task_name: (task_id, [episode_id, ...])} -> stable ids for incremental planning
    stable_ids_by_task = {
        task_name: [f"{task_id}/{eid}" for eid in eids]
        for task_name, (task_id, eids) in task_episodes.items()
    }
    task_id_by_name = {task_name: task_id for task_name, (task_id, _eids) in task_episodes.items()}

    plan = plan_incremental_conversion(v3_root, stable_ids_by_task, exists=exists, force=force)

    # Pre-assign episode_index to every to-convert entry BEFORE attempting
    # extraction (same convention convert_to_lerobot_v3 uses) -- a failed
    # episode just leaves its index unused, the manifest is built from this
    # fixed mapping either way, not from a counter that only advances on success.
    specs = []  # (task_name, task_id, episode_id, episode_index)
    episode_index = plan.next_episode_index
    for task_name, stable_id in plan.to_convert:
        task_id, episode_id = stable_id.split("/")
        specs.append((task_name, task_id, episode_id, episode_index))
        episode_index += 1

    new_records: list[EpisodeRecord] = []
    out_dir = tempfile.mkdtemp(prefix="agibot_extract_")
    try:
        if specs:
            print(f"\n=== resolve shards + extract {len(specs)} new episode(s) ===")
            # Group by task_id (shards are downloaded per-task) then by shard path.
            by_task_id: dict[str, list[tuple[str, str, int]]] = {}  # task_id -> [(task_name, episode_id, episode_index), ...]
            for task_name, task_id, episode_id, ep_idx in specs:
                by_task_id.setdefault(task_id, []).append((task_name, episode_id, ep_idx))

            # proprio_stats is one archive shared across every task -- download it
            # once here, distinct from any task's observations download below, so
            # a multi-task run (or a later run) never re-downloads it.
            print("  downloading proprio_stats shard(s) (shared across every task) ...")
            proprio_shards = download_proprio_shards(source_uri, agibot_cfg, token)

            proprio_paths: dict[str, str] = {}  # episode_id -> local path
            video_paths: dict[str, dict[str, str]] = {}  # episode_id -> {camera_key: local path}

            for task_id, entries in by_task_id.items():
                episode_ids = [eid for _tn, eid, _ei in entries]

                print(f"  task {task_id}: downloading observations shard(s) ...")
                obs_shards = download_task_observation_shards(source_uri, task_id, agibot_cfg, token)
                shard_map = resolve_local_shards(episode_ids, obs_shards, proprio_shards)

                obs_wanted: dict[str, list[str]] = {}  # local obs shard path -> [member_name, ...]
                proprio_wanted: dict[str, list[str]] = {}  # local proprio shard path -> [member_name, ...]
                for eid in episode_ids:
                    obs_path, proprio_path = shard_map[eid]
                    for cam_key in robot.camera_keys:
                        video_filename = agibot_cfg.video_filename_template.format(camera_key=cam_key)
                        obs_wanted.setdefault(obs_path, []).append(f"{eid}/videos/{video_filename}")
                    proprio_wanted.setdefault(proprio_path, []).append(f"{task_id}/{eid}/proprio_stats.h5")

                print(f"  task {task_id}: scanning {len(obs_wanted)} local observations shard(s) + "
                      f"{len(proprio_wanted)} local proprio shard(s) for {len(episode_ids)} episode(s) ...")
                refs = [_scan_local_shard.remote(path, names, out_dir) for path, names in obs_wanted.items()]
                refs += [_scan_local_shard.remote(path, names, out_dir) for path, names in proprio_wanted.items()]
                for found in ray.get(refs):
                    for member_name, local_path in found.items():
                        if member_name.endswith("proprio_stats.h5"):
                            eid = member_name.split("/")[1]
                            proprio_paths[eid] = local_path
                        else:
                            eid, _videos, filename = member_name.split("/")
                            cam_key = next(
                                c for c in robot.camera_keys
                                if agibot_cfg.video_filename_template.format(camera_key=c) == filename
                            )
                            video_paths.setdefault(eid, {})[cam_key] = local_path

            for task_name, task_id, episode_id, ep_idx in specs:
                have_proprio = episode_id in proprio_paths
                have_videos = len(video_paths.get(episode_id, {})) == len(robot.camera_keys)
                if not (have_proprio and have_videos):
                    print(f"WARNING: skipping episode {episode_id} (task {task_name}): "
                          f"not fully found in its shard(s) (proprio={have_proprio}, "
                          f"videos={len(video_paths.get(episode_id, {}))}/{len(robot.camera_keys)})")
                    continue
                try:
                    aligned = _transform_episode(
                        proprio_paths[episode_id], video_paths[episode_id], robot, agibot_cfg, image_size,
                    )
                except (OSError, KeyError, ValueError) as e:
                    print(f"WARNING: skipping episode {episode_id} (task {task_name}): transform failure: {e}")
                    continue
                record = write_episode(v3_root, ep_idx, task_name, plan.task_to_index[task_name], aligned, robot)
                new_records.append(record)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    reused_records = [
        EpisodeRecord(episode_index=e["episode_index"], task=e["task"], length=e["length"])
        for e in plan.reused
    ]
    all_records = reused_records + new_records
    if not all_records:
        raise RuntimeError("no episodes available -- every conversion failed and nothing was reused")

    finalize_dataset(v3_root, all_records, robot, image_size, plan.task_to_index)
    write_conversion_params(v3_root, expected_params)

    idx_to_stable = {ep_idx: f"{task_id}/{episode_id}" for _tn, task_id, episode_id, ep_idx in specs}
    idx_to_stable.update({e["episode_index"]: e["rel_path"] for e in plan.reused})
    write_episode_manifest(v3_root, [
        {
            "rel_path": idx_to_stable[r.episode_index],
            "task": r.task,
            "episode_index": r.episode_index,
            "task_index": plan.task_to_index[r.task],
            "length": r.length,
        }
        for r in all_records
    ])
    return v3_root
