"""Convert MCAP episodes into one combined LeRobot v3 dataset root (all
requested tasks in one root, distinguished by task_index).

Source is DataConfig.source_uri -- hf://, s3://, or gs:// -- see
training/data/source.py. Only hf:// has been run against real data.

Two independent, configurable trade-offs (see ConvertConfig in
training/config.py): mode="stream" (default, no local .mcap ever written)
vs. mode="download" (pulls episode.mcap to local disk first); and
parallel_camera_decode (all cameras decode concurrently within one episode
task vs. one after another -- faster per-episode isn't the same as better
throughput if memory is what actually caps max_concurrent).

One Ray task per episode does the expensive part (read + decode + align +
video encode) in parallel; a cheap sequential pass on the driver then
patches global row indices and writes dataset-level metadata (see
lerobot_v3_writer.py). Concurrency is bounded by estimated per-episode
memory use (_estimate_episode_memory_bytes), since each episode task holds
a full episode's decoded video frames in memory at once.
"""
from __future__ import annotations

import concurrent.futures
import itertools

import ray

from training.config import ConvertConfig, DataConfig
from training.data.decode import read_raw_episode, read_raw_episode_streaming
from training.data.episode import align_episode
from training.data.source import download_one
from training.data.lerobot_v3_writer import (
    EpisodeRecord,
    compute_conversion_params,
    dataset_exists,
    finalize_dataset,
    read_conversion_params,
    read_episode_manifest,
    wipe_dataset,
    write_conversion_params,
    write_episode,
    write_episode_manifest,
)
from training.data.video_decode import decode_camera_stream


def _cpus_per_episode(convert_cfg: ConvertConfig) -> int:
    # Camera decode dominates episode time; parallel decode needs one CPU
    # per camera to actually run concurrently rather than just interleave.
    return 3 if convert_cfg.parallel_camera_decode else 1


@ray.remote
def _convert_one(
    source_uri: str, rel_path: str, task: str, episode_index: int, task_index: int,
    cfg: DataConfig, out_root: str, token: str | None, convert_cfg: ConvertConfig,
) -> EpisodeRecord | None:
    try:
        if convert_cfg.mode == "download":
            local_path = download_one(source_uri, rel_path, token)
            raw = read_raw_episode(local_path, cfg)
        else:
            raw = read_raw_episode_streaming(source_uri, rel_path, cfg, token)

        if convert_cfg.parallel_camera_decode:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(raw.cameras) or 1) as pool:
                futures = {
                    cam_key: pool.submit(decode_camera_stream, raw.cameras[cam_key], cfg.image_size)
                    for cam_key in list(raw.cameras)
                }
                for cam_key, fut in futures.items():
                    raw.cameras[cam_key] = fut.result()
        else:
            for cam_key in list(raw.cameras):
                raw.cameras[cam_key] = decode_camera_stream(raw.cameras[cam_key], cfg.image_size)

        aligned = align_episode(raw, cfg)
    except ValueError as e:
        print(f"WARNING: skipping {rel_path} ({task}): decode/align failure: {e}")
        return None
    except (OSError, ConnectionError) as e:
        # One flaky episode shouldn't take down the whole batch; it's just
        # missing from the converted dataset (skipped count reported at the end).
        print(f"WARNING: skipping {rel_path} ({task}): network error: {e}")
        return None
    return write_episode(out_root, episode_index, task, task_index, aligned, cfg)


def _estimate_episode_memory_bytes(
    cfg: DataConfig, convert_cfg: ConvertConfig, max_frames_estimate: int = 6000,
) -> int:
    """Rough per-episode peak-memory estimate, used both as the memory=
    resource hint on each task and to compute a default max_concurrent.
    All cameras' decoded frame lists must coexist by the time align_episode
    runs; parallel_camera_decode additionally keeps that many PyAV decode
    contexts simultaneously active. Not independently measured (only
    wall-time was) -- treat this as a reasoned estimate, not a verified figure.
    """
    h, w = cfg.image_size
    frame_bytes = h * w * 3  # uint8 RGB
    n_cameras = len(cfg.robot.camera_keys)
    one_camera_decoded = frame_bytes * max_frames_estimate  # one camera's finished decoded-frame list

    if convert_cfg.parallel_camera_decode:
        # All n_cameras actively decoding at once, each also duplicated by
        # align_episode's stacked copy: n_cameras * 2x.
        total = one_camera_decoded * 2 * n_cameras
    else:
        # 1 camera actively decoding (2x: list + in-flight buffering) while
        # the other (n_cameras - 1) already-finished cameras' lists sit idle.
        total = one_camera_decoded * (2 + (n_cameras - 1))

    return int(total * 1.3)  # +30% margin for parsing/object overhead


def convert_to_lerobot_v3(
    episodes_by_task: dict[str, list[str]], cfg: DataConfig, out_root: str, token: str | None,
    requested_tasks: list[str], max_episodes_per_task: int,
    source_uri: str,
    convert_cfg: ConvertConfig | None = None,
    force: bool = False,
) -> str:
    """episodes_by_task: {task: [source_uri-relative episode.mcap paths, ...]}
    -- the output of training/data/discover.py's select_episodes(). Whether
    anything gets downloaded to local disk depends on convert_cfg.mode (see
    ConvertConfig in training/config.py). token is only meaningful for
    hf:// source_uri.

    requested_tasks/max_episodes_per_task: the RAW request (CLI substrings,
    the cap you asked for), used to detect a stale conversion -- e.g. asking
    for more episodes/task than a prior conversion of the same task names.
    "meta/info.json exists" alone doesn't catch that.

    Incremental: episodes already present (by HF rel_path, via the episode
    manifest) are reused untouched -- no re-streaming/re-decoding. If any
    previously-converted episode is no longer in the request (shrinking the
    cap, dropping a task, etc.), this falls back to a full wipe + rebuild
    instead of an in-place removal: LeRobot v3 requires contiguous 0-based
    episode_index (lerobot_datasource.py enforces this), so removing one in
    place would mean renumbering and renaming every subsequent episode's
    files. A full rebuild is simpler, still correct, and (via wipe_dataset)
    avoids leaving the removed episodes' files orphaned on disk.
    """
    convert_cfg = convert_cfg or ConvertConfig()
    if convert_cfg.mode not in ("stream", "download"):
        raise ValueError(f"convert_cfg.mode must be 'stream' or 'download', got {convert_cfg.mode!r}")

    expected_params = compute_conversion_params(requested_tasks, max_episodes_per_task, cfg)
    exists = dataset_exists(out_root)

    if exists and not force:
        stored_params = read_conversion_params(out_root)
        if stored_params == expected_params:
            print(f"LeRobot v3 dataset already exists at {out_root}, matching this request "
                  f"-- skipping conversion (pass --reconvert to rebuild it anyway)")
            return out_root
        print(
            f"LeRobot v3 dataset at {out_root} exists but doesn't match this request.\n"
            f"  stored:    {stored_params}\n  requested: {expected_params}"
        )
        # No stored params at all (pre-dates this check, or a prior wipe+rebuild
        # that hasn't run yet) counts as a mismatch too -- everything below is
        # then treated as new.

    requested = [
        (task, rel_path) for task in sorted(episodes_by_task) for rel_path in episodes_by_task[task]
    ]
    requested_rel_paths = {rel_path for _, rel_path in requested}

    manifest = [] if force else read_episode_manifest(out_root)
    removed = [e for e in manifest if e["rel_path"] not in requested_rel_paths]
    if manifest and removed:
        print(
            f"{len(removed)} previously-converted episode(s) are no longer requested -- "
            f"wiping {out_root} and doing a full reconvert."
        )
        wipe_dataset(out_root)
        manifest = []
    elif force and exists:
        print(f"--reconvert: wiping {out_root} and rebuilding from scratch.")
        wipe_dataset(out_root)

    manifest_by_rel_path = {e["rel_path"]: e for e in manifest}
    to_convert = [(task, rp) for task, rp in requested if rp not in manifest_by_rel_path]
    reused = [manifest_by_rel_path[rp] for _, rp in requested if rp in manifest_by_rel_path]

    if reused:
        print(f"{len(reused)} episode(s) already converted -- reusing (no re-stream/re-decode).")
    if not to_convert:
        print("Nothing new to convert.")
    else:
        print(f"Converting {len(to_convert)} new episode(s) ...")

    # task_index must stay stable across runs -- it's baked permanently into
    # each episode's data file. Existing tasks keep their assigned index;
    # only genuinely new tasks get a new one, appended after the current max.
    existing_task_index = {e["task"]: e["task_index"] for e in manifest}
    next_task_index = max(existing_task_index.values(), default=-1) + 1
    task_to_index = dict(existing_task_index)
    for task in sorted(episodes_by_task):
        if task not in task_to_index:
            task_to_index[task] = next_task_index
            next_task_index += 1

    # Same for episode_index: existing episodes keep theirs, new ones append.
    next_episode_index = max((e["episode_index"] for e in manifest), default=-1) + 1
    specs = []  # (rel_path, task, episode_index, task_index)
    for task, rel_path in to_convert:
        specs.append((rel_path, task, next_episode_index, task_to_index[task]))
        next_episode_index += 1

    new_records: list[EpisodeRecord] = []
    if specs:
        cpus_per_episode = _cpus_per_episode(convert_cfg)
        per_episode_memory = _estimate_episode_memory_bytes(cfg, convert_cfg)
        max_concurrent = convert_cfg.max_concurrent
        if max_concurrent is None:
            resources = ray.cluster_resources()
            # Each episode task may decode its cameras concurrently on its
            # own threads (see _cpus_per_episode) -- account for that real
            # CPU usage, not just 1 core/task.
            cpu_bound = max(1, int(resources.get("CPU", 1)) // cpus_per_episode)
            available_memory = resources.get("memory", 0)
            mem_bound = max(1, int(available_memory // per_episode_memory)) if available_memory else cpu_bound
            max_concurrent = max(1, min(cpu_bound, mem_bound))

        print(
            f"{convert_cfg.mode}-mode from {source_uri} (up to {max_concurrent} concurrent x "
            f"{cpus_per_episode} CPUs, ~{per_episode_memory / 1e9:.1f}GB budgeted/episode, "
            f"{'parallel' if convert_cfg.parallel_camera_decode else 'sequential'} camera decode) ..."
        )

        pending: dict[ray.ObjectRef, tuple] = {}

        def _submit(spec: tuple) -> None:
            rel_path, task, ep_idx, task_idx = spec
            ref = _convert_one.options(num_cpus=cpus_per_episode, memory=per_episode_memory).remote(
                source_uri, rel_path, task, ep_idx, task_idx, cfg, out_root, token, convert_cfg,
            )
            pending[ref] = spec

        specs_iter = iter(specs)
        for spec in itertools.islice(specs_iter, max_concurrent):
            _submit(spec)

        results = []
        while pending:
            done, _ = ray.wait(list(pending.keys()), num_returns=1, timeout=30)
            if not done:
                print(f"  {len(results)}/{len(specs)} converted (waiting on stragglers)")
                continue
            for ref in done:
                pending.pop(ref)
                results.append(ray.get(ref))
                next_spec = next(specs_iter, None)
                if next_spec is not None:
                    _submit(next_spec)
            print(f"  {len(results)}/{len(specs)} new episodes converted")

        new_records = [r for r in results if r is not None]
        skipped = len(specs) - len(new_records)
        if skipped:
            print(f"  {skipped} episode(s) skipped (decode/align/network failure -- see WARNINGs above)")

    reused_records = [
        EpisodeRecord(episode_index=e["episode_index"], task=e["task"], length=e["length"]) for e in reused
    ]
    all_records = reused_records + new_records
    if not all_records:
        raise RuntimeError("no episodes available -- every conversion failed and nothing was reused")

    finalize_dataset(out_root, all_records, cfg, task_to_index)
    write_conversion_params(out_root, expected_params)

    # rel_path is only known here (not on EpisodeRecord); rebuild the manifest
    # by joining new_records back to their spec via episode_index, which was
    # assigned per-spec before submission and is therefore a stable join key
    # regardless of the order Ray tasks actually complete in.
    ep_idx_to_rel_path = {ep_idx: rel_path for rel_path, _task, ep_idx, _task_idx in specs}
    ep_idx_to_rel_path.update({e["episode_index"]: e["rel_path"] for e in reused})
    write_episode_manifest(out_root, [
        {
            "rel_path": ep_idx_to_rel_path[r.episode_index],
            "task": r.task,
            "episode_index": r.episode_index,
            "task_index": task_to_index[r.task],
            "length": r.length,
        }
        for r in all_records
    ])
    return out_root
