"""Convert episodes into one combined LeRobot v3 dataset root (all
requested tasks in one root, distinguished by task_index). Orchestration
(concurrency, incremental manifest diffing, finalize_dataset) is dataset-
agnostic -- only the actual decode+align step is per ingestion strategy,
injected as decode_and_align_fn (see training/data_prep/strategies/mcap.py's
decode_and_align, the only one this project has exercised against real data
so far; strategies/agibot_hdf5.py will use this the same way once its
schema is filled in -- see that module's docstring).

Source is DataConfig.source_uri -- hf://, s3://, or gs:// -- see
training/data_prep/source.py. Only hf:// has been run against real data.

Two independent, configurable trade-offs (see ConvertConfig in
training/data_prep/config.py): mode="stream" (default, no local file ever
written) vs. mode="download" (pulls the raw episode file to local disk
first); and parallel_camera_decode (all cameras decode concurrently within
one episode task vs. one after another -- faster per-episode isn't the same
as better throughput if memory is what actually caps max_concurrent).

One Ray task per episode does the expensive part (decode_and_align_fn) in
parallel; a cheap sequential pass on the driver then patches global row
indices and writes dataset-level metadata (see lerobot_v3_writer.py).
Concurrency is bounded by estimated per-episode memory use
(_estimate_episode_memory_bytes), since each episode task holds a full
episode's decoded video frames in memory at once.
"""
from __future__ import annotations

import itertools
from typing import Callable

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
    renumber_episode,
    write_conversion_params,
    write_episode,
    write_episode_manifest,
)


def _cpus_per_episode(convert_cfg: ConvertConfig) -> int:
    # Camera decode dominates episode time; parallel decode needs one CPU
    # per camera to actually run concurrently rather than just interleave.
    return 3 if convert_cfg.parallel_camera_decode else 1


@ray.remote
def _convert_one(
    decode_and_align_fn: Callable[..., dict], source_uri: str, rel_path: str, task: str,
    episode_index: int, task_index: int, robot: RobotSchema, image_size: tuple[int, int],
    out_root: str, token: str | None, convert_cfg: ConvertConfig, ingestion_cfg: object,
) -> EpisodeRecord | None:
    try:
        aligned = decode_and_align_fn(source_uri, rel_path, robot, ingestion_cfg, token, convert_cfg, image_size)
    except ValueError as e:
        print(f"WARNING: skipping {rel_path} ({task}): decode/align failure: {e}")
        return None
    except (OSError, ConnectionError) as e:
        # One flaky episode shouldn't take down the whole batch; it's just
        # missing from the converted dataset (skipped count reported at the end).
        print(f"WARNING: skipping {rel_path} ({task}): network error: {e}")
        return None
    return write_episode(out_root, episode_index, task, task_index, aligned, robot)


def _estimate_episode_memory_bytes(
    robot: RobotSchema, image_size: tuple[int, int], convert_cfg: ConvertConfig,
    max_frames_estimate: int = 6000,
) -> int:
    """Rough per-episode peak-memory estimate, used both as the memory=
    resource hint on each task and to compute a default max_concurrent.
    All cameras' decoded frame lists must coexist by the time
    decode_and_align_fn returns; parallel_camera_decode additionally keeps
    that many decode contexts simultaneously active (for strategies that
    decode video, e.g. mcap -- a strategy without per-camera video decode
    at all would see this as a conservative overestimate, which is fine).
    Not independently measured (only wall-time was) -- treat this as a
    reasoned estimate, not a verified figure.
    """
    h, w = image_size
    frame_bytes = h * w * 3  # uint8 RGB
    n_cameras = len(robot.camera_keys)
    one_camera_decoded = frame_bytes * max_frames_estimate  # one camera's finished decoded-frame list

    if convert_cfg.parallel_camera_decode:
        # All n_cameras actively decoding at once, each also duplicated by
        # the aligned stacked copy: n_cameras * 2x.
        total = one_camera_decoded * 2 * n_cameras
    else:
        # 1 camera actively decoding (2x: list + in-flight buffering) while
        # the other (n_cameras - 1) already-finished cameras' lists sit idle.
        total = one_camera_decoded * (2 + (n_cameras - 1))

    return int(total * 1.3)  # +30% margin for parsing/object overhead


def convert_to_lerobot_v3(
    episodes_by_task: dict[str, list[str]], robot: RobotSchema, image_size: tuple[int, int],
    out_root: str, token: str | None,
    requested_tasks: list[str], max_episodes_per_task: int,
    source_uri: str, dataset_source: str,
    decode_and_align_fn: Callable[..., dict], ingestion_cfg: object,
    convert_cfg: ConvertConfig | None = None,
    force: bool = False,
) -> str:
    """episodes_by_task: {task: [source_uri-relative episode paths, ...]}
    -- the output of training/data_prep/discover.py's select_episodes().
    Whether anything gets downloaded to local disk depends on
    convert_cfg.mode (see ConvertConfig). token is only meaningful for hf://
    source_uri. decode_and_align_fn/ingestion_cfg come from whichever
    ingestion strategy this dataset_source uses (see strategies/registry.py)
    -- e.g. strategies/mcap.py's decode_and_align + McapIngestionConfig.

    requested_tasks/max_episodes_per_task: the RAW request (CLI substrings,
    the cap you asked for), used to detect a stale conversion -- e.g. asking
    for more episodes/task than a prior conversion of the same task names.
    "meta/info.json exists" alone doesn't catch that.

    Incremental: episodes already present (by rel_path, via the episode
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

    expected_params = compute_conversion_params(
        requested_tasks, max_episodes_per_task, robot, image_size, dataset_source,
        revision=getattr(ingestion_cfg, "revision", None),
    )
    exists = dataset_exists(out_root)

    if exists and not force:
        stored_params = read_conversion_params(out_root)
        if stored_params == expected_params:
            print(f"LeRobot v3 dataset already exists at {out_root}, matching this request "
                  f"-- skipping conversion (pass --reconvert to rebuild it anyway)")
            return out_root
        print(
            f"LeRobot v3 dataset at {out_root} exists but doesn't match this request -- forcing a full rebuild.\n"
            f"  stored:    {stored_params}\n  requested: {expected_params}"
        )
        force = True  # a schema/param mismatch must not be reused via incremental planning below

    plan = plan_incremental_conversion(out_root, episodes_by_task, exists=exists, force=force)
    to_convert, reused, task_to_index = plan.to_convert, plan.reused, plan.task_to_index

    specs = []  # (rel_path, task, episode_index, task_index)
    next_episode_index = plan.next_episode_index
    for task, rel_path in to_convert:
        specs.append((rel_path, task, next_episode_index, task_to_index[task]))
        next_episode_index += 1

    new_records: list[EpisodeRecord] = []
    if specs:
        cpus_per_episode = _cpus_per_episode(convert_cfg)
        per_episode_memory = _estimate_episode_memory_bytes(robot, image_size, convert_cfg)
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
                decode_and_align_fn, source_uri, rel_path, task, ep_idx, task_idx,
                robot, image_size, out_root, token, convert_cfg, ingestion_cfg,
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

    # rel_path lookup by each spec's ORIGINAL (pre-assigned, possibly gapped)
    # index -- built before any compaction below.
    old_ep_idx_to_rel_path = {ep_idx: rel_path for rel_path, _task, ep_idx, _task_idx in specs}

    # A skipped episode leaves a gap in episode_index (Ray tasks run in
    # parallel, so a later-numbered episode can finish and write its files
    # before an earlier-numbered one fails) -- lerobot_datasource.py requires
    # strictly contiguous 0-based indices, so compact the survivors back to a
    # contiguous range here, renaming/repatching their already-written files
    # to match (renumber_episode), instead of writing a dataset with holes
    # that later fails to even open.
    new_records.sort(key=lambda r: r.episode_index)
    new_ep_idx_to_rel_path: dict[int, str] = {}
    next_compact_index = plan.next_episode_index
    for rec in new_records:
        old_index = rec.episode_index
        renumber_episode(out_root, old_index, next_compact_index, robot.camera_keys)
        new_ep_idx_to_rel_path[next_compact_index] = old_ep_idx_to_rel_path[old_index]
        rec.episode_index = next_compact_index
        next_compact_index += 1

    reused_records = [
        EpisodeRecord(
            episode_index=e["episode_index"], task=e["task"], length=e["length"],
            # Mirrors agibot_hdf5.py's reused-record fix (mcap has no depth cameras today).
            depth_shapes={k: tuple(v) for k, v in e.get("depth_shapes", {}).items()},
            depth_dtypes=e.get("depth_dtypes", {}),
        )
        for e in reused
    ]
    all_records = reused_records + new_records
    if not all_records:
        raise RuntimeError("no episodes available -- every conversion failed and nothing was reused")

    finalize_dataset(out_root, all_records, robot, image_size, task_to_index)
    write_conversion_params(out_root, expected_params)

    # rel_path is only known here (not on EpisodeRecord); rebuild the manifest
    # by joining records back to their rel_path via the (now-compacted) index.
    ep_idx_to_rel_path = dict(new_ep_idx_to_rel_path)
    ep_idx_to_rel_path.update({e["episode_index"]: e["rel_path"] for e in reused})
    write_episode_manifest(out_root, [
        {
            "rel_path": ep_idx_to_rel_path[r.episode_index],
            "task": r.task,
            "episode_index": r.episode_index,
            "task_index": task_to_index[r.task],
            "length": r.length,
            "depth_shapes": r.depth_shapes,
            "depth_dtypes": r.depth_dtypes,
        }
        for r in all_records
    ])
    return out_root
