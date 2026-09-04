"""Discover real task directories and download up to `max_episodes_per_task`
episodes for each requested task, from whatever backend a dataset's
source_uri points at (HF Hub, S3, or GCS -- see training/data_prep/source.py).

Directory layout and episode-file matching come from the caller (a schema's
mcap.path_prefix/episode_filename today, via strategies/mcap.py) -- not
hardcoded here, so this stays usable by any future dataset that's also
task-organized on HF/S3/GCS with one file per episode, regardless of that
file's format.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Callable

from training.data_prep.source import download_one, is_hf_uri, open_fs, parse_hf_uri

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")


def list_episodes_by_task(
    token: str | None, source_uri: str, dataset_source: str,
    path_to_split_and_task: Callable[[str], tuple[str, str]],
    episode_filename: str,
    cache_path: str | None = None, refresh: bool = False,
) -> dict[str, list[str]]:
    """Full source listing, grouped by task. Walks the entire tree, so this
    is the slowest step -- cached to `cache_path` (default:
    training/.cache/<dataset_source>_listing.json) so repeat runs skip
    straight to selection/download. Pass refresh=True (--refresh-listing)
    to force a re-scan.
    """
    if cache_path is None:
        cache_path = os.path.join(DEFAULT_CACHE_DIR, f"{dataset_source}_listing.json")
    if cache_path and not refresh and os.path.exists(cache_path):
        print(f"Using cached file listing from {cache_path} (pass --refresh-listing to re-scan)")
        with open(cache_path) as f:
            by_task = json.load(f)
        total = sum(len(v) for v in by_task.values())
        print(f"  {total:,} episodes across {len(by_task)} tasks (from cache)")
        return by_task

    print(f"Listing files in {source_uri} (walks the whole tree) ...")
    t0 = time.perf_counter()
    if is_hf_uri(source_uri):
        # HfApi.list_repo_files is faster here than fsspec's generic fs.find().
        from huggingface_hub import HfApi
        repo_type, repo_id = parse_hf_uri(source_uri)
        api = HfApi(token=token)
        all_files = list(api.list_repo_files(repo_id, repo_type=repo_type))
    else:
        fs, fs_root = open_fs(source_uri)
        all_files = [p[len(fs_root):].lstrip("/") for p in fs.find(fs_root)]
    print(f"  {len(all_files):,} files listed in {time.perf_counter() - t0:.1f}s")

    episode_files = [f for f in all_files if f.endswith(episode_filename)]
    by_task: dict[str, list[str]] = defaultdict(list)
    for p in episode_files:
        _split, task = path_to_split_and_task(p)
        by_task[task].append(p)
    print(f"  {len(episode_files):,} episodes across {len(by_task)} tasks")
    by_task = dict(by_task)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(by_task, f)
        print(f"  cached listing -> {cache_path}")

    return by_task


def select_episodes(
    by_task: dict[str, list[str]], tasks: list[str], max_per_task: int
) -> dict[str, list[str]]:
    """Resolve requested task-name substrings to real task directories and
    cap each to max_per_task episodes (fewer if that many don't exist)."""
    selected: dict[str, list[str]] = {}
    for requested in tasks:
        matches = [t for t in by_task if requested in t]
        if not matches:
            raise ValueError(
                f"No task in the dataset contains {requested!r}. Run "
                "`python -m training.data_prep.discover` first to see real task names."
            )
        if len(matches) > 1:
            print(f"  WARNING: {requested!r} matches {len(matches)} tasks {matches}; using {matches[0]!r}")
        task = matches[0]
        available = by_task[task]
        n = min(max_per_task, len(available))
        note = "" if n == max_per_task else f" (only {len(available)} available, < requested {max_per_task})"
        print(f"  {task}: using {n} episodes{note}")
        selected[task] = available[:n]
    return selected


def download_episodes(
    token: str | None, source_uri: str,
    selected: dict[str, list[str]], cache_dir: str | None = None,
) -> dict[str, list[str]]:
    """Download every selected episode file. Returns {task: [local_path, ...]}."""
    local: dict[str, list[str]] = {}
    total = sum(len(v) for v in selected.values())
    done = 0
    for task, rel_paths in selected.items():
        local[task] = []
        for rel_path in rel_paths:
            done += 1
            print(f"[{done}/{total}] downloading {rel_path}")
            local_path = download_one(source_uri, rel_path, token, cache_dir)
            local[task].append(local_path)
    return local


if __name__ == "__main__":
    # `python -m training.data_prep.discover --dataset-source abc130k` lists
    # tasks, no download. Only meaningful for a dataset whose ingestion
    # strategy uses this module (mcap today).
    import argparse
    import sys

    from training.data_prep.source import requires_hf_token
    from training.data_prep.strategies.mcap import path_to_split_and_task

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", default="abc130k")
    args = parser.parse_args()

    from training.data_prep.strategies.registry import get_dataset_source

    source = get_dataset_source(args.dataset_source)
    if source.ingestion_strategy != "mcap":
        sys.exit(
            f"--dataset-source {args.dataset_source!r} uses the {source.ingestion_strategy!r} "
            f"ingestion strategy, which doesn't discover per-episode files this way."
        )
    tok = os.environ.get("HF_TOKEN")
    if requires_hf_token(source.default_source_uri) and not tok:
        sys.exit("HF_TOKEN not set.")
    by_task = list_episodes_by_task(
        tok, source.default_source_uri, args.dataset_source,
        path_to_split_and_task=lambda p: path_to_split_and_task(p, source.ingestion_config),
        episode_filename=source.ingestion_config.episode_filename,
    )
    for task, files in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        print(f"{len(files):6d}  {task}")
