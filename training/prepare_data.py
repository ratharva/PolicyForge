"""Standalone data-prep entrypoint: discover -> download (capped per task) ->
convert to LeRobot v3, so training/train.py can start straight from a ready
dataset instead of redoing discover/download/convert on every invocation.

Usage (run from the repo root):
    export HF_TOKEN=hf_...   # only needed for the default hf:// source; see --source-uri
    python -m training.prepare_data --tasks arrange_the_flowers box_folding --max-episodes-per-task 300

Needs Ray for the parallel conversion step (attaches to an existing cluster
if one's running, otherwise starts a local instance) but no GPU.
"""
from __future__ import annotations

import argparse

from training.config import ConvertConfig, DataConfig
from training.data.prepare import is_prepared, prepare_dataset, require_token, resolve_v3_root
from training.ray_setup import build_runtime_env, connect_ray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True, help="task-name substrings")
    parser.add_argument("--max-episodes-per-task", type=int, default=300)
    parser.add_argument("--source-uri", default=None,
                         help="where the raw MCAP dataset lives: hf://datasets/<repo_id>, "
                              "s3://<bucket>/<prefix>, or gs://<bucket>/<prefix> (default: "
                              "config.py's DataConfig.source_uri). Only hf:// is verified against "
                              "real data; s3://gs:// are unverified -- see README")
    parser.add_argument("--v3-root", default=None,
                         help="where to write the LeRobot v3 dataset -- local path, s3://<bucket>/<prefix>, "
                              "or gs://<bucket>/<prefix> (default: training/lerobot_v3/<sorted task names>, "
                              "local). Only local roots are verified against real data; s3://gs:// are "
                              "unverified -- see README")
    parser.add_argument("--reconvert", action="store_true",
                         help="rebuild even if a dataset already exists at --v3-root")
    parser.add_argument("--refresh-listing", action="store_true",
                         help="re-scan the full source tree instead of using the cached "
                              "task/episode listing from a previous run")
    parser.add_argument("--mode", choices=("stream", "download"), default="stream",
                         help="stream (default): read MCAP directly from --source-uri, nothing local "
                              "but the converted output. download: pull each episode.mcap to local "
                              "disk first, then convert")
    parser.add_argument("--sequential-camera-decode", action="store_true",
                         help="decode a episode's 3 cameras one after another instead of concurrently -- "
                              "slower per episode (~172s vs ~83s measured) but lower memory, which can "
                              "mean MORE episodes convert concurrently and better aggregate throughput if "
                              "you're memory-bound (see the --max-concurrent print at conversion start)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                         help="override the auto-computed number of concurrent episode conversions "
                              "(normally derived from live CPU/memory) -- set this deliberately, not "
                              "casually: too high risks OOM (see README's incident notes)")
    args = parser.parse_args()

    v3_root = resolve_v3_root(args.v3_root, args.tasks)
    data_cfg = DataConfig()
    if args.source_uri:
        data_cfg.source_uri = args.source_uri
    convert_cfg = ConvertConfig(
        mode=args.mode,
        parallel_camera_decode=not args.sequential_camera_decode,
        max_concurrent=args.max_concurrent,
    )

    if is_prepared(v3_root, args.tasks, args.max_episodes_per_task, data_cfg) and not args.reconvert:
        print(f"Already prepared at {v3_root} -- pass --reconvert to rebuild it.")
        return

    token = require_token(data_cfg.source_uri)

    print("\n=== connect to Ray ===")
    connect_ray(build_runtime_env())

    prepare_dataset(
        token, args.tasks, args.max_episodes_per_task, data_cfg, v3_root,
        convert_cfg=convert_cfg,
        refresh_listing=args.refresh_listing, reconvert=args.reconvert,
    )

    print(f"\nReady: {v3_root}")
    print(f"Train with: python -m training.train --tasks {' '.join(args.tasks)} --v3-root {v3_root}")


if __name__ == "__main__":
    main()
