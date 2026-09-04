"""Data-prep entrypoint: discover -> download (capped per task) -> convert
to LeRobot v3, for whichever --dataset-source you pick. Run this before
training/train.py, which assumes the dataset is already prepared and does
no discover/download/convert of its own.

Usage (run from the repo root):
    export HF_TOKEN=hf_...   # only needed for hf:// sources; see --source-uri
    python -m training.prepare_data --tasks arrange_the_flowers box_folding --max-episodes-per-task 300

--dataset-source's choices come from training/data_prep/schemas/*.yaml --
adding a dataset that reuses an existing ingestion strategy (mcap,
hf_lerobot_mirror) is a new schema file, not a new flag here.
"""
from __future__ import annotations

import argparse

from training.data_prep.config import ConvertConfig
from training.data_prep.prepare import is_prepared, prepare_dataset, require_token, resolve_v3_root
from training.data_prep.strategies.registry import available_dataset_sources, get_dataset_source
from training.common.ray_setup import build_runtime_env, connect_ray

DEFAULT_IMAGE_SIZE = (224, 224)  # (H, W) -- matches DataConfig's own default; not yet a CLI flag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True, help="task-name substrings")
    parser.add_argument("--dataset-source", required=True, choices=available_dataset_sources())
    parser.add_argument("--max-episodes-per-task", type=int, default=300)
    parser.add_argument("--source-uri", default=None,
                         help="where the raw dataset lives (default: the chosen --dataset-source's "
                              "own default_source_uri). Only meaningful for the mcap ingestion "
                              "strategy -- hf_lerobot_mirror always uses its schema's full_repo_id.")
    parser.add_argument("--v3-root", default=None,
                         help="where to write the LeRobot v3 dataset -- local path, s3://<bucket>/<prefix>, "
                              "or gs://<bucket>/<prefix> (default: training/lerobot_v3/<dataset_source>/"
                              "<sorted task names>, local). Only local roots are verified against real "
                              "data; s3://gs:// are unverified -- see README")
    parser.add_argument("--reconvert", action="store_true",
                         help="rebuild even if a dataset already exists at --v3-root")
    parser.add_argument("--refresh-listing", action="store_true",
                         help="re-scan the full source tree instead of using the cached "
                              "task/episode listing from a previous run -- mcap strategy only")
    parser.add_argument("--mode", choices=("stream", "download"), default="stream",
                         help="stream (default): read the source over HTTP, nothing local "
                              "but the converted output. download: pull each raw episode file to "
                              "local disk first, then convert -- mcap strategy only")
    parser.add_argument("--sequential-camera-decode", action="store_true",
                         help="decode a episode's cameras one at a time instead of concurrently -- "
                              "slower/episode, lower memory, can mean more episodes convert at once "
                              "-- mcap strategy only")
    parser.add_argument("--max-concurrent", type=int, default=None,
                         help="override the auto-computed number of concurrent episode conversions "
                              "(normally derived from live CPU/memory) -- set this deliberately, not "
                              "casually: too high risks OOM (see README's incident notes) -- mcap "
                              "strategy only")
    args = parser.parse_args()

    source = get_dataset_source(args.dataset_source)
    source_uri = args.source_uri or source.default_source_uri
    v3_root = resolve_v3_root(args.v3_root, args.dataset_source, args.tasks)

    if source.ingestion_strategy == "hf_lerobot_mirror":
        from training.data_prep.strategies.hf_lerobot_mirror import prepare as hf_mirror_prepare

        if not args.reconvert and is_prepared(
            v3_root, args.tasks, args.max_episodes_per_task, source.robot, DEFAULT_IMAGE_SIZE, args.dataset_source,
            revision=source.ingestion_config.revision,
        ):
            print(f"Already prepared at {v3_root} -- pass --reconvert to rebuild it.")
            return
        hf_mirror_prepare(
            source.robot, source.ingestion_config, v3_root,
            args.tasks, args.max_episodes_per_task, DEFAULT_IMAGE_SIZE, args.dataset_source,
        )
        print(f"\nReady: {v3_root}")
        print(f"Train with: python -m training.train --tasks {' '.join(args.tasks)} --v3-root {v3_root}")
        return

    if source.ingestion_strategy == "agibot_hdf5":
        from training.data_prep.strategies.agibot_hdf5 import prepare as agibot_prepare

        if not args.reconvert and is_prepared(
            v3_root, args.tasks, args.max_episodes_per_task, source.robot, DEFAULT_IMAGE_SIZE, args.dataset_source,
            revision=source.ingestion_config.revision,
        ):
            print(f"Already prepared at {v3_root} -- pass --reconvert to rebuild it.")
            return
        token = require_token(source_uri)
        print("\n=== connect to Ray ===")
        connect_ray(build_runtime_env())
        agibot_prepare(
            source.robot, source.ingestion_config, v3_root,
            args.tasks, args.max_episodes_per_task, DEFAULT_IMAGE_SIZE, args.dataset_source,
            source_uri, token, force=args.reconvert,
        )
        print(f"\nReady: {v3_root}")
        print(f"Train with: python -m training.train --tasks {' '.join(args.tasks)} --v3-root {v3_root}")
        return

    if source.ingestion_strategy != "mcap":
        raise SystemExit(
            f"--dataset-source {args.dataset_source!r} uses the {source.ingestion_strategy!r} "
            f"ingestion strategy, which prepare_data.py doesn't know how to run yet "
            f"(see training/data_prep/strategies/{source.ingestion_strategy}.py)."
        )

    convert_cfg = ConvertConfig(
        mode=args.mode,
        parallel_camera_decode=not args.sequential_camera_decode,
        max_concurrent=args.max_concurrent,
    )
    image_size = DEFAULT_IMAGE_SIZE

    if is_prepared(
        v3_root, args.tasks, args.max_episodes_per_task, source.robot, image_size, args.dataset_source,
        revision=source.ingestion_config.revision,
    ) and not args.reconvert:
        print(f"Already prepared at {v3_root} -- pass --reconvert to rebuild it.")
        return

    token = require_token(source_uri)

    print("\n=== connect to Ray ===")
    connect_ray(build_runtime_env())

    from training.data_prep.strategies.mcap import path_to_split_and_task

    mcap_cfg = source.ingestion_config
    prepare_dataset(
        token, args.tasks, args.max_episodes_per_task, source.robot, image_size,
        source_uri, args.dataset_source, v3_root,
        path_to_split_and_task=lambda p: path_to_split_and_task(p, mcap_cfg),
        episode_filename=mcap_cfg.episode_filename,
        decode_and_align_fn=source.decode_and_align, ingestion_cfg=mcap_cfg,
        convert_cfg=convert_cfg,
        refresh_listing=args.refresh_listing, reconvert=args.reconvert,
    )

    print(f"\nReady: {v3_root}")
    print(f"Train with: python -m training.train --tasks {' '.join(args.tasks)} --v3-root {v3_root}")


if __name__ == "__main__":
    main()
