"""Discover -> convert pipeline for "flat task-organized episode files"
datasets (the mcap ingestion strategy today; any future dataset sharing
that shape reuses this unchanged). Run standalone via
training/prepare_data.py, which dispatches to this for
ingestion_strategy == "mcap" and to a different path per other strategy
(training/data_prep/strategies/hf_lerobot_mirror.py's own prepare() for
"hf_lerobot_mirror"; "agibot_hdf5" isn't wired into prepare_data.py yet --
see that strategy module's docstring).

training/train.py never calls prepare_dataset() itself -- it only reads
has_lerobot_v3_data()/resolve_v3_root() to find data prepare_data.py
already converted, and errors out if none exists.

Source is DataConfig.source_uri (hf://, s3://, or gs:// -- see
training/data_prep/source.py). No download step by default: episodes stream
directly into the LeRobot v3 converter (training/data_prep/convert.py).
"""
from __future__ import annotations

import os
import sys
from typing import Callable

from training.common.robots import RobotSchema
from training.data_prep.config import ConvertConfig
from training.data_prep.convert import convert_to_lerobot_v3
from training.data_prep.discover import list_episodes_by_task, select_episodes
from training.data_prep.lerobot_v3_writer import compute_conversion_params, dataset_exists, read_conversion_params
from training.data_prep.source import requires_hf_token


def require_token(source_uri: str) -> str | None:
    """None for s3://gs:// sources (ambient credentials, no token concept).
    hf:// sources need HF_TOKEN if the repo is gated -- exits with a clear
    message if it's not set."""
    if not requires_hf_token(source_uri):
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            f"HF_TOKEN is not set, and {source_uri!r} is an hf:// source. If it's a gated "
            "repo, accept its terms on huggingface.co first, then "
            "`export HF_TOKEN=hf_xxx` before running this script."
        )
    return token


def default_v3_root(dataset_source: str, tasks: list[str]) -> str:
    # Deterministic (not timestamped) so re-running on the same task set
    # finds -- and reuses -- the same converted dataset. Namespaced by
    # dataset_source so two different datasets requesting overlapping task
    # names don't collide on the same directory.
    return os.path.join("training", "lerobot_v3", dataset_source, "-".join(sorted(tasks)))


def resolve_v3_root(v3_root: str | None, dataset_source: str, tasks: list[str]) -> str:
    """CLI --v3-root, or the deterministic per-dataset-source-per-task-set
    default. A local path is made absolute (relative to wherever the script
    was run from); an s3://gs:// URI is left untouched -- os.path.abspath
    would otherwise mangle it (treating "s3://bucket/x" as a relative local
    path)."""
    root = v3_root or default_v3_root(dataset_source, tasks)
    if "://" in root:
        return root
    return os.path.abspath(root)


def is_prepared(
    v3_root: str, tasks: list[str], max_episodes_per_task: int,
    robot: RobotSchema, image_size: tuple[int, int], dataset_source: str, revision: str | None = None,
) -> bool:
    """True only if v3_root exists AND was converted for this exact
    tasks/max_episodes_per_task/schema request -- not just "some dataset
    exists here". Catches asking for a different episode cap (or task list,
    or camera/tick settings) against an already-converted directory."""
    if not dataset_exists(v3_root):
        return False
    stored = read_conversion_params(v3_root)
    expected = compute_conversion_params(tasks, max_episodes_per_task, robot, image_size, dataset_source, revision)
    return stored == expected


def has_lerobot_v3_data(v3_root: str) -> bool:
    """True if *any* valid-looking LeRobot v3 dataset exists at v3_root --
    regardless of whether it matches the current request, or was even built
    by this pipeline. What train.py checks before it will run -- it trusts
    whatever's there as-is rather than re-verifying against is_prepared()'s
    exact tasks/max_episodes_per_task/schema match."""
    return dataset_exists(v3_root)


def prepare_dataset(
    token: str | None, tasks: list[str], max_episodes_per_task: int,
    robot: RobotSchema, image_size: tuple[int, int], source_uri: str, dataset_source: str,
    v3_root: str, path_to_split_and_task: Callable[[str], tuple[str, str]], episode_filename: str,
    decode_and_align_fn: Callable[..., dict], ingestion_cfg: object,
    convert_cfg: ConvertConfig | None = None,
    refresh_listing: bool = False, reconvert: bool = False,
) -> str:
    """Discover real task directories, then convert up to
    max_episodes_per_task per task into the LeRobot v3 dataset at v3_root.
    Requires an initialized Ray (convert_to_lerobot_v3 fans out per-episode
    conversion as Ray tasks)."""
    print("\n=== discover ===")
    by_task = list_episodes_by_task(
        token, source_uri, dataset_source,
        path_to_split_and_task=path_to_split_and_task, episode_filename=episode_filename,
        refresh=refresh_listing, revision=getattr(ingestion_cfg, "revision", None),
    )
    selected = select_episodes(by_task, tasks, max_episodes_per_task)

    print("\n=== convert -> LeRobot v3 ===")
    convert_to_lerobot_v3(
        selected, robot, image_size, v3_root, token,
        requested_tasks=tasks, max_episodes_per_task=max_episodes_per_task,
        source_uri=source_uri, dataset_source=dataset_source,
        decode_and_align_fn=decode_and_align_fn, ingestion_cfg=ingestion_cfg,
        convert_cfg=convert_cfg, force=reconvert,
    )
    return v3_root
