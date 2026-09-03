"""Shared discover -> convert pipeline.

Used by both training/prepare_data.py (standalone, run ahead of time) and
training/train.py (inline, only when no prepared dataset exists yet at
--v3-root). Keeping this in one place means the two entrypoints can't drift
out of sync on what "prepare the data" actually does.

Source is DataConfig.source_uri (hf://, s3://, or gs:// -- see
training/data/source.py). No download step by default: episodes stream
directly into the LeRobot v3 converter (training/data/convert.py).
"""
from __future__ import annotations

import os
import sys

from training.config import ConvertConfig, DataConfig
from training.data.convert import convert_to_lerobot_v3
from training.data.discover import list_episodes_by_task, select_episodes
from training.data.lerobot_v3_writer import compute_conversion_params, dataset_exists, read_conversion_params
from training.data.source import requires_hf_token


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


def default_v3_root(tasks: list[str]) -> str:
    # Deterministic (not timestamped) so re-running on the same task set
    # finds -- and reuses -- the same converted dataset.
    return os.path.join("training", "lerobot_v3", "-".join(sorted(tasks)))


def resolve_v3_root(v3_root: str | None, tasks: list[str]) -> str:
    """CLI --v3-root, or the deterministic per-task-set default. A local path
    is made absolute (relative to wherever the script was run from); an
    s3://gs:// URI is left untouched -- os.path.abspath would otherwise
    mangle it (treating "s3://bucket/x" as a relative local path)."""
    root = v3_root or default_v3_root(tasks)
    if "://" in root:
        return root
    return os.path.abspath(root)


def is_prepared(v3_root: str, tasks: list[str], max_episodes_per_task: int, cfg: DataConfig) -> bool:
    """True only if v3_root exists AND was converted for this exact
    tasks/max_episodes_per_task/schema request -- not just "some dataset
    exists here". Catches asking for a different episode cap (or task list,
    or camera/tick settings) against an already-converted directory."""
    if not dataset_exists(v3_root):
        return False
    stored = read_conversion_params(v3_root)
    expected = compute_conversion_params(tasks, max_episodes_per_task, cfg)
    return stored == expected


def has_lerobot_v3_data(v3_root: str) -> bool:
    """True if *any* valid-looking LeRobot v3 dataset exists at v3_root --
    regardless of whether it matches the current request, or was even built
    by this pipeline. For --use-existing-v3: point straight at a LeRobot v3
    dataset from any source and skip discover/convert entirely, trusting it
    as-is. Deliberately looser than is_prepared(), which requires an exact
    match to a conversion THIS pipeline tracked itself.
    """
    return dataset_exists(v3_root)


def prepare_dataset(
    token: str | None, tasks: list[str], max_episodes_per_task: int,
    data_cfg: DataConfig, v3_root: str,
    convert_cfg: ConvertConfig | None = None,
    refresh_listing: bool = False, reconvert: bool = False,
) -> str:
    """Discover real task directories, then convert up to
    max_episodes_per_task per task into the LeRobot v3 dataset at v3_root.
    Requires an initialized Ray (convert_to_lerobot_v3 fans out per-episode
    conversion as Ray tasks)."""
    print("\n=== discover ===")
    by_task = list_episodes_by_task(
        token, data_cfg.source_uri,
        path_to_split_and_task=data_cfg.robot.path_to_split_and_task,
        refresh=refresh_listing,
    )
    selected = select_episodes(by_task, tasks, max_episodes_per_task)

    print("\n=== convert MCAP -> LeRobot v3 ===")
    convert_to_lerobot_v3(
        selected, data_cfg, v3_root, token,
        requested_tasks=tasks, max_episodes_per_task=max_episodes_per_task,
        source_uri=data_cfg.source_uri,
        convert_cfg=convert_cfg, force=reconvert,
    )
    return v3_root
