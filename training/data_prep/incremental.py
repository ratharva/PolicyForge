"""Shared incremental-conversion planning: diff a requested set of episodes
against what's already converted (via meta/episode_manifest.json), decide
what's new vs. reused, and keep task_index/episode_index stable across
runs. Extracted from convert_to_lerobot_v3's original inline logic (same
behavior) so both the mcap strategy (via convert.py's
convert_to_lerobot_v3) and the agibot_hdf5 strategy's own prepare() share
one implementation -- this logic already had two real bugs found and fixed
in it (a stale-serve bug, then a global-row-index double-offset corruption,
see docs/customizing-datasets.md's "Incremental conversion" history);
reimplementing it per strategy would risk re-deriving the same class of bug.

The manifest's "rel_path" JSON key name is kept as-is here (not renamed to
something more generic) for backward compatibility with
meta/episode_manifest.json files already on disk -- semantically it's just
"this episode's stable id" now (an MCAP path for mcap, "<task_id>/<episode_id>"
for agibot_hdf5).
"""
from __future__ import annotations

from dataclasses import dataclass

from training.data_prep.lerobot_v3_writer import read_episode_manifest, wipe_dataset


@dataclass
class IncrementalPlan:
    to_convert: list[tuple[str, str]]  # (task, stable_id), not yet converted
    reused: list[dict]  # manifest entries to keep as-is (no re-fetch/re-decode)
    task_to_index: dict[str, int]  # stable across runs -- baked into each episode's data file
    next_episode_index: int  # first free episode_index to assign to new episodes


def plan_incremental_conversion(
    out_root: str, episodes_by_task: dict[str, list[str]], exists: bool, force: bool,
) -> IncrementalPlan:
    """episodes_by_task: {task: [stable_id, ...]} -- the full requested set.

    Wipes out_root (via wipe_dataset) if any previously-converted episode is
    no longer requested (shrinking the request) -- LeRobot v3 requires
    contiguous 0-based episode_index (lerobot_datasource.py enforces this),
    so removing one in place would mean renumbering every later episode's
    files; a full wipe + rebuild is simpler and still correct. Also wipes on
    force=True (--reconvert) if a dataset already exists.
    """
    requested = [
        (task, sid) for task in sorted(episodes_by_task) for sid in episodes_by_task[task]
    ]
    requested_ids = {sid for _, sid in requested}

    manifest = [] if force else read_episode_manifest(out_root)
    removed = [e for e in manifest if e["rel_path"] not in requested_ids]
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

    manifest_by_id = {e["rel_path"]: e for e in manifest}
    to_convert = [(task, sid) for task, sid in requested if sid not in manifest_by_id]
    reused = [manifest_by_id[sid] for _, sid in requested if sid in manifest_by_id]

    if reused:
        print(f"{len(reused)} episode(s) already converted -- reusing (no re-fetch/re-decode).")
    if not to_convert:
        print("Nothing new to convert.")
    else:
        print(f"{len(to_convert)} new episode(s) to convert.")

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

    return IncrementalPlan(
        to_convert=to_convert, reused=reused,
        task_to_index=task_to_index, next_episode_index=next_episode_index,
    )
