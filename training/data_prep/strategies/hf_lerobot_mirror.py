"""The "hf_lerobot_mirror" ingestion strategy: for a dataset that already
has a LeRobot-format mirror on the HF Hub (DROID today; OXE/RoboCasa/LIBERO/
GR00T variants are increasingly common too) -- download it and reuse
lerobot's own official conversion tooling instead of writing a decoder.

Does NOT go through convert_to_lerobot_v3/write_episode at all (unlike
mcap/agibot_hdf5) -- lerobot's own convert_dataset() already produces a real
v3 root; this module's job is just to call it correctly and then stamp our
own conversion_params.json into the result so train.py's metadata-driven
--dataset-source resolution works on it like any other prepared dataset.

Verified directly against the installed lerobot package (not assumed from
docs): convert_dataset_v21_to_v30.convert_dataset(repo_id, root, push_to_hub)
first checks whether repo_id already has a v3.0 tag on the hub (fast path,
no conversion), and validate_local_dataset_version() hard-fails for
anything that isn't exactly codebase_version "v2.1" -- see
training/data_prep/schemas/droid.yaml's comment for why full_repo_id points
at cadene/droid_1.0.1 (real v2.1) and not IPEC-COMMUNITY/droid_lerobot
(real v2.0, unsupported by this script).
"""
from __future__ import annotations

from dataclasses import dataclass

from training.common.robots import RobotSchema
from training.data_prep.lerobot_v3_writer import compute_conversion_params, write_conversion_params


@dataclass
class HfLerobotMirrorIngestionConfig:
    full_repo_id: str
    source_format_version: str
    camera_key_map: dict[str, str]
    read_path_smoke_test_repo_id: str | None = None
    revision: str | None = None


def build_ingestion_config(spec: dict) -> HfLerobotMirrorIngestionConfig:
    m = spec["hf_lerobot_mirror"]
    return HfLerobotMirrorIngestionConfig(
        full_repo_id=m["full_repo_id"],
        source_format_version=m["source_format_version"],
        camera_key_map=dict(m["camera_key_map"]),
        read_path_smoke_test_repo_id=m.get("read_path_smoke_test_repo_id"),
        revision=spec.get("revision"),
    )


def prepare(
    robot: RobotSchema, ingestion_cfg: HfLerobotMirrorIngestionConfig, v3_root: str,
    tasks: list[str], max_episodes_per_task: int, image_size: tuple[int, int],
    dataset_source: str, repo_id: str | None = None,
) -> str:
    """Downloads (if needed) + converts ingestion_cfg.full_repo_id (or an
    explicit repo_id override, e.g. for the read-path smoke test) to a v3
    root at v3_root, via lerobot's own official migration script.

    tasks/max_episodes_per_task are NOT applied as a filter here -- this
    strategy downloads/converts the source repo as a whole. Real per-source
    task-filtering semantics for a monolithic mirror like this one (e.g.
    subsetting cadene/droid_1.0.1's ~95k episodes by matching
    meta/tasks.parquet strings) are real, unresolved future work -- flagged
    in the plan, not implemented here to avoid guessing at a shape that
    hasn't been checked against real post-migration output yet.
    """
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    target_repo_id = repo_id or ingestion_cfg.full_repo_id
    convert_dataset(repo_id=target_repo_id, branch=ingestion_cfg.revision, root=v3_root, push_to_hub=False)

    params = compute_conversion_params(
        tasks, max_episodes_per_task, robot, image_size, dataset_source, revision=ingestion_cfg.revision,
    )
    write_conversion_params(v3_root, params)
    return v3_root
