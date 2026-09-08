"""Loads a training/data_prep/schemas/<name>.yaml file into a
(RobotSchema, ingestion_strategy_name, strategy_config_dict) triple.

Plain dict-driven parsing, matching this project's existing style (no new
validation-library dependency) -- a malformed/missing field raises a plain
KeyError/ValueError with the offending file name, not a silent wrong value.
"""
from __future__ import annotations

import glob
import os

import yaml

from training.common.robots import RobotSchema

SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), "schemas")


def available_dataset_sources() -> list[str]:
    """Every schema file's dataset_source, in file order -- what
    --dataset-source's valid choices are derived from. Adding a dataset
    that reuses an existing ingestion_strategy is just a new file here."""
    names = []
    for path in sorted(glob.glob(os.path.join(SCHEMAS_DIR, "*.yaml"))):
        with open(path) as f:
            names.append(yaml.safe_load(f)["dataset_source"])
    return names


def load_schema(dataset_source: str) -> dict:
    path = os.path.join(SCHEMAS_DIR, f"{dataset_source}.yaml")
    if not os.path.exists(path):
        raise ValueError(
            f"no schema file for dataset_source {dataset_source!r} at {path} -- "
            f"available: {available_dataset_sources()}"
        )
    with open(path) as f:
        spec = yaml.safe_load(f)
    if spec.get("dataset_source") != dataset_source:
        raise ValueError(
            f"{path}'s dataset_source field ({spec.get('dataset_source')!r}) doesn't "
            f"match its filename ({dataset_source!r})"
        )
    return spec


def build_robot_schema(spec: dict) -> RobotSchema:
    try:
        return RobotSchema(
            camera_keys=tuple(spec["camera_keys"]),
            depth_camera_keys=tuple(spec.get("depth_camera_keys", ())),
            state_components=tuple((n, d) for n, d in spec["state_components"]),
            action_components=tuple((n, d) for n, d in spec["action_components"]),
            tick_fps=spec["tick_fps"],
            action_space_components={
                space: tuple(names) for space, names in spec.get("action_space_components", {}).items()
            },
        )
    except KeyError as e:
        raise ValueError(f"schema for {spec.get('dataset_source')!r} is missing field {e}") from e
