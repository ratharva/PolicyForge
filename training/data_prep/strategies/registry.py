"""Dispatch table between dataset sources (abc130k, droid, agibot_alpha,
...) and their ingestion strategy (mcap, hf_lerobot_mirror, agibot_hdf5),
so prepare_data.py/train.py don't hardcode which dataset they're running --
mirrors training/model/registry.py's PolicyAdapter pattern.

Choices are DISCOVERED from training/data_prep/schemas/*.yaml, not a
hardcoded list -- a new dataset that reuses an existing ingestion_strategy
is just a new schema file, no change here. Only a genuinely new raw format
needs a new strategy module registered in _STRATEGY_MODULES below.

Strategy modules are imported lazily inside get_dataset_source(), same
rationale as model/registry.py: an environment that never selects
agibot_hdf5 never needs h5py installed, etc.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from training.common.robots import RobotSchema
from training.data_prep import schema_loader

# ingestion_strategy name -> (module path, ingestion-config builder attr,
# decode_and_align_fn attr or None if this strategy doesn't do per-episode
# Ray-parallel conversion at all -- see hf_lerobot_mirror.py).
_STRATEGY_MODULES: dict[str, str] = {
    "mcap": "training.data_prep.strategies.mcap",
    "hf_lerobot_mirror": "training.data_prep.strategies.hf_lerobot_mirror",
    "agibot_hdf5": "training.data_prep.strategies.agibot_hdf5",
}


@dataclass
class DatasetSource:
    dataset_source: str
    ingestion_strategy: str
    robot: RobotSchema
    default_source_uri: str
    ingestion_config: Any  # McapIngestionConfig | HfLerobotMirrorIngestionConfig | AgiBotHdf5IngestionConfig
    # None for strategies that don't go through convert_to_lerobot_v3 (hf_lerobot_mirror) --
    # present for strategies that do their own Ray-parallel per-episode conversion.
    decode_and_align: Callable[..., dict] | None


def available_dataset_sources() -> list[str]:
    return schema_loader.available_dataset_sources()


def get_dataset_source(dataset_source: str) -> DatasetSource:
    spec = schema_loader.load_schema(dataset_source)
    robot = schema_loader.build_robot_schema(spec)
    strategy_name = spec["ingestion_strategy"]
    if strategy_name not in _STRATEGY_MODULES:
        raise ValueError(
            f"schema {dataset_source!r} names ingestion_strategy {strategy_name!r}, "
            f"but no such strategy is registered -- known: {list(_STRATEGY_MODULES)}"
        )
    module = importlib.import_module(_STRATEGY_MODULES[strategy_name])
    ingestion_config = module.build_ingestion_config(spec)
    decode_and_align = getattr(module, "decode_and_align", None)
    return DatasetSource(
        dataset_source=dataset_source,
        ingestion_strategy=strategy_name,
        robot=robot,
        default_source_uri=spec["default_source_uri"],
        ingestion_config=ingestion_config,
        decode_and_align=decode_and_align,
    )
