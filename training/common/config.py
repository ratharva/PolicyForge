"""DataConfig -- shared between data prep and training. Every tunable here
describes the dataset being used, independent of how it was prepared or how
a policy trains on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from training.common.robots import RobotSchema


@dataclass
class DataConfig:
    # "hf://datasets/<repo_id>" (gated repos need HF_TOKEN), "s3://<bucket>/<prefix>",
    # or "gs://<bucket>/<prefix>". See training/data_prep/source.py for backend
    # details -- only the hf:// path has been run against real data.
    source_uri: str = ""
    cache_dir: str | None = None  # None -> huggingface_hub's default cache

    # Which dataset source_uri's data follows -- camera keys, state/action
    # dims, and tick rate live on this object (see training/common/robots.py).
    # Populated by training/data_prep/schema_loader.py from a --dataset-source
    # selection, not hardcoded to any one dataset -- see
    # training/data_prep/strategies/registry.py.
    robot: RobotSchema | None = None

    # Training-time preprocessing choice, not a property of the dataset itself.
    image_size: tuple[int, int] = (224, 224)  # (H, W), resized on decode
