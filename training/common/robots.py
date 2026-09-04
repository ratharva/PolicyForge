"""Generic, dataset-agnostic facts about a robot/dataset -- the properties
every policy/training-time consumer needs regardless of how the raw data
was ingested (camera_keys, state/action dims, tick rate). How to actually
ingest a given dataset's raw format (MCAP topics, an HDF5 layout, an
existing HF LeRobot mirror) is a separate concern -- see
training/data_prep/schema_loader.py and training/data_prep/strategies/.

Always constructed by schema_loader.py from a training/data_prep/schemas/
*.yaml file, never subclassed by hand.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RobotSchema:
    camera_keys: tuple[str, ...]

    # (component name, dim) pairs, concatenated in order into one state/
    # action vector. What "component name" means is up to whichever
    # ingestion strategy interprets it (an MCAP topic name, an HDF5 group
    # path, a LeRobot column name) -- RobotSchema itself only needs the
    # dims, to derive state_dim/action_dim generically.
    state_components: tuple[tuple[str, int], ...]
    action_components: tuple[tuple[str, int], ...]

    # Common alignment/tick rate this dataset's episodes are resampled to.
    tick_fps: float

    @property
    def state_dim(self) -> int:
        return sum(d for _, d in self.state_components)

    @property
    def action_dim(self) -> int:
        return sum(d for _, d in self.action_components)
