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

from dataclasses import dataclass, field, replace


@dataclass
class RobotSchema:
    camera_keys: tuple[str, ...]

    # Depth camera keys, disjoint from camera_keys -- these carry single-
    # channel depth maps, not RGB. Empty for every dataset that doesn't
    # ingest depth (which is all of them today except agibot_alpha).
    depth_camera_keys: tuple[str, ...] = ()

    # (component name, dim) pairs, concatenated in order into one state/
    # action vector. What "component name" means is up to whichever
    # ingestion strategy interprets it (an MCAP topic name, an HDF5 group
    # path, a LeRobot column name) -- RobotSchema itself only needs the
    # dims, to derive state_dim/action_dim generically.
    state_components: tuple[tuple[str, int], ...] = ()
    action_components: tuple[tuple[str, int], ...] = ()

    # Common alignment/tick rate this dataset's episodes are resampled to.
    tick_fps: float = 0.0

    # Named subsets of action_components that represent an alternative
    # "action space" a policy could be trained on (e.g. "joint" ->
    # [joint/position] vs "end_effector" -> [end/position, end/orientation]
    # for agibot_alpha, which records both). A component name absent from
    # every space here is space-agnostic and always included (e.g. a
    # gripper). Empty (the default) means this dataset only ever recorded
    # one space, so action-space selection isn't offered for it.
    action_space_components: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return sum(d for _, d in self.state_components)

    @property
    def action_dim(self) -> int:
        return sum(d for _, d in self.action_components)

    def select_action_space(self, space: str | None) -> "RobotSchema":
        """Returns a new RobotSchema whose action_components is narrowed to
        `space`'s named components plus every component that isn't listed
        under any space (always included regardless of selection). `space`
        of None returns self unchanged -- the default, zero-behavior-change
        case. Raises ValueError if `space` isn't a key of
        action_space_components (including when this schema declares none
        at all)."""
        if space is None:
            return self
        if space not in self.action_space_components:
            raise ValueError(
                f"action space {space!r} not available for this dataset -- "
                f"available: {sorted(self.action_space_components)}"
            )
        selected_names = set(self.action_space_components[space])
        all_space_names = {n for names in self.action_space_components.values() for n in names}
        kept = tuple(
            (name, dim) for name, dim in self.action_components
            if name in selected_names or name not in all_space_names
        )
        return replace(self, action_components=kept)
