"""Per-robot MCAP schema: properties of WHICH ROBOT recorded a dataset (MCAP
topics, message field names, tick rate, directory layout), independent of how
it's preprocessed for training. DataConfig.robot (training/config.py) holds
the active instance; downstream code reads schema values off it
(cfg.robot.camera_keys, etc.) rather than off DataConfig directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RobotSchema(ABC):
    # Fixed-arity camera set this robot's episodes decode into. "top" is
    # populated from whichever of top_camera_candidates is present in a given
    # episode, keeping downstream shapes (e.g. ACTConfig's input_features)
    # fixed regardless of which station recorded an episode.
    camera_keys: tuple[str, ...]

    # (topic, dim) pairs, concatenated in order into one state/action vector.
    state_topics: tuple[tuple[str, int], ...]
    action_topics: tuple[tuple[str, int], ...]

    # Only one "top" feed is kept even if a station publishes more than one
    # top-camera topic -- locked to whichever candidate is seen first in an
    # episode (see training/data/decode.py's _extract()).
    top_camera_candidates: tuple[str, ...]
    wrist_camera_topics: dict[str, str]

    # Field names within a decoded protobuf message.
    state_value_field: str  # decoded state/action message's value field
    camera_data_field: str  # decoded camera message's raw-bytes field
    camera_format_field: str  # decoded camera message's codec-name field

    # Recording tick rate this robot's export used -- the common alignment
    # grid episodes get resampled onto (training/data/align.py).
    tick_fps: float

    @abstractmethod
    def path_to_split_and_task(self, path: str) -> tuple[str, str]:
        """Episode-relative path -> (split, task), per this robot's own
        directory convention."""
        ...

    @property
    def state_dim(self) -> int:
        return sum(d for _, d in self.state_topics)

    @property
    def action_dim(self) -> int:
        return sum(d for _, d in self.action_topics)


@dataclass
class XDOFABCRobot(RobotSchema):
    """XDOF/ABC-130k's teleop-station export."""

    camera_keys: tuple[str, ...] = ("top", "left_wrist", "right_wrist")
    state_topics: tuple[tuple[str, int], ...] = (
        ("/left-arm-state", 6), ("/left-ee-state", 1),
        ("/right-arm-state", 6), ("/right-ee-state", 1),
    )
    action_topics: tuple[tuple[str, int], ...] = (
        ("/left-arm-action", 6), ("/left-ee-action", 1),
        ("/right-arm-action", 6), ("/right-ee-action", 1),
    )
    top_camera_candidates: tuple[str, ...] = ("/top-camera", "/top-left-camera")
    wrist_camera_topics: dict[str, str] = field(default_factory=lambda: {
        "left_wrist": "/left-wrist-camera",
        "right_wrist": "/right-wrist-camera",
    })
    state_value_field: str = "position"
    camera_data_field: str = "data"
    camera_format_field: str = "format"
    tick_fps: float = 30.0

    def path_to_split_and_task(self, path: str) -> tuple[str, str]:
        """"data/{split}/{task}/episode_{uuid}/episode.mcap" -> (split, task)."""
        parts = path.split("/")
        if len(parts) >= 4 and parts[0] == "data":
            return parts[1], parts[2]
        return "unknown", "unknown"
