"""Turn one RawEpisode into per-tick aligned arrays.

align_episode() is policy-agnostic and used by convert.py to build the
LeRobot v3 dataset. chunk_actions() is ACT-specific chunking used only by
the direct-MCAP debug path; the primary path gets chunking for free from
training/vendor/lerobot_datasource.py instead.
"""
from __future__ import annotations

import numpy as np

from training.config import DataConfig
from training.data.align import build_tick_grid, floor_align
from training.data.decode import RawEpisode


def align_episode(raw: RawEpisode, cfg: DataConfig) -> dict[str, np.ndarray]:
    all_times: list[int] = []
    for topic_msgs in list(raw.state.values()) + list(raw.action.values()):
        all_times.extend(t for t, _ in topic_msgs)
    for cam_frames in raw.cameras.values():
        all_times.extend(t for t, _ in cam_frames)
    if not all_times:
        raise ValueError("episode has no messages on any expected topic")

    start_ns, end_ns = min(all_times), max(all_times)
    ticks = build_tick_grid(start_ns, end_ns, cfg.robot.tick_fps)
    if len(ticks) == 0:
        raise ValueError("episode too short to produce even one tick")

    def aligned_concat(topic_dims: tuple, source: dict) -> np.ndarray:
        parts = []
        for topic, _dim in topic_dims:
            msgs = source.get(topic)
            if not msgs:
                raise ValueError(f"expected topic {topic!r} missing from this episode")
            msgs = sorted(msgs, key=lambda m: m[0])
            times = np.array([t for t, _ in msgs], dtype=np.int64)
            values = np.stack([v for _, v in msgs])
            parts.append(floor_align(times, values, ticks))
        return np.concatenate(parts, axis=1)

    state = aligned_concat(cfg.robot.state_topics, raw.state)
    action = aligned_concat(cfg.robot.action_topics, raw.action)

    images: dict[str, np.ndarray] = {}
    for cam_key in cfg.robot.camera_keys:
        frames = raw.cameras.get(cam_key)
        if not frames:
            raise ValueError(f"episode is missing camera {cam_key!r} (station variant mismatch?)")
        times = np.array([t for t, _ in frames], dtype=np.int64)
        stack = np.stack([f for _, f in frames])  # (n, H, W, 3) uint8
        idx = np.searchsorted(times, ticks, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        images[cam_key] = stack[idx]

    result: dict[str, np.ndarray] = {"state": state, "action": action}
    for k, v in images.items():
        result[f"image.{k}"] = v
    return result


def chunk_actions(action: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, action_dim) -> chunks (T, chunk_size, action_dim) + is_pad (T, chunk_size).

    Past episode end, the last real action repeats and is_pad marks it -- same
    convention as training/vendor/lerobot_datasource.py's _chunk_action_column.
    """
    t = action.shape[0]
    last = t - 1
    idx = np.arange(t)[:, None] + np.arange(chunk_size)[None, :]
    is_pad = idx > last
    chunks = action[np.minimum(idx, last)]
    return chunks, is_pad
