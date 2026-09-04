"""Floor-align independently-timed streams (state, action, cameras) onto a
common tick grid; state and action publish at different rates, so this
step is required, not optional.
"""
from __future__ import annotations

import numpy as np


def build_tick_grid(start_ns: int, end_ns: int, fps: float) -> np.ndarray:
    step_ns = int(1e9 / fps)
    return np.arange(start_ns, end_ns, step_ns, dtype=np.int64)


def floor_align(times: np.ndarray, values: np.ndarray, tick_times: np.ndarray) -> np.ndarray:
    """For each tick, the value from the most recent `times` entry at or
    before it. Ticks before the first message repeat the first value."""
    idx = np.searchsorted(times, tick_times, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return values[idx]
