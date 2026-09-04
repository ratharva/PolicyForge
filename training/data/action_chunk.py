"""Action chunking -- generic, no dependency on how the action array was
produced. Used by data_prep/verify_decode.py and ray_dataset.py's
build_dataset_direct() debug path."""
from __future__ import annotations

import numpy as np


def chunk_actions(action: np.ndarray, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, action_dim) -> chunks (T, chunk_size, action_dim) + is_pad (T, chunk_size).

    Past episode end, the last real action repeats and is_pad marks it --
    same convention as training/vendor/lerobot_datasource.py's
    _chunk_action_column.
    """
    t = action.shape[0]
    last = t - 1
    idx = np.arange(t)[:, None] + np.arange(chunk_size)[None, :]
    is_pad = idx > last
    chunks = action[np.minimum(idx, last)]
    return chunks, is_pad
