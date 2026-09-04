"""Tunables for dataset conversion. Only meaningful to ingestion strategies
that do their own Ray-parallel per-episode conversion (mcap, agibot_hdf5) --
see training/data_prep/strategies/. hf_lerobot_mirror ignores this entirely,
it just downloads and migrates an existing dataset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvertConfig:
    mode: str = "stream"  # "stream" (read source over HTTP) | "download" (cache locally first)
    parallel_camera_decode: bool = True  # decode all cameras per episode concurrently vs. sequentially
    max_concurrent: int | None = None  # None -> derive from live CPU/memory
