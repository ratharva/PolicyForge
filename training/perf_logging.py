"""Opt-in training performance instrumentation
(training/train_loop.py's --log-perf-metrics path): GPU utilization/VRAM,
per-step timing windows, effective batch size/throughput. Deliberately
standalone -- no Ray/lerobot coupling -- mirroring training/vendor/util.py's
style, so it's usable/testable outside a real Ray Train worker.

Off by default: accurate step timing needs torch.cuda.synchronize() calls,
which serialize async CUDA work and cost real throughput whenever they're
on -- every entry point here is only ever called from train_loop.py when
run_cfg.train.log_perf_metrics is True.
"""
from __future__ import annotations

import logging
import threading
import time

import torch

log = logging.getLogger("perf_logging")

_nvml_warned = False


def gpu_snapshot(device: torch.device) -> dict[str, float]:
    """VRAM allocator stats (always available, no extra dependency) plus
    GPU compute/NVML-memory utilization % (needs `nvidia-ml-py` --
    torch.cuda.utilization()/memory_usage() both raise ModuleNotFoundError
    without it, confirmed directly against the installed torch; degrades
    to VRAM-only with one warning rather than crashing the run over an
    optional dependency). Also catches pynvml's OWN exception type
    (`pynvml.NVMLError` and subclasses -- e.g. a real
    NVMLError_LibRmVersionMismatch hit in practice on a machine with a
    kernel-module/userspace NVML version skew) -- an environment-specific
    NVML failure is just as much "GPU util % isn't available right now" as
    a missing import, and must not be allowed to silently kill the
    background poller thread (an uncaught exception in a Python thread
    just ends that thread, with no visible effect on the training run
    itself other than every perf/gpu_* metric going quiet from then on --
    confirmed by hitting exactly that)."""
    global _nvml_warned
    snap = {
        "vram_allocated_mb": torch.cuda.memory_allocated(device) / 1e6,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) / 1e6,
        "vram_peak_mb": torch.cuda.max_memory_allocated(device) / 1e6,
    }
    try:
        snap["gpu_util_pct"] = float(torch.cuda.utilization(device))
        snap["gpu_mem_util_pct"] = float(torch.cuda.memory_usage(device))
    except ModuleNotFoundError:
        if not _nvml_warned:
            log.warning(
                "nvidia-ml-py not installed -- perf/gpu_util_pct and perf/gpu_mem_util_pct "
                "won't be logged (VRAM stats still will be). `pip install nvidia-ml-py` to get them."
            )
            _nvml_warned = True
    except Exception as e:
        # Broad except deliberately: pynvml raises its own NVMLError
        # hierarchy (not importable/catchable by name without importing
        # pynvml ourselves, which we want to avoid doing unconditionally
        # since it's optional) for a variety of real environment issues
        # (driver/library version skew, GPU reset, permissions) -- all of
        # them mean the same thing here: this sample's util%% isn't
        # available, not "crash the poller thread forever."
        if not _nvml_warned:
            log.warning(
                "GPU compute-utilization query failed (%s: %s) -- perf/gpu_util_pct and "
                "perf/gpu_mem_util_pct won't be logged (VRAM stats still will be).",
                type(e).__name__, e,
            )
            _nvml_warned = True
    return snap


def start_gpu_poller(tb_writer, device: torch.device, step_ref: list[int], interval_s: float = 1.5) -> threading.Event:
    """Starts a daemon thread that calls gpu_snapshot() every interval_s
    and writes it to tb_writer under perf/* -- nvidia-smi/NVML calls
    aren't free, so this samples on a wall-clock interval, not every step.
    Tagged with `step_ref[0]` (a 1-element list the caller mutates after
    every training step) rather than its own counter, so every perf/*
    scalar shares one step-based x-axis in TensorBoard. Returns the stop
    Event -- call .set() on it when training ends, before closing
    tb_writer, so the thread doesn't write to a closed writer."""
    stop_event = threading.Event()

    def _poll():
        while not stop_event.is_set():
            try:
                snap = gpu_snapshot(device)
                step = step_ref[0]
                for k, v in snap.items():
                    tb_writer.add_scalar(f"perf/{k}", v, global_step=step)
            except Exception:
                # A daemon thread's uncaught exception silently ends the
                # thread with no effect on the training run other than
                # every perf/gpu_* metric going quiet -- log once and keep
                # the thread alive instead (gpu_snapshot already handles
                # the expected NVML failure modes itself; this is a last
                # line of defense against anything else, e.g. tb_writer
                # racing a concurrent close()).
                log.exception("perf GPU poller iteration failed, continuing")
            stop_event.wait(interval_s)

    thread = threading.Thread(target=_poll, daemon=True, name="perf-gpu-poller")
    thread.start()
    return stop_event


def effective_batch_size(train_cfg, world_size: int) -> int:
    return train_cfg.batch_size * train_cfg.grad_accum * world_size


class Timer:
    """Wall-clock timer for one named window (data_wait_s/preprocess_s/
    compute_s/optimizer_step_s/checkpoint_save_s -- see train_loop.py).
    Synchronizes before AND after the timed span when `device` is CUDA, so
    `.elapsed` reflects real GPU completion, not just async kernel-launch
    return (the entire reason this is opt-in -- see module docstring)."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.elapsed = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> bool:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self._t0
        return False
