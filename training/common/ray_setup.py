"""Shared Ray connection setup for train.py and prepare_data.py."""
from __future__ import annotations

import os

import ray


def _quiet_placement_group_cleaner() -> None:
    """worker_process_setup_hook -- runs once per Ray worker process at
    startup. Filters out ONLY PlacementGroupCleaner's benign, high-frequency
    "State API may be temporarily unavailable" message (a periodic health-
    check retry, not a real problem) -- NOT a setLevel(ERROR) on the whole
    logger, which would also hide its other real warnings (cleanup failure,
    monitor-thread shutdown timeout, dead-controller detection -- confirmed
    by reading every logger.warning() call site in placement_group_cleaner.py)
    that matter for noticing leaked resources."""
    import logging

    class _TransientStateApiWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "State API may be temporarily unavailable" not in record.getMessage()

    logging.getLogger(
        "ray.train.v2._internal.execution.controller.placement_group_cleaner"
    ).addFilter(_TransientStateApiWarningFilter())


def build_runtime_env(storage_root: str | None = None) -> dict:
    """working_dir="." ships the repo root so `training.*` imports resolve
    identically on Ray workers. excludes keeps secrets, caches, converted
    video, and checkpoints out of that upload -- "training/runs/**" (the
    default storage_root) is always excluded, and storage_root is
    additionally excluded by its real resolved path so a --storage-root
    override elsewhere in the repo stays correct too.
    """
    excludes = [
        "hf_tok.txt", "*.txt",
        "training/verify_out/**", "training/.cache/**", "training/lerobot_v3/**",
        "training/runs/**",
        "**/__pycache__", "**/.git",
    ]
    if storage_root:
        try:
            rel = os.path.relpath(storage_root, os.getcwd())
        except ValueError:
            rel = None  # e.g. different drive on Windows
        if rel and not rel.startswith(".."):
            excludes.append(f"{rel}/**")
    return {
        "working_dir": ".", "excludes": excludes,
        "worker_process_setup_hook": _quiet_placement_group_cleaner,
    }


def connect_ray(runtime_env: dict) -> "ray.runtime_context.RuntimeContext":
    try:
        ctx = ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
        print("connected to existing cluster")
    except ConnectionError:
        # Loopback-only by default (Ray's dashboard has no auth). Override via
        # ROBOPOLICYKERNEL_DASHBOARD_HOST for remote access, or tunnel instead:
        # ssh -L 8265:localhost:8265 <host>.
        # `or None` normalizes an explicitly-empty value to "unset" -- os.environ.get(key,
        # default) only falls back on a MISSING key, not a present-but-empty one, which
        # would otherwise silently produce dashboard_host="" instead of the real fallback.
        dashboard_host = os.environ.get("ROBOPOLICYKERNEL_DASHBOARD_HOST") or "127.0.0.1"
        if dashboard_host not in ("127.0.0.1", "localhost", "::1"):
            print(f"WARNING: Ray dashboard binding to {dashboard_host!r} -- no built-in auth, "
                  f"anyone who can reach it can execute code on this machine.")
        ctx = ray.init(ignore_reinit_error=True, runtime_env=runtime_env, dashboard_host=dashboard_host)
        print("started a local Ray instance")
    dashboard_url = getattr(ctx, "dashboard_url", None)
    if dashboard_url:
        print(f"Ray Dashboard: http://{dashboard_url}")
    return ctx
