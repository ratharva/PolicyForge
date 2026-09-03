"""Shared Ray connection setup for train.py and prepare_data.py."""
from __future__ import annotations

import os

import ray


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
    return {"working_dir": ".", "excludes": excludes}


def connect_ray(runtime_env: dict) -> "ray.runtime_context.RuntimeContext":
    try:
        ctx = ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
        print("connected to existing cluster")
    except ConnectionError:
        # dashboard_host="0.0.0.0" (default is loopback-only "127.0.0.1") so the
        # dashboard is reachable from outside this machine. Only applies to a
        # cluster this call starts -- an already-running external cluster needs
        # the same flag at its own head-node startup instead.
        ctx = ray.init(ignore_reinit_error=True, runtime_env=runtime_env, dashboard_host="0.0.0.0")
        print("started a local Ray instance")
    dashboard_url = getattr(ctx, "dashboard_url", None)
    if dashboard_url:
        print(f"Ray Dashboard: http://{dashboard_url}")
    return ctx
