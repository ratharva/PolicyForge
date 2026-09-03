"""Durable, file-based record of every training run -- config, outcome, and
where to find its checkpoint/TensorBoard logs. One JSON line appended to
<storage_root>/history.jsonl per run (success or failure); `cat`/`jq` it, or
`python -m training.history` for a formatted view.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

from training.config import RunConfig


def history_path(storage_root: str) -> str:
    return os.path.join(storage_root, "history.jsonl")


def record_run(
    run_cfg: RunConfig, v3_root: str, num_workers: int, use_gpu: bool,
    status: str,  # "completed" | "failed"
    metrics: dict | None = None, checkpoint_path: str | None = None,
    run_path: str | None = None, error: str | None = None,
) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": run_cfg.run_name,
        "status": status,
        "tasks": run_cfg.tasks,
        "max_episodes_per_task": run_cfg.max_episodes_per_task,
        "v3_root": v3_root,
        "num_workers": num_workers,
        "use_gpu": use_gpu,
        "policy_type": run_cfg.policy_type,
        "hyperparams": {
            "num_epochs": run_cfg.train.num_epochs,
            "batch_size": run_cfg.train.batch_size,
            "max_train_steps": run_cfg.train.max_train_steps,
            "lr": run_cfg.train.lr,
            "lr_backbone": run_cfg.train.lr_backbone,
            # Whole model-override dataclass, not cherry-picked fields --
            # policy-agnostic since asdict() captures whichever override
            # type this run actually used.
            "model": dataclasses.asdict(run_cfg.model),
        },
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "run_path": run_path,
        "tensorboard_dir": os.path.join(run_cfg.storage_root, run_cfg.run_name, "tensorboard"),
        "error": error,
    }
    path = history_path(run_cfg.storage_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run history -> {path}")


def _load(storage_root: str) -> list[dict]:
    path = history_path(storage_root)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_history(storage_root: str, limit: int = 20) -> None:
    records = _load(storage_root)
    if not records:
        print(f"No run history yet at {history_path(storage_root)}")
        return
    for rec in records[-limit:]:
        m = rec.get("metrics") or {}
        loss = m.get("loss")
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "n/a"
        tasks = ",".join(rec.get("tasks", []))
        print(
            f"{rec['timestamp']}  {rec['status']:9s}  {rec['run_name']:40s}  "
            f"tasks={tasks:30s}  final_loss={loss_str}"
        )
        if rec["status"] == "failed" and rec.get("error"):
            print(f"    error: {rec['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", default="./runs")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print_history(os.path.abspath(args.storage_root), args.limit)
