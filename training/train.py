"""Entrypoint: build Ray Data pipeline from an already-converted LeRobot v3
dataset (via training/vendor/lerobot_datasource.py) -> launch Ray Train.

Assumes training/prepare_data.py has already been run for these --tasks --
this script does no discover/download/convert of its own. If no prepared
dataset is found at --v3-root, it exits with the prepare_data.py command to
run first.

Usage (run from the repo root, i.e. the parent of this training/ directory):
    python -m training.prepare_data --tasks arrange_the_flowers box_folding --max-episodes-per-task 300
    python -m training.train --tasks arrange_the_flowers box_folding
"""
from __future__ import annotations

import argparse
import os
import time

import ray
import ray.data
import ray.train
import ray.train.torch
import torch

from training.common.ray_setup import build_runtime_env, connect_ray
from training.config import MolmoAct2ConfigOverrides, Pi05ConfigOverrides, RunConfig
from training.data.ray_dataset import build_lerobot_v3_dataset, offload_molmoact2_preprocessing, transpose_for_training
from training.data.stats import compute_dataset_stats
from training.data_prep.lerobot_v3_writer import read_conversion_params
from training.data_prep.prepare import has_lerobot_v3_data, resolve_v3_root
from training.data_prep.strategies.registry import available_dataset_sources, get_dataset_source
from training.history import record_run
from training.train_loop import train_loop_per_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", required=True,
                         help="task-name substrings -- must match what training.prepare_data was run with")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--storage-root", default=None)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5,
                         help="base learning rate -- the fallback every MolmoAct2 per-group LR flag "
                              "(--molmoact2-vit-lr etc.) uses when left unset")
    parser.add_argument("--lr-backbone", type=float, default=1e-5,
                         help="ACT-specific backbone LR group -- unused by MolmoAct2/pi05, which have "
                              "their own LR-group flags")
    parser.add_argument("--grad-accum", type=int, default=1,
                         help="gradient accumulation steps")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                         help="AdamW weight decay")
    parser.add_argument("--max-train-steps", type=int, default=None,
                         help="cap total steps for a smoke run; omit for a full run over the data")
    parser.add_argument("--eval-every-steps", type=int, default=200,
                         help="report/checkpoint/early-stop-check granularity, in steps")
    parser.add_argument("--early-stop-patience", type=int, default=5,
                         help="stop after this many eval windows with no loss improvement; "
                              "pass 0 or a negative number to disable early stopping")
    parser.add_argument("--save-only-on-improvement", action="store_true",
                         help="only write a checkpoint when the loss improves, instead of on every "
                              "report (the default) -- less write I/O, but a resume after an "
                              "interruption can lose progress back to whenever the loss last "
                              "improved, since no checkpoint exists for steps after that")
    parser.add_argument("--checkpoint-max-to-keep", type=int, default=3,
                         help="max checkpoints kept on disk. With the default (every report "
                              "checkpointed, scored by loss), this keeps the N best-scoring "
                              "checkpoints PLUS the single most recent one (for resuming from "
                              "close to wherever training actually stopped), even when the most "
                              "recent isn't among the N best -- native Ray Train checkpoint-manager "
                              "behavior, not custom logic here")
    parser.add_argument("--num-workers", type=int, default=None,
                         help="Ray Train DDP workers; default = live GPU count")
    parser.add_argument("--v3-root", default=None,
                         help="where the already-converted LeRobot v3 dataset lives -- local path, "
                              "s3://<bucket>/<prefix>, or gs://<bucket>/<prefix> (default: "
                              "training/lerobot_v3/<dataset-source>/<sorted task names>, local, "
                              "matching training.prepare_data's own default). Must already exist -- "
                              "run training.prepare_data first if it doesn't.")
    parser.add_argument("--dataset-source", default=None, choices=available_dataset_sources(),
                         help="which dataset schema to use for camera_keys/state_dim/action_dim/"
                              "tick_fps -- default: read from the prepared dataset's own "
                              "conversion_params.json (written by prepare_data.py), no flag needed "
                              "in the common case. Only needed as an override for a hand-built "
                              "--v3-root with no conversion_params.json.")

    # --- MolmoAct2 ---
    parser.add_argument("--policy-type", choices=("act", "molmoact2", "pi05"), required=True,
                         help="which policy family to train -- act (this repo's original "
                              "lightweight policy, trains from random init), molmoact2 (~8B-param VLA), "
                              "or pi05 (~2.3B-param VLA, needs --pi05-pretrained-path) -- all three "
                              "share the same lerobot install, see README")
    parser.add_argument("--molmoact2-checkpoint-path", default="allenai/MolmoAct2",
                         help="HF repo id or local path MolmoAct2's VLM backbone loads from")
    parser.add_argument("--molmoact2-setup-type", default="",
                         help="required iff --policy-type molmoact2 -- free-text embodiment prompt fed "
                              "into MolmoAct2's VLM, e.g. 'dual-arm robot with wrist and top cameras'")
    parser.add_argument("--molmoact2-control-mode", default="",
                         help="required iff --policy-type molmoact2 -- free-text control-mode prompt, "
                              "e.g. 'delta joint position'")
    parser.add_argument("--molmoact2-action-mode", choices=("continuous", "discrete", "both"), default="continuous",
                         help="MolmoAct2Config's own default is 'both'; narrowed here to skip the "
                              "discrete FAST-tokenizer dependency/setup entirely")
    parser.add_argument("--molmoact2-train-mode", choices=("fft", "lora", "freeze"), default="lora",
                         help="lora (default): LoRA on the VLM, action expert stays fully trainable -- "
                              "fits under plain DDP on one modern GPU (~20GB at batch 8, per real "
                              "measurements). fft: full fine-tune of everything, needs "
                              "--molmoact2-distributed-strategy fsdp2 to be practical (~60GB/GPU at "
                              "batch 32 under plain DDP otherwise). freeze: VLM frozen, only the action "
                              "expert trains (requires --molmoact2-action-mode continuous). Translated "
                              "internally to MolmoAct2Config's real enable_lora_vlm/"
                              "enable_lora_action_expert/train_action_expert_only fields -- see "
                              "model/molmoact2.py's build_molmoact2_config")
    parser.add_argument("--molmoact2-lora-rank", type=int, default=64,
                         help="MolmoAct2Config's own real default")
    parser.add_argument("--molmoact2-lora-alpha", type=int, default=16,
                         help="MolmoAct2Config's own real default (yes, alpha < rank here)")
    parser.add_argument("--molmoact2-lora-dropout", type=float, default=0.05)
    parser.add_argument("--molmoact2-no-gradient-checkpointing", action="store_false",
                         dest="molmoact2_gradient_checkpointing", default=True,
                         help="disable gradient checkpointing -- only if you've confirmed you have "
                              "the memory headroom without it (see README's memory table)")
    parser.add_argument("--molmoact2-vit-lr", type=float, default=None,
                         help="default: None -> falls back to --lr")
    parser.add_argument("--molmoact2-connector-lr", type=float, default=None,
                         help="default: None -> falls back to --lr")
    parser.add_argument("--molmoact2-action-expert-lr", type=float, default=None,
                         help="default: None -> falls back to --lr")
    parser.add_argument("--molmoact2-distributed-strategy", choices=("ddp", "fsdp2"), default="ddp",
                         help="ddp (default): plain DDP via ray.train.torch.prepare_model, same as ACT "
                              "-- only sane with --molmoact2-train-mode lora. fsdp2: accelerate-driven "
                              "FSDP2 sharding inside the Ray Train worker, for full fine-tuning at scale "
                              "-- see README's FSDP2 section before using this, it carries real "
                              "unverified checkpoint-resume risk")
    parser.add_argument("--molmoact2-fsdp-cpu-offload", action="store_true",
                         help="trades speed for fitting on fewer/smaller GPUs -- fsdp2 strategy only")
    parser.add_argument("--molmoact2-offload-tokenization", action="store_true",
                         help="run MolmoAct2's preprocessor (tokenizer + image processor -- real CPU "
                              "cost per batch) as a Ray Data stage upstream of the Ray Train workers "
                              "instead of inline in the training loop -- see README, has an unverified "
                              "Ray-Data-Arrow round-trip risk of its own")
    parser.add_argument("--molmoact2-offload-concurrency", type=int, default=None,
                         help="Ray Data actor-pool size for --molmoact2-offload-tokenization -- "
                              "default: derived from live CPU count, same convention as "
                              "--max-concurrent for conversion")

    # --- pi05 ---
    parser.add_argument("--pi05-pretrained-path", default="",
                         help="required iff --policy-type pi05 -- HF repo id or local path a real "
                              "pretrained pi05 checkpoint loads from (PI05Policy.from_pretrained(...); "
                              "no verified real default to fall back to, find one on the HF Hub yourself)")
    parser.add_argument("--pi05-freeze-vision-encoder", action="store_true",
                         help="freeze the vision tower only -- language model + action expert still train. "
                              "No LoRA/peft support exists in this lerobot integration (PI05Config.use_peft "
                              "is a dead field, confirmed unreferenced in modeling_pi05.py) -- this and "
                              "--pi05-train-expert-only are the only ways to reduce the trainable surface")
    parser.add_argument("--pi05-train-expert-only", action="store_true",
                         help="only the action expert trains, everything else frozen -- combinable with "
                              "--pi05-freeze-vision-encoder (no validation prevents it, confirmed via a "
                              "real PI05Config(...) construction call)")
    parser.add_argument("--pi05-no-gradient-checkpointing", action="store_false",
                         dest="pi05_gradient_checkpointing", default=True,
                         help="disable gradient checkpointing -- only if you've confirmed you have the "
                              "memory headroom without it. Auto-wires from PI05Config's own flag at "
                              "construction time, no extra wiring needed unlike MolmoAct2")
    parser.add_argument("--pi05-empty-cameras", type=int, default=0,
                         help="pads input_features with dummy observation.images.empty_camera_{i} "
                              "VISUAL features up to this count -- for when --pi05-pretrained-path's "
                              "checkpoint expects more camera slots than this dataset has")
    args = parser.parse_args()

    if args.policy_type == "molmoact2":
        if not args.molmoact2_setup_type or not args.molmoact2_control_mode:
            parser.error("--molmoact2-setup-type and --molmoact2-control-mode are required when --policy-type molmoact2")
        if args.molmoact2_distributed_strategy == "fsdp2" and args.molmoact2_train_mode != "fft":
            parser.error("--molmoact2-distributed-strategy fsdp2 only makes sense with --molmoact2-train-mode fft")
    if args.policy_type == "pi05" and not args.pi05_pretrained_path:
        parser.error("--pi05-pretrained-path is required when --policy-type pi05")

    run_cfg = RunConfig(tasks=args.tasks)
    run_cfg.policy_type = args.policy_type
    run_cfg.train.num_epochs = args.num_epochs
    run_cfg.train.batch_size = args.batch_size
    run_cfg.train.lr = args.lr
    run_cfg.train.lr_backbone = args.lr_backbone
    run_cfg.train.grad_accum = args.grad_accum
    run_cfg.train.weight_decay = args.weight_decay
    run_cfg.train.max_train_steps = args.max_train_steps
    run_cfg.train.eval_every_steps = args.eval_every_steps
    run_cfg.train.early_stop_patience = args.early_stop_patience if args.early_stop_patience > 0 else None
    run_cfg.train.save_only_on_improvement = args.save_only_on_improvement
    run_cfg.train.checkpoint_max_to_keep = args.checkpoint_max_to_keep
    if args.policy_type == "molmoact2":
        # RunConfig()'s default `model` is ACTConfigOverrides -- overwritten
        # here now that --policy-type is known.
        run_cfg.model = MolmoAct2ConfigOverrides(
            checkpoint_path=args.molmoact2_checkpoint_path,
            setup_type=args.molmoact2_setup_type,
            control_mode=args.molmoact2_control_mode,
            action_mode=args.molmoact2_action_mode,
            train_mode=args.molmoact2_train_mode,
            lora_rank=args.molmoact2_lora_rank,
            lora_alpha=args.molmoact2_lora_alpha,
            lora_dropout=args.molmoact2_lora_dropout,
            gradient_checkpointing=args.molmoact2_gradient_checkpointing,
            optimizer_vit_lr=args.molmoact2_vit_lr,
            optimizer_connector_lr=args.molmoact2_connector_lr,
            optimizer_action_expert_lr=args.molmoact2_action_expert_lr,
            distributed_strategy=args.molmoact2_distributed_strategy,
            fsdp_cpu_offload=args.molmoact2_fsdp_cpu_offload,
            offload_tokenization=args.molmoact2_offload_tokenization,
        )
    if args.policy_type == "pi05":
        run_cfg.model = Pi05ConfigOverrides(
            pretrained_path=args.pi05_pretrained_path,
            freeze_vision_encoder=args.pi05_freeze_vision_encoder,
            train_expert_only=args.pi05_train_expert_only,
            gradient_checkpointing=args.pi05_gradient_checkpointing,
            empty_cameras=args.pi05_empty_cameras,
        )
    if args.storage_root:
        run_cfg.storage_root = args.storage_root
    run_cfg.storage_root = os.path.abspath(run_cfg.storage_root)
    run_cfg.run_name = args.run_name or f"{args.policy_type}-{'-'.join(args.tasks)}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not args.v3_root and not args.dataset_source:
        raise SystemExit(
            "Can't determine where the prepared dataset lives -- pass either --v3-root "
            "(an explicit path) or --dataset-source (to derive the default path prepare_data.py "
            "would have used)."
        )
    v3_root = resolve_v3_root(args.v3_root, args.dataset_source, args.tasks)

    if not has_lerobot_v3_data(v3_root):
        raise SystemExit(
            f"No prepared LeRobot v3 dataset found at {v3_root}. Run data prep first:\n"
            f"    python -m training.prepare_data --tasks {' '.join(args.tasks)}"
            + (f" --v3-root {args.v3_root}" if args.v3_root else "")
        )
    conversion_params = read_conversion_params(v3_root)
    dataset_source = args.dataset_source or (conversion_params or {}).get("dataset_source")
    if not dataset_source:
        raise SystemExit(
            f"{v3_root}'s conversion_params.json doesn't record a dataset_source (likely a "
            f"hand-built v3 root) -- pass --dataset-source explicitly."
        )
    source = get_dataset_source(dataset_source)
    run_cfg.data.robot = source.robot
    run_cfg.data.source_uri = source.default_source_uri
    if conversion_params:
        run_cfg.max_episodes_per_task = conversion_params.get(
            "max_episodes_per_task", run_cfg.max_episodes_per_task
        )

    print("\n=== connect to Ray ===")
    connect_ray(build_runtime_env(storage_root=run_cfg.storage_root))
    print("  Data tab   -> per-operator throughput / object-store memory while decoding")
    print("  Jobs/Train -> per-worker status and the loss + policy-specific metrics reported below")

    ray.data.DataContext.get_current().enable_rich_progress_bars = True
    ray.data.DataContext.get_current().use_ray_tqdm = False

    resources = ray.cluster_resources()
    live_gpus = int(resources.get("GPU", 0))
    num_workers = args.num_workers or max(1, live_gpus)
    use_gpu = live_gpus > 0
    print(f"Ray sees {live_gpus} GPU(s); using {num_workers} Ray Train worker(s), use_gpu={use_gpu}")
    if not torch.cuda.is_available() and use_gpu:
        print("WARNING: Ray reports GPUs but this driver process sees none -- check CUDA setup.")

    print("\n=== build Ray Data pipeline (from LeRobot v3) ===")
    raw_ds = build_lerobot_v3_dataset(v3_root, run_cfg.model.chunk_size)

    print("\n=== compute normalization stats ===")
    # Sampled from raw_ds (pre-transpose, HWC) -- see training/data/stats.py.
    dataset_stats = compute_dataset_stats(raw_ds, run_cfg.data)

    ds = transpose_for_training(raw_ds, run_cfg.data.robot.camera_keys, name=run_cfg.run_name)

    if run_cfg.policy_type == "molmoact2" and run_cfg.model.offload_tokenization:
        print("\n=== offload MolmoAct2 preprocessing (tokenizer + image processor) to Ray Data ===")
        concurrency = args.molmoact2_offload_concurrency or max(1, int(resources.get("CPU", 1)) // 2)
        ds = offload_molmoact2_preprocessing(
            ds, run_cfg.data, run_cfg.model, run_cfg.train, dataset_stats,
            concurrency=concurrency, name=run_cfg.run_name,
        )

    if run_cfg.train.save_only_on_improvement:
        # A checkpoint is only attached when the loss improved, so every
        # saved checkpoint is already better than every earlier one --
        # "most recent N" is equivalent to "best N" here.
        checkpoint_config = ray.train.CheckpointConfig(num_to_keep=run_cfg.train.checkpoint_max_to_keep)
    else:
        # Default: a checkpoint is attached to every report. Ray Train's
        # checkpoint manager keeps the N lowest-loss ones plus the single
        # most recently written one, which is what a resumed run picks up
        # from (see TrainConfig.save_only_on_improvement in config.py).
        checkpoint_config = ray.train.CheckpointConfig(
            num_to_keep=run_cfg.train.checkpoint_max_to_keep,
            checkpoint_score_attribute="loss", checkpoint_score_order="min",
        )

    print(f"\n=== launch Ray Train: {run_cfg.run_name} ===")
    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"run_cfg": run_cfg, "dataset_stats": dataset_stats},
        scaling_config=ray.train.ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=ray.train.RunConfig(
            name=run_cfg.run_name,
            storage_path=run_cfg.storage_root,
            failure_config=ray.train.FailureConfig(max_failures=1),
            checkpoint_config=checkpoint_config,
        ),
        datasets={"train": ds},
    )
    try:
        result = trainer.fit()
    except Exception as e:
        # trainer.fit() raises on failure rather than returning a Result with
        # .error set -- record the failure before re-raising so a crashed run
        # still leaves a durable trace.
        record_run(
            run_cfg, v3_root, num_workers, use_gpu, status="failed",
            error=f"{type(e).__name__}: {e}",
        )
        raise

    record_run(
        run_cfg, v3_root, num_workers, use_gpu, status="completed",
        metrics=result.metrics,
        checkpoint_path=str(result.checkpoint) if result.checkpoint else None,
        run_path=result.path,
    )
    print(f"\nfinal metrics: {result.metrics}")
    print(f"checkpoint: {result.checkpoint}")
    print(f"\nView history:    python -m training.history --storage-root {run_cfg.storage_root}")
    print(f"View TensorBoard: tensorboard --logdir {os.path.join(run_cfg.storage_root, run_cfg.run_name, 'tensorboard')}")


if __name__ == "__main__":
    main()
