"""Ray Train per-worker loop -- policy-agnostic (ACT, MolmoAct2, or pi05; see
training/model/registry.py). Policy-specific concerns (model construction,
forward/loss, optional distributed-wrapping strategy) are dispatched through
a PolicyAdapter rather than hardcoded here. Also implements Lightning-style
best-N-checkpoints + step-windowed early stopping.
"""
from __future__ import annotations

import logging
import os
import pickle
import tempfile

import torch
import torch.distributed
import ray.train
import ray.train.torch
from torch.utils.tensorboard import SummaryWriter

from training.vendor.util import NumpyToTorchCollate
from training.config import RunConfig
from training.model.registry import PolicyAdapter, get_adapter

log = logging.getLogger("act_train")


def _sync_should_stop(should_stop: bool, device: torch.device) -> bool:
    """Broadcasts rank 0's early-stop decision to every rank -- each rank
    trains on a different data shard and could otherwise disagree, which
    would hang DDP's collective ops. Every rank must call this."""
    if not torch.distributed.is_initialized():
        return should_stop
    flag = torch.tensor([1.0 if should_stop else 0.0], device=device)
    torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


def _report_with_checkpoint(
    metrics: dict, adapter: PolicyAdapter, unwrapped_policy, optimizer, dist_ctx,
    run_cfg: RunConfig, epoch: int, step: int, epoch_complete: bool, rank: int, save_checkpoint: bool,
) -> None:
    """dist_ctx is the accelerator from adapter.wrap_for_training (FSDP2 path)
    or None (DDP path). epoch_complete distinguishes a step-windowed
    (mid-epoch) checkpoint from an end-of-epoch one -- required for the
    resume logic in train_loop_per_worker to pick the correct epoch."""
    if not save_checkpoint:
        ray.train.report(metrics)
        return

    if adapter.save_checkpoint:
        # Sharded FSDP2 save needs every rank to participate and all ranks to
        # write to the same directory, so this uses one fixed, deterministic
        # path derived from run_cfg rather than a per-rank temp dir.
        ckpt_dir = os.path.join(run_cfg.storage_root, run_cfg.run_name, "_fsdp2_checkpoint_tmp")
        os.makedirs(ckpt_dir, exist_ok=True)
        adapter.save_checkpoint(dist_ctx, ckpt_dir, epoch, step, epoch_complete)
        if rank == 0:
            ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(ckpt_dir))
        else:
            ray.train.report(metrics)
        # Rank 0's report() must finish reading ckpt_dir before any rank
        # starts overwriting it for the next checkpoint.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # Default (DDP) path: pickle the unwrapped model/optimizer state_dict,
    # rank 0 only.
    if rank == 0:
        state = {
            "model": unwrapped_policy.state_dict(),
            "optim": optimizer.state_dict(),
            "epoch": epoch, "step": step, "epoch_complete": epoch_complete,
        }
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "state.pkl"), "wb") as f:
                pickle.dump(state, f)
            ray.train.report(metrics, checkpoint=ray.train.Checkpoint.from_directory(d))
    else:
        ray.train.report(metrics)


def train_loop_per_worker(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_cfg: RunConfig = config["run_cfg"]
    data_cfg = run_cfg.data
    model_cfg = run_cfg.model
    train_cfg = run_cfg.train

    adapter = get_adapter(
        run_cfg.policy_type,
        distributed_strategy=getattr(model_cfg, "distributed_strategy", "ddp"),
        offload_tokenization=getattr(model_cfg, "offload_tokenization", False),
    )

    dataset_stats = config["dataset_stats"]
    policy, preprocessor = adapter.build(data_cfg, model_cfg, train_cfg, dataset_stats, device=str(device))
    policy = policy.to(device)
    if adapter.post_build_hook:
        adapter.post_build_hook(policy, model_cfg)

    # Built from the unwrapped policy, before any distributed wrapping: DDP
    # doesn't change parameter tensor identity, and accelerate's FSDP2 path
    # (adapter.wrap_for_training) wraps model+optimizer together via
    # accelerator.prepare(), which rebinds the optimizer's param references
    # to the sharded parameters -- so the optimizer must exist first.
    optim_params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    optimizer = torch.optim.AdamW(optim_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    dist_ctx = None
    if adapter.wrap_for_training:
        policy, optimizer, dist_ctx = adapter.wrap_for_training(policy, optimizer, model_cfg, device)
        unwrapped_policy = dist_ctx.unwrap_model(policy)  # accelerate's own unwrap, not the DDP .module pattern
    else:
        policy = ray.train.torch.prepare_model(
            policy, parallel_strategy_kwargs=adapter.prepare_model_kwargs or {},
        )
        # prepare_model only wraps in DistributedDataParallel when world_size
        # > 1, so `policy` may have no `.module` -- unwrap defensively.
        unwrapped_policy = policy.module if hasattr(policy, "module") else policy

    policy.train()  # load-bearing for MolmoAct2's train_mode="freeze" (see model/molmoact2.py)

    start_epoch, step = 0, 0
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        if adapter.load_checkpoint:
            with checkpoint.as_directory() as d:
                state = adapter.load_checkpoint(dist_ctx, d)
        else:
            with checkpoint.as_directory() as d:
                with open(os.path.join(d, "state.pkl"), "rb") as f:
                    state = pickle.load(f)
                unwrapped_policy.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optim"])
        # Only advance to the next epoch when the checkpoint actually
        # finished one; otherwise resume the same epoch (its data iterator
        # restarts from row 0). Checkpoints predating epoch_complete default
        # to True, preserving epoch-boundary-only resume behavior.
        epoch_complete = state.get("epoch_complete", True)
        start_epoch = state["epoch"] + 1 if epoch_complete else state["epoch"]
        step = state["step"]

    if start_epoch >= train_cfg.num_epochs:
        log.warning(
            "RESUMED A COMPLETED RUN: checkpoint is at epoch %d and num_epochs=%d, "
            "so no training steps will run. Raise num_epochs or use a new run_name "
            "to retrain.", start_epoch - 1, train_cfg.num_epochs,
        )
        ray.train.report({"epoch": start_epoch - 1, "steps": step, "skipped_resumed_complete": True})
        return

    rank = ray.train.get_context().get_world_rank()
    shard = ray.train.get_dataset_shard("train")
    image_keys = {f"observation.images.{k}" for k in data_cfg.robot.camera_keys}
    collate = NumpyToTorchCollate(device, image_keys=image_keys)

    # Only rank 0 writes -- every worker's shard is a different data slice,
    # so only rank 0's loss curve is coherent, and multiple workers writing
    # to the same run would corrupt one events file.
    tb_writer = None
    if rank == 0:
        tb_dir = os.path.join(run_cfg.storage_root, run_cfg.run_name, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
        print(f"TensorBoard logs: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    # Shared by early-stop patience and checkpoint-improvement-gating -- one
    # running "best loss seen this run", updated at both window and epoch
    # reports. Not restored from a resumed checkpoint.
    best_loss_seen = float("inf")
    windows_without_improvement = 0
    should_stop = False

    for epoch in range(start_epoch, train_cfg.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        extra_sums: dict[str, float] = {}
        n = 0
        window_loss_sum = 0.0
        window_extra_sums: dict[str, float] = {}
        window_n = 0

        for batch in shard.iter_torch_batches(batch_size=train_cfg.batch_size, collate_fn=collate):
            if not adapter.needs_task:
                batch.pop("task", None)  # language conditioning this policy doesn't use
            # If a Ray Data stage already ran the policy's full preprocessor
            # upstream (see offload_molmoact2_preprocessing), the batch is
            # already normalized/tokenized -- skip calling it again here.
            inputs = batch if adapter.preprocessing_offloaded else preprocessor(batch)
            if dist_ctx is not None and hasattr(dist_ctx, "autocast"):
                # FSDP2/accelerate path only. Accelerator(mixed_precision="bf16")
                # casts model parameters to bf16 but not input batches, so the
                # forward needs autocast() to cast activations on the fly.
                with dist_ctx.autocast():
                    loss, step_metrics = adapter.forward_loss(policy, inputs)
            else:
                loss, step_metrics = adapter.forward_loss(policy, inputs)

            (loss / train_cfg.grad_accum).backward()
            step += 1
            accum += 1
            loss_sum += loss.item()
            n += 1
            window_loss_sum += loss.item()
            window_n += 1
            for k, v in step_metrics.items():
                extra_sums[k] = extra_sums.get(k, 0.0) + v
                window_extra_sums[k] = window_extra_sums.get(k, 0.0) + v

            if accum % train_cfg.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

            if step % 10 == 0 and rank == 0:
                log.info(
                    "epoch=%d step=%d loss=%.4f %s",
                    epoch, step, loss.item(),
                    " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items()),
                )
                tb_writer.add_scalar("train/loss", loss.item(), step)
                for k, v in step_metrics.items():
                    tb_writer.add_scalar(f"train/{k}", v, step)
                for i, g in enumerate(optimizer.param_groups):
                    tb_writer.add_scalar(f"train/lr_group{i}", g["lr"], step)
                tb_writer.flush()

            # Step-windowed report: finer-grained than epoch, so early
            # stopping and best-checkpoint scoring have more than one data
            # point even within a single epoch or a max_train_steps-capped run.
            if step % train_cfg.eval_every_steps == 0:
                window_metrics = {
                    "epoch": epoch, "steps": step, "report_kind": "window",
                    "loss": window_loss_sum / max(window_n, 1),
                    **{k: v / max(window_n, 1) for k, v in window_extra_sums.items()},
                }
                window_loss_sum = 0.0
                window_extra_sums = {}
                window_n = 0

                improved = False
                if rank == 0:
                    for k, v in window_metrics.items():
                        if k not in ("epoch", "steps", "report_kind"):
                            tb_writer.add_scalar(f"window/{k}", v, step)
                    tb_writer.flush()

                    improved = window_metrics["loss"] < best_loss_seen
                    if improved:
                        best_loss_seen = window_metrics["loss"]

                    if train_cfg.early_stop_patience is not None:
                        windows_without_improvement = 0 if improved else windows_without_improvement + 1
                        should_stop = windows_without_improvement >= train_cfg.early_stop_patience
                        if should_stop:
                            log.info(
                                "EARLY STOP at step=%d: loss hasn't improved for %d windows (best=%.4f)",
                                step, windows_without_improvement, best_loss_seen,
                            )

                should_stop = _sync_should_stop(should_stop, device)  # every rank must call this
                save_checkpoint = improved if train_cfg.save_only_on_improvement else True
                _report_with_checkpoint(
                    window_metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
                    run_cfg, epoch, step, epoch_complete=False, rank=rank, save_checkpoint=save_checkpoint,
                )

            if should_stop:
                break
            if train_cfg.max_train_steps and step >= train_cfg.max_train_steps:
                break

        if accum > 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Coarser per-epoch summary, kept alongside the step-windowed reports above.
        metrics = {
            "epoch": epoch, "steps": step, "report_kind": "epoch",
            "loss": loss_sum / max(n, 1),
            **{k: v / max(n, 1) for k, v in extra_sums.items()},
        }

        improved = False
        if rank == 0:
            for k, v in metrics.items():
                if k not in ("epoch", "steps", "report_kind"):
                    tb_writer.add_scalar(f"epoch/{k}", v, epoch)
            tb_writer.flush()

            improved = metrics["loss"] < best_loss_seen
            if improved:
                best_loss_seen = metrics["loss"]

        save_checkpoint = improved if train_cfg.save_only_on_improvement else True
        _report_with_checkpoint(
            metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
            run_cfg, epoch, step, epoch_complete=True, rank=rank, save_checkpoint=save_checkpoint,
        )

        if should_stop or (train_cfg.max_train_steps and step >= train_cfg.max_train_steps):
            break

    if rank == 0:
        # Per-operator Ray Data execution stats for exactly what this
        # worker's shard consumed -- not ds.stats() on the driver's original
        # dataset, which doesn't reflect Ray Train's per-worker splitting.
        stats_text = shard.stats()
        log.info("Ray Data shard stats:\n%s", stats_text)
        stats_path = os.path.join(run_cfg.storage_root, run_cfg.run_name, "ray_data_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_text)
        print(f"Ray Data stats -> {stats_path}")
        if tb_writer is not None:
            tb_writer.add_text("ray_data/shard_stats", f"```\n{stats_text}\n```", step)

    if tb_writer is not None:
        tb_writer.close()
