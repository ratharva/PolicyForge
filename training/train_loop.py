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
import random
import tempfile
import time

import torch
import torch.distributed
import ray.train
import ray.train.torch
from torch.utils.tensorboard import SummaryWriter

from training import perf_logging
from training.vendor.util import NumpyToTorchCollate
from training.config import RunConfig, TrainConfig
from training.model.image_normalization import apply_image_normalization
from training.model.registry import PolicyAdapter, get_adapter
from training.wandb_logging import filter_metrics, sample_episode_frames

log = logging.getLogger("act_train")

# Caps the held-out eval pass (--eval-split-fraction) to a bounded number of
# batches per eval_every_steps window, regardless of how large the eval
# split is -- an unbounded full pass every window would defeat the point
# of frequent windowed reporting.
_EVAL_MAX_BATCHES = 50


def _sync_should_stop(should_stop: bool, device: torch.device) -> bool:
    """Broadcasts rank 0's early-stop decision to every rank -- each rank
    trains on a different data shard and could otherwise disagree, which
    would hang DDP's collective ops. Every rank must call this."""
    if not torch.distributed.is_initialized():
        return should_stop
    flag = torch.tensor([1.0 if should_stop else 0.0], device=device)
    torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


def _log_all(
    tb_writer, wandb_run, prefix: str, values: dict, step: int, train_cfg: TrainConfig,
    wandb_step: int | None = None,
) -> None:
    """Writes `values` (bare keys, e.g. {"loss": ...}) to both loggers
    under `f"{prefix}/{key}"` -- TensorBoard unfiltered (as today), W&B
    subject to train_cfg.wandb_metrics/wandb_exclude_metrics (see
    training/wandb_logging.py's filter_metrics). Either logger being None
    (--no-tensorboard / --wandb not passed) is a no-op for that logger.

    `wandb_step` (defaults to `step`) exists because W&B's `step` is ONE
    shared counter across every key in a run, unlike TensorBoard's
    per-tag x-axis -- confirmed by reproducing it directly: logging
    epoch/* at `step=epoch` (0, 1, 2, ...) after window/*/train/* already
    used much larger training-step values silently DROPPED the epoch/*
    point entirely (wandb only accepts non-decreasing steps; it does not
    error, it just discards). The epoch-report call site passes the real
    training step here instead, keeping `epoch` as a value inside the
    dict rather than as the step."""
    prefixed = {f"{prefix}/{k}": v for k, v in values.items()}
    if tb_writer is not None:
        for k, v in prefixed.items():
            tb_writer.add_scalar(k, v, step)
    if wandb_run is not None:
        filtered = filter_metrics(prefixed, train_cfg.wandb_metrics, train_cfg.wandb_exclude_metrics)
        if filtered:
            wandb_run.log(filtered, step=wandb_step if wandb_step is not None else step)


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
    image_keys |= {f"observation.images.depth_{k}" for k in data_cfg.robot.depth_camera_keys}
    collate = NumpyToTorchCollate(device, image_keys=image_keys)

    # Held-out eval split (opt-in -- --eval-split-fraction): a second named
    # dataset registered by train.py alongside "train" when set, absent
    # otherwise. Real held-out inference pass; see the eval_every_steps
    # block below.
    eval_shard = ray.train.get_dataset_shard("eval") if train_cfg.eval_split_fraction else None

    # Only rank 0 writes TensorBoard -- every worker's shard is a different
    # data slice, so only rank 0's loss curve is coherent, and multiple
    # workers writing to the same run would corrupt one events file.
    # train_cfg.tensorboard=False (--no-tensorboard) skips this entirely,
    # e.g. when only W&B is wanted.
    tb_writer = None
    if rank == 0 and train_cfg.tensorboard:
        tb_dir = os.path.join(run_cfg.storage_root, run_cfg.run_name, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir, flush_secs=10)
        print(f"TensorBoard logs: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    # W&B (training/wandb_logging.py) -- setup_wandb() does its own
    # rank-zero check internally (via Ray's own session), so it's called on
    # EVERY rank, not gated by `if rank == 0:` the way tb_writer is above;
    # non-zero ranks get back a disabled no-op run object. Imported lazily
    # (like every other optional dependency in this codebase -- see
    # perf_logging.py's nvidia-ml-py handling) so wandb never needs to be
    # installed unless --wandb is actually passed.
    wandb_run = None
    if train_cfg.wandb:
        from ray.air.integrations.wandb import setup_wandb

        wandb_run = setup_wandb(
            config={"run_name": run_cfg.run_name, "policy_type": run_cfg.policy_type},
            project=train_cfg.wandb_project, entity=train_cfg.wandb_entity, name=run_cfg.run_name,
        )

    # Episode-preview GIFs (training/wandb_logging.py's sample_episode_frames)
    # -- opt-in per camera. Reads directly from the converted v3 root's own
    # video files, not from Ray Data, so it needs v3_root (plumbed in by
    # train.py, not part of RunConfig's own schema -- see that file) and,
    # when a real eval split exists, the held-out episode-index set so GIFs
    # come specifically from data the model never trained on.
    v3_root = config.get("v3_root")
    eval_episode_indices = config.get("eval_episode_indices")
    gif_rng = random.Random(0)
    # Computed once here, not per-window: the held-out set when a real eval
    # split exists (so GIFs come specifically from data never trained on),
    # else every real episode_index in the dataset (one lightweight
    # meta/episodes read, not a Ray Data query). None/empty when GIFs
    # aren't configured at all, so no extra I/O happens in the common case.
    gif_episode_pool: list[int] | None = None
    if rank == 0 and train_cfg.wandb_gif_cameras and v3_root:
        if eval_episode_indices:
            gif_episode_pool = list(eval_episode_indices)
        else:
            from training.vendor.lerobot_datasource import LeRobotDatasourceMetadata

            gif_episode_pool = list(range(LeRobotDatasourceMetadata(v3_root).total_episodes))

    # Shared by early-stop patience and checkpoint-improvement-gating -- one
    # running "best loss seen this run", updated at both window and epoch
    # reports. Not restored from a resumed checkpoint.
    best_loss_seen = float("inf")
    windows_without_improvement = 0
    should_stop = False

    # --- perf instrumentation (training/perf_logging.py), opt-in via
    # --log-perf-metrics -- see that module's docstring for why it's not
    # always on. `sync_device` is passed to every perf_logging.Timer below;
    # None (the off case) skips torch.cuda.synchronize() entirely, so the
    # only always-on cost is a couple of cheap perf_counter() calls per
    # step, not the real synchronization overhead this flag exists to make
    # opt-in.
    sync_device = device if train_cfg.log_perf_metrics else None
    step_ref = [0]  # mutated every step below; read by the GPU poller thread
    gpu_poll_stop = None
    world_size = ray.train.get_context().get_world_size()
    if train_cfg.log_perf_metrics and rank == 0:
        eff_bs = perf_logging.effective_batch_size(train_cfg, world_size)
        _log_all(tb_writer, wandb_run, "perf", {"effective_batch_size": eff_bs}, 0, train_cfg)
        print(
            f"Effective batch size: {eff_bs} (batch_size={train_cfg.batch_size} x "
            f"grad_accum={train_cfg.grad_accum} x world_size={world_size})"
        )
        gpu_poll_stop = perf_logging.start_gpu_poller(tb_writer, device, step_ref)

    prof = None
    prof_started = False
    if train_cfg.log_perf_metrics and train_cfg.profile_steps and rank == 0:
        # torch.profiler is already reachable off the module-level `import
        # torch` above -- an `import torch.profiler` statement here would
        # make `torch` itself a local name for this WHOLE function (Python
        # decides local-vs-global at compile time from any local
        # import/assignment anywhere in the function body), breaking the
        # very first line's `torch.device(...)` with an UnboundLocalError.
        # Confirmed by hitting exactly that in a real run.
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_dir),
        )
    # Single continuous timeline across the whole run (NOT reset per epoch,
    # unlike the window_* sums below) -- inter-epoch time (epoch-end
    # reporting/logging) is real idle time and legitimately belongs in the
    # next epoch's first data_wait_s measurement, not discarded.
    t_iter_end = time.perf_counter()

    for epoch in range(start_epoch, train_cfg.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        loss_sum = 0.0
        extra_sums: dict[str, float] = {}
        n = 0
        window_loss_sum = 0.0
        window_extra_sums: dict[str, float] = {}
        window_n = 0
        window_data_wait_sum = window_preprocess_sum = window_compute_sum = 0.0
        window_optimizer_step_sum, window_optimizer_step_n = 0.0, 0
        window_wall_start = time.perf_counter()

        for batch in shard.iter_torch_batches(batch_size=train_cfg.batch_size, collate_fn=collate):
            data_wait_s = time.perf_counter() - t_iter_end

            if not adapter.needs_task:
                batch.pop("task", None)  # language conditioning this policy doesn't use
            # If a Ray Data stage already ran the policy's full preprocessor
            # upstream (see offload_molmoact2_preprocessing), the batch is
            # already image-normalized/normalized/tokenized -- skip doing
            # any of that again here.
            with perf_logging.Timer(sync_device) as t_prep:
                if not adapter.preprocessing_offloaded:
                    batch = apply_image_normalization(
                        batch, data_cfg.image_normalization, data_cfg.default_image_normalization,
                        dataset_stats, data_cfg.image_normalization_max,
                    )
                inputs = batch if adapter.preprocessing_offloaded else preprocessor(batch)

            with perf_logging.Timer(sync_device) as t_compute:
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
            step_ref[0] = step
            accum += 1
            loss_sum += loss.item()
            n += 1
            window_loss_sum += loss.item()
            window_n += 1
            window_data_wait_sum += data_wait_s
            window_preprocess_sum += t_prep.elapsed
            window_compute_sum += t_compute.elapsed
            for k, v in step_metrics.items():
                extra_sums[k] = extra_sums.get(k, 0.0) + v
                window_extra_sums[k] = window_extra_sums.get(k, 0.0) + v

            ran_optimizer_step = False
            if accum % train_cfg.grad_accum == 0:
                with perf_logging.Timer(sync_device) as t_opt:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                ran_optimizer_step = True
                window_optimizer_step_sum += t_opt.elapsed
                window_optimizer_step_n += 1
                accum = 0

            if prof is not None:
                start_step, end_step = train_cfg.profile_steps
                if step == start_step and not prof_started:
                    prof.start()
                    prof_started = True
                if prof_started:
                    prof.step()
                if step == end_step and prof_started:
                    prof.stop()
                    prof_started = False

            if step % 10 == 0 and rank == 0:
                log.info(
                    "epoch=%d step=%d loss=%.4f %s",
                    epoch, step, loss.item(),
                    " ".join(f"{k}={v:.4f}" for k, v in step_metrics.items()),
                )
                train_values = {"loss": loss.item(), **step_metrics}
                for i, g in enumerate(optimizer.param_groups):
                    train_values[f"lr_group{i}"] = g["lr"]
                _log_all(tb_writer, wandb_run, "train", train_values, step, train_cfg)
                if train_cfg.log_perf_metrics:
                    perf_values = {
                        "data_wait_s": data_wait_s, "preprocess_s": t_prep.elapsed, "compute_s": t_compute.elapsed,
                    }
                    if ran_optimizer_step:
                        perf_values["optimizer_step_s"] = t_opt.elapsed
                    _log_all(tb_writer, wandb_run, "perf", perf_values, step, train_cfg)
                if tb_writer is not None:
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
                window_n_for_perf = window_n  # window_n gets zeroed right below
                window_n = 0

                improved = False
                if rank == 0:
                    _log_all(
                        tb_writer, wandb_run, "window",
                        {k: v for k, v in window_metrics.items() if k not in ("epoch", "steps", "report_kind")},
                        step, train_cfg,
                    )
                    if train_cfg.log_perf_metrics:
                        window_wall_s = time.perf_counter() - window_wall_start
                        io_bound_denom = window_data_wait_sum + window_preprocess_sum + window_compute_sum
                        perf_window_values = {
                            "io_bound_fraction": window_data_wait_sum / io_bound_denom if io_bound_denom > 0 else 0.0,
                            "samples_per_sec": (
                                window_n_for_perf * train_cfg.batch_size * world_size / window_wall_s
                                if window_wall_s > 0 else 0.0
                            ),
                        }
                        if window_optimizer_step_n > 0:
                            perf_window_values["optimizer_step_s_avg"] = (
                                window_optimizer_step_sum / window_optimizer_step_n
                            )
                        _log_all(tb_writer, wandb_run, "perf", perf_window_values, step, train_cfg)
                    if tb_writer is not None:
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

                window_data_wait_sum = window_preprocess_sum = window_compute_sum = 0.0
                window_optimizer_step_sum, window_optimizer_step_n = 0.0, 0
                window_wall_start = time.perf_counter()

                # Real held-out eval pass (--eval-split-fraction only) --
                # reuses adapter.forward_loss unmodified, just under
                # policy.eval()/no_grad() and fed from the held-out shard.
                # Deliberately does NOT feed into checkpoint-improvement
                # scoring/early-stopping above (still training-loss-based,
                # unchanged) -- this is additional observability, not a
                # change to the existing selection criteria.
                if eval_shard is not None:
                    policy.eval()
                    eval_loss_sum, eval_n = 0.0, 0
                    eval_metrics_sum: dict[str, float] = {}
                    last_eval_inputs = None
                    with torch.no_grad():
                        for i, eval_batch in enumerate(
                            eval_shard.iter_torch_batches(batch_size=train_cfg.batch_size, collate_fn=collate)
                        ):
                            if i >= _EVAL_MAX_BATCHES:
                                break
                            if not adapter.needs_task:
                                eval_batch.pop("task", None)
                            if not adapter.preprocessing_offloaded:
                                eval_batch = apply_image_normalization(
                                    eval_batch, data_cfg.image_normalization, data_cfg.default_image_normalization,
                                    dataset_stats, data_cfg.image_normalization_max,
                                )
                            eval_inputs = eval_batch if adapter.preprocessing_offloaded else preprocessor(eval_batch)
                            last_eval_inputs = eval_inputs
                            eval_loss, eval_step_metrics = adapter.forward_loss(policy, eval_inputs)
                            eval_loss_sum += eval_loss.item()
                            eval_n += 1
                            for k, v in eval_step_metrics.items():
                                eval_metrics_sum[k] = eval_metrics_sum.get(k, 0.0) + v

                        # Predicted-frames GIFs -- pure groundwork for a
                        # future policy that actually predicts visual
                        # frames (e.g. a world model); see PolicyAdapter.
                        # predict_frames's docstring. Inert no-op for
                        # ACT/MolmoAct2/PI05 today (adapter.predict_frames
                        # is None for all three). Logged at this eval
                        # pass's own step, same key across the run, so
                        # W&B's own per-step media history is what shows
                        # the generated frames improving over training --
                        # no extra "compare across steps" logic needed.
                        # --wandb-predict-frames-every-steps controls how
                        # often, relative to eval passes (can only be a
                        # multiple of eval_every_steps -- generating one
                        # needs a real eval batch, which only exists here).
                        predict_frames_every = train_cfg.wandb_predict_frames_every_steps or train_cfg.eval_every_steps
                        if (
                            adapter.predict_frames is not None and rank == 0
                            and wandb_run is not None and last_eval_inputs is not None
                            and step % predict_frames_every == 0
                        ):
                            predicted = adapter.predict_frames(policy, last_eval_inputs)
                            if predicted:
                                import wandb

                                wandb_run.log(
                                    {
                                        f"eval_gif/{name}": wandb.Video(
                                            frames.transpose(0, 3, 1, 2), format="gif", fps=data_cfg.robot.tick_fps,
                                        )
                                        for name, frames in predicted.items()
                                    },
                                    step=step,
                                )
                    policy.train()
                    if rank == 0 and eval_n > 0:
                        eval_values = {
                            "loss": eval_loss_sum / eval_n,
                            **{k: v / eval_n for k, v in eval_metrics_sum.items()},
                        }
                        _log_all(tb_writer, wandb_run, "eval", eval_values, step, train_cfg)

                should_stop = _sync_should_stop(should_stop, device)  # every rank must call this
                save_checkpoint = improved if train_cfg.save_only_on_improvement else True
                with perf_logging.Timer() as t_ckpt:
                    _report_with_checkpoint(
                        window_metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
                        run_cfg, epoch, step, epoch_complete=False, rank=rank, save_checkpoint=save_checkpoint,
                    )
                if train_cfg.log_perf_metrics and rank == 0 and save_checkpoint:
                    _log_all(tb_writer, wandb_run, "perf", {"checkpoint_save_s": t_ckpt.elapsed}, step, train_cfg)

            # Episode-preview GIFs -- independent cadence from the window
            # report above (wandb_gif_every_steps can differ from
            # eval_every_steps), so this is its own top-level check, not
            # nested in the block above.
            if rank == 0 and wandb_run is not None and gif_episode_pool:
                gif_every = train_cfg.wandb_gif_every_steps or train_cfg.eval_every_steps
                if step % gif_every == 0:
                    ep_idx = gif_rng.choice(gif_episode_pool)
                    for cam in train_cfg.wandb_gif_cameras:
                        try:
                            frames = sample_episode_frames(
                                v3_root, ep_idx, cam, train_cfg.wandb_gif_frames, data_cfg.image_size,
                            )
                        except ValueError as e:
                            log.warning("GIF sampling failed for camera %r, episode %d: %s", cam, ep_idx, e)
                            continue
                        import wandb

                        frames_chw = frames.transpose(0, 3, 1, 2)
                        wandb_run.log(
                            {f"gif/{cam}": wandb.Video(frames_chw, format="gif", fps=data_cfg.robot.tick_fps)},
                            step=step,
                        )

            t_iter_end = time.perf_counter()
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
            _log_all(
                tb_writer, wandb_run, "epoch",
                {k: v for k, v in metrics.items() if k not in ("epoch", "steps", "report_kind")},
                epoch, train_cfg, wandb_step=step,
            )
            if tb_writer is not None:
                tb_writer.flush()

            improved = metrics["loss"] < best_loss_seen
            if improved:
                best_loss_seen = metrics["loss"]

        save_checkpoint = improved if train_cfg.save_only_on_improvement else True
        with perf_logging.Timer() as t_ckpt:
            _report_with_checkpoint(
                metrics, adapter, unwrapped_policy, optimizer, dist_ctx,
                run_cfg, epoch, step, epoch_complete=True, rank=rank, save_checkpoint=save_checkpoint,
            )
        if train_cfg.log_perf_metrics and rank == 0 and save_checkpoint:
            _log_all(tb_writer, wandb_run, "perf", {"checkpoint_save_s": t_ckpt.elapsed}, step, train_cfg)

        if should_stop or (train_cfg.max_train_steps and step >= train_cfg.max_train_steps):
            break

    if prof is not None and prof_started:
        # max_train_steps/should_stop broke out of the loop before reaching
        # profile_steps' end -- stop it anyway so on_trace_ready still
        # fires and a partial trace is written, not silently dropped.
        prof.stop()

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

    if gpu_poll_stop is not None:
        gpu_poll_stop.set()  # stop the poller thread before closing tb_writer under it

    if tb_writer is not None:
        tb_writer.close()

    if wandb_run is not None:
        wandb_run.finish()
