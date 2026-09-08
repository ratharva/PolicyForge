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
import sys
import time

import draccus
import draccus.utils
import ray
import ray.data
import ray.train
import ray.train.torch
import torch

from training.common.ray_setup import build_runtime_env, connect_ray
from training.config import ACTConfigOverrides, MolmoAct2ConfigOverrides, Pi05ConfigOverrides, RunConfig
from training.data.ray_dataset import (
    build_lerobot_v3_dataset, offload_molmoact2_preprocessing, select_action_space,
    to_relative_action_space, transpose_for_training,
)
from training.data.stats import compute_dataset_stats
from training.data_prep.lerobot_v3_writer import read_conversion_params
from training.data_prep.prepare import has_lerobot_v3_data, resolve_v3_root
from training.data_prep.strategies.registry import available_dataset_sources, get_dataset_source
from training.history import record_run
from training.model.image_normalization import MODES as IMAGE_NORMALIZATION_MODES
from training.train_loop import train_loop_per_worker


def _parse_kv_pairs(pairs: list[str] | None, value_type=str) -> dict:
    """Parses ["CAM=VALUE", ...] CLI args (--image-normalization,
    --image-normalization-max) into {CAM: value_type(VALUE)}."""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"expected CAMERA=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key] = value_type(value)
    return out


# --- --config-file (training/README.md's "Config files" section) ---
# A YAML file, parsed into a base RunConfig via draccus.parse (the same
# library lerobot's own ACTConfig/PI05Config/MolmoAct2Config are built on --
# every named CLI flag below still works exactly as it does today and
# takes precedence over anything the YAML sets; --config-file only fills
# in values nothing else was explicitly passed for.
_POLICY_MODEL_CLASSES = {
    "act": ACTConfigOverrides, "molmoact2": MolmoAct2ConfigOverrides, "pi05": Pi05ConfigOverrides,
}


def _apply_if_explicit(
    target, attr: str, args: argparse.Namespace, dest: str, parser: argparse.ArgumentParser, transform=None,
) -> None:
    """Only overwrites target.attr when --dest was actually passed on the
    command line (its parsed value differs from the parser's own default
    for it) -- so a --config-file-loaded value survives when the user
    didn't explicitly override that particular flag. No add_argument
    default= needed to change for this: parser.get_default(dest) already
    reflects whatever's declared there. `transform`, if given, is applied
    to the raw CLI value before assignment (e.g. early_stop_patience's
    "<= 0 means disabled" -> None convention) -- explicitness is still
    judged on the RAW value, before transform. Known, accepted limitation:
    a CLI value that happens to equal its own default is indistinguishable
    from not having passed the flag at all, and the config-file's value
    (if any) wins in that case."""
    value = getattr(args, dest)
    if value != parser.get_default(dest):
        setattr(target, attr, transform(value) if transform else value)


def _load_base_run_config(config_file: str | None) -> RunConfig:
    if not config_file:
        return RunConfig()
    try:
        return draccus.parse(RunConfig, config_path=config_file, args=[])
    except draccus.utils.DraccusException as e:
        detail = f"{e}" + (f" (caused by: {e.__cause__})" if e.__cause__ else "")
        raise SystemExit(f"--config-file {config_file}: {detail}") from e


def main() -> None:
    # Checked before parser.parse_args() (and before --tasks/--policy-type,
    # both required=True below, would reject a bare invocation) so this
    # works as a standalone, no-other-flags-needed discovery command --
    # see training/wandb_logging.py's format_metrics_catalog.
    if "--list-wandb-metrics" in sys.argv:
        from training.wandb_logging import format_metrics_catalog

        print(format_metrics_catalog())
        return

    parser = argparse.ArgumentParser(description=__doc__)
    # Registered here too (in addition to the sys.argv check above) purely
    # so it shows up in --help -- the real check above always short-
    # circuits before this flag would otherwise need to be parsed.
    parser.add_argument("--list-wandb-metrics", action="store_true",
                         help="print every W&B metric group/name this pipeline can emit and exit -- "
                              "works standalone, no other flags needed")
    parser.add_argument("--tasks", nargs="+", required=True,
                         help="task-name substrings -- must match what training.prepare_data was run with")
    parser.add_argument("--config-file", default=None,
                         help="optional YAML file (draccus-parsed, same library lerobot's own policy "
                              "configs use) providing a base RunConfig -- every named flag below still "
                              "works exactly as today and takes precedence over anything the YAML sets; "
                              "this only fills in values nothing else was explicitly passed for. See "
                              "training/README.md's config-file section for the YAML shape")
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

    # --- TensorBoard / W&B (training/wandb_logging.py) ---
    parser.add_argument("--no-tensorboard", action="store_false", dest="tensorboard", default=True,
                         help="disable TensorBoard logging (on by default) -- e.g. when only --wandb "
                              "is wanted. Independently toggleable from --wandb, not either/or")
    parser.add_argument("--wandb", action="store_true",
                         help="log to Weights & Biases via ray.air.integrations.wandb.setup_wandb() "
                              "-- off by default. Additive: everything TensorBoard already logs also "
                              "goes to W&B at the same cadences. Needs `wandb` installed "
                              "(see training/requirements.txt) and a real W&B login/API key")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None,
                         help="default: unset -- the WANDB_MODE env var (or wandb's own 'online' "
                              "default) decides. 'online': real syncing, needs a real login (`wandb "
                              "login`). 'offline': no network/login needed, writes locally -- good for "
                              "a smoke test, sync later with `wandb sync`. 'disabled': wandb's own "
                              "no-op mode. Passing this explicitly overrides WANDB_MODE if both are set")
    parser.add_argument("--wandb-project", default=None,
                         help="W&B project name -- default: falls back to the WANDB_PROJECT env var "
                              "(wandb's own behavior) if set, else wandb's own default")
    parser.add_argument("--wandb-entity", default=None,
                         help="W&B entity (team/user) -- default: falls back to the WANDB_ENTITY env "
                              "var (wandb's own behavior) if set, else wandb's own default")
    parser.add_argument("--wandb-metrics", nargs="+", default=None, metavar="NAME",
                         help="allowlist of metric names to send to W&B (fnmatch globs OK, e.g. "
                              "'perf/*') -- default: everything already being computed. Doesn't "
                              "affect TensorBoard, which always gets everything. Merged with "
                              "--wandb-metric-groups below if both are given. Run --list-wandb-metrics "
                              "to see every real metric name/group")
    parser.add_argument("--wandb-metric-groups", nargs="+", default=[], metavar="NAME",
                         help="friendly names for common --wandb-metrics glob groups (e.g. 'core', "
                              "'perf', 'media') -- run --list-wandb-metrics to see every real group "
                              "and what it expands to")
    parser.add_argument("--wandb-exclude-metrics", nargs="+", default=[], metavar="NAME",
                         help="denylist of metric names to keep OUT of W&B (fnmatch globs OK), "
                              "applied after --wandb-metrics/--wandb-metric-groups")
    parser.add_argument("--wandb-gif-cameras", nargs="+", default=[], metavar="CAMERA",
                         help="log a short GIF from this camera every --wandb-gif-every-steps, "
                              "sampled from a real recorded episode's own consecutive frames (not a "
                              "shuffled training batch) -- off by default. Requires --wandb")
    parser.add_argument("--wandb-gif-every-steps", type=int, default=None,
                         help="GIF logging cadence -- default: reuse --eval-every-steps")
    parser.add_argument("--wandb-gif-frames", type=int, default=30,
                         help="frames per GIF")
    parser.add_argument("--wandb-predict-frames-every-steps", type=int, default=None,
                         help="how often to log a policy's OWN predicted-frames GIF (only meaningful "
                              "for a future policy that implements PolicyAdapter.predict_frames -- "
                              "inert for act/molmoact2/pi05 today, which only predict actions) -- "
                              "default: log one every eval pass. Can only be a multiple of "
                              "--eval-every-steps, since generating one needs a real eval batch, which "
                              "only exists when the eval pass itself runs")
    parser.add_argument("--eval-split-fraction", type=float, default=None,
                         help="hold out this fraction of episodes from training entirely, for a real "
                              "eval pass (policy.eval()/no_grad(), reusing the same forward_loss) "
                              "every --eval-every-steps, logged under eval/* -- default: no split, no "
                              "eval pass, matching today's behavior exactly. When set, "
                              "--wandb-gif-cameras also sample specifically from the held-out set "
                              "instead of anywhere in the dataset")

    # --- perf instrumentation (training/perf_logging.py) ---
    parser.add_argument("--log-perf-metrics", action="store_true",
                         help="log GPU utilization/VRAM, per-step timing (data-wait/preprocess/"
                              "compute/optimizer-step), effective batch size, throughput, and an "
                              "IO-bound-vs-compute-bound ratio to TensorBoard under perf/* -- off by "
                              "default since accurate timing needs torch.cuda.synchronize() calls, "
                              "which cost real throughput whenever they're on. Needs `nvidia-ml-py` "
                              "installed for GPU compute-utilization %% (perf/gpu_util_pct, "
                              "perf/gpu_mem_util_pct) -- VRAM stats work without it")
    parser.add_argument("--profile-steps", default=None, metavar="START:END",
                         help="capture a real torch.profiler trace for steps START..END (inclusive) "
                              "into the run's tensorboard/ dir, viewable in TensorBoard's PyTorch "
                              "Profiler tab -- requires --log-perf-metrics. Keep the range small "
                              "(rank 0 only, but traces still get large fast)")
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

    # --- action space (component selection + absolute/delta) ---
    parser.add_argument("--action-space", default=None,
                         help="which named subset of the dataset's action_components to train on, "
                              "e.g. 'joint' or 'end_effector' for agibot_alpha (which records both) -- "
                              "default: every component (current behavior). Valid names are dataset-"
                              "specific (the schema's action_space_components keys); validated once "
                              "--dataset-source is resolved, not here")
    parser.add_argument("--action-representation", choices=("absolute", "delta"), default="absolute",
                         help="'absolute' (default): action column unchanged. 'delta': action -= "
                              "observation.state (per masked dim), using each action component's "
                              "same-named state component as the reference point -- components with no "
                              "same-named state component stay absolute (no reference point to use)")
    parser.add_argument("--action-delta-exclude", nargs="+", default=[],
                         help="action component names to keep absolute even under "
                              "--action-representation delta (e.g. a gripper component name)")

    # --- per-camera image normalization ---
    parser.add_argument("--image-normalization", nargs="+", default=[],
                         metavar="CAMERA=MODE",
                         help=f"per-camera normalization mode, e.g. head=unit01 wrist=mean_std -- modes: "
                              f"{', '.join(IMAGE_NORMALIZATION_MODES)}. A camera not listed here uses "
                              f"--image-normalization-default")
    parser.add_argument("--image-normalization-default", choices=IMAGE_NORMALIZATION_MODES, default="mean_std",
                         help="mode for any camera not covered by --image-normalization -- 'mean_std' "
                              "(default) replicates current behavior exactly")
    parser.add_argument("--image-normalization-max", nargs="+", default=[],
                         metavar="CAMERA=VALUE",
                         help="required for any camera using the 'depth' or 'log' mode -- the raw-value "
                              "ceiling to clip/scale by (e.g. max depth in millimeters); no guessed default")

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

    # Policy-specific "required iff"/cross-field validation moved below,
    # AFTER --config-file layering -- it must check the FINAL resolved
    # run_cfg.model (which a YAML file can also populate), not raw CLI args
    # alone, or a value coming only from --config-file would silently skip
    # validation entirely.

    # Parsed here (CLI-only); merged onto --config-file's own
    # image_normalization/image_normalization_max dicts (CLI wins per-key)
    # further down, once run_cfg exists -- full validation (unknown mode
    # name, missing max for a depth/log camera) happens after that merge,
    # against the FINAL resolved dicts, not these CLI-only ones.
    image_normalization = _parse_kv_pairs(args.image_normalization)
    image_normalization_max = _parse_kv_pairs(args.image_normalization_max, value_type=float)

    profile_steps = None
    if args.profile_steps is not None:
        if not args.log_perf_metrics:
            parser.error("--profile-steps requires --log-perf-metrics")
        try:
            start_str, end_str = args.profile_steps.split(":")
            profile_steps = (int(start_str), int(end_str))
        except ValueError:
            parser.error(f"--profile-steps must look like START:END, got {args.profile_steps!r}")
        if profile_steps[0] >= profile_steps[1]:
            parser.error(f"--profile-steps START must be < END, got {args.profile_steps!r}")
        if profile_steps[1] - profile_steps[0] > 50:
            print(
                f"WARNING: --profile-steps {args.profile_steps} covers "
                f"{profile_steps[1] - profile_steps[0]} steps -- torch.profiler traces get large "
                f"fast, consider a smaller range (10-20 steps is usually plenty)."
            )

    # --- base RunConfig: --config-file's YAML (draccus-parsed) if given,
    # else the plain dataclass defaults -- see _load_base_run_config. Every
    # field below is then layered on top ONLY if its CLI flag was actually
    # passed (_apply_if_explicit), so an unset flag leaves whatever the
    # config-file (or the dataclass default) already had, and today's
    # exact behavior is unchanged when --config-file is omitted.
    run_cfg = _load_base_run_config(args.config_file)
    _apply_if_explicit(run_cfg, "tasks", args, "tasks", parser)
    _apply_if_explicit(run_cfg, "policy_type", args, "policy_type", parser)
    _apply_if_explicit(run_cfg.train, "num_epochs", args, "num_epochs", parser)
    _apply_if_explicit(run_cfg.train, "batch_size", args, "batch_size", parser)
    _apply_if_explicit(run_cfg.train, "lr", args, "lr", parser)
    _apply_if_explicit(run_cfg.train, "lr_backbone", args, "lr_backbone", parser)
    _apply_if_explicit(run_cfg.train, "grad_accum", args, "grad_accum", parser)
    _apply_if_explicit(run_cfg.train, "weight_decay", args, "weight_decay", parser)
    _apply_if_explicit(run_cfg.train, "max_train_steps", args, "max_train_steps", parser)
    _apply_if_explicit(run_cfg.train, "eval_every_steps", args, "eval_every_steps", parser)
    _apply_if_explicit(
        run_cfg.train, "early_stop_patience", args, "early_stop_patience", parser,
        transform=lambda v: v if v > 0 else None,
    )
    _apply_if_explicit(run_cfg.train, "save_only_on_improvement", args, "save_only_on_improvement", parser)
    _apply_if_explicit(run_cfg.train, "checkpoint_max_to_keep", args, "checkpoint_max_to_keep", parser)
    _apply_if_explicit(run_cfg.train, "log_perf_metrics", args, "log_perf_metrics", parser)
    if args.profile_steps is not None:  # already parsed into the (start, end) tuple `profile_steps` above
        run_cfg.train.profile_steps = profile_steps
    _apply_if_explicit(run_cfg.train, "tensorboard", args, "tensorboard", parser)
    _apply_if_explicit(run_cfg.train, "wandb", args, "wandb", parser)
    _apply_if_explicit(run_cfg.train, "wandb_mode", args, "wandb_mode", parser)
    _apply_if_explicit(run_cfg.train, "wandb_project", args, "wandb_project", parser)
    _apply_if_explicit(run_cfg.train, "wandb_entity", args, "wandb_entity", parser)
    _apply_if_explicit(run_cfg.train, "wandb_metrics", args, "wandb_metrics", parser)
    _apply_if_explicit(run_cfg.train, "wandb_metric_groups", args, "wandb_metric_groups", parser)
    _apply_if_explicit(run_cfg.train, "wandb_exclude_metrics", args, "wandb_exclude_metrics", parser)
    _apply_if_explicit(run_cfg.train, "wandb_gif_cameras", args, "wandb_gif_cameras", parser)
    _apply_if_explicit(run_cfg.train, "wandb_gif_every_steps", args, "wandb_gif_every_steps", parser)
    _apply_if_explicit(run_cfg.train, "wandb_gif_frames", args, "wandb_gif_frames", parser)
    _apply_if_explicit(
        run_cfg.train, "wandb_predict_frames_every_steps", args, "wandb_predict_frames_every_steps", parser,
    )
    _apply_if_explicit(run_cfg.train, "eval_split_fraction", args, "eval_split_fraction", parser)
    if run_cfg.train.eval_split_fraction is not None and not (0 < run_cfg.train.eval_split_fraction < 1):
        parser.error(
            f"--eval-split-fraction must be strictly between 0 and 1, got {run_cfg.train.eval_split_fraction}"
        )

    if run_cfg.train.wandb_gif_cameras and not run_cfg.train.wandb:
        parser.error("--wandb-gif-cameras requires --wandb")
    if (
        run_cfg.train.wandb_predict_frames_every_steps
        and run_cfg.train.wandb_predict_frames_every_steps % run_cfg.train.eval_every_steps != 0
    ):
        parser.error(
            f"--wandb-predict-frames-every-steps {run_cfg.train.wandb_predict_frames_every_steps} must be a "
            f"multiple of --eval-every-steps {run_cfg.train.eval_every_steps} -- a predicted-frames GIF can "
            f"only be generated when the eval pass itself runs, so any other value would never fire"
        )
    if run_cfg.train.wandb_metric_groups:
        try:
            from training.wandb_logging import expand_metric_groups

            expand_metric_groups(run_cfg.train.wandb_metric_groups)
        except ValueError as e:
            parser.error(f"{e} (see --list-wandb-metrics)")

    # --- model: make sure run_cfg.model's concrete type actually matches
    # the resolved policy_type before layering any --molmoact2-*/--pi05-*
    # flags onto it. A still-untouched default ACTConfigOverrides means
    # the config-file had no opinion (indistinguishable, after parsing,
    # from never having a `model:` section at all -- both produce exactly
    # this) -- silently build the right type instead, same as today's
    # pre-config-file behavior. Anything else mismatched (the config-file
    # explicitly chose or customized a DIFFERENT policy's model) is a real
    # conflict, not silently resolved.
    expected_model_cls = _POLICY_MODEL_CLASSES[run_cfg.policy_type]
    if not isinstance(run_cfg.model, expected_model_cls):
        if isinstance(run_cfg.model, ACTConfigOverrides) and run_cfg.model == ACTConfigOverrides():
            run_cfg.model = expected_model_cls()
        else:
            raise SystemExit(
                f"--config-file's model section is for a different policy than the resolved "
                f"--policy-type {run_cfg.policy_type!r} -- match --config-file's model.type to "
                f"--policy-type, or drop one of them."
            )

    if run_cfg.policy_type == "molmoact2":
        m = run_cfg.model
        _apply_if_explicit(m, "checkpoint_path", args, "molmoact2_checkpoint_path", parser)
        _apply_if_explicit(m, "setup_type", args, "molmoact2_setup_type", parser)
        _apply_if_explicit(m, "control_mode", args, "molmoact2_control_mode", parser)
        _apply_if_explicit(m, "action_mode", args, "molmoact2_action_mode", parser)
        _apply_if_explicit(m, "train_mode", args, "molmoact2_train_mode", parser)
        _apply_if_explicit(m, "lora_rank", args, "molmoact2_lora_rank", parser)
        _apply_if_explicit(m, "lora_alpha", args, "molmoact2_lora_alpha", parser)
        _apply_if_explicit(m, "lora_dropout", args, "molmoact2_lora_dropout", parser)
        _apply_if_explicit(m, "gradient_checkpointing", args, "molmoact2_gradient_checkpointing", parser)
        _apply_if_explicit(m, "optimizer_vit_lr", args, "molmoact2_vit_lr", parser)
        _apply_if_explicit(m, "optimizer_connector_lr", args, "molmoact2_connector_lr", parser)
        _apply_if_explicit(m, "optimizer_action_expert_lr", args, "molmoact2_action_expert_lr", parser)
        _apply_if_explicit(m, "distributed_strategy", args, "molmoact2_distributed_strategy", parser)
        _apply_if_explicit(m, "fsdp_cpu_offload", args, "molmoact2_fsdp_cpu_offload", parser)
        _apply_if_explicit(m, "offload_tokenization", args, "molmoact2_offload_tokenization", parser)
        if not m.setup_type or not m.control_mode:
            parser.error(
                "--molmoact2-setup-type and --molmoact2-control-mode are required when --policy-type "
                "molmoact2 (either as CLI flags or in --config-file's model section)"
            )
        if m.distributed_strategy == "fsdp2" and m.train_mode != "fft":
            parser.error("--molmoact2-distributed-strategy fsdp2 only makes sense with --molmoact2-train-mode fft")

    if run_cfg.policy_type == "pi05":
        m = run_cfg.model
        _apply_if_explicit(m, "pretrained_path", args, "pi05_pretrained_path", parser)
        _apply_if_explicit(m, "freeze_vision_encoder", args, "pi05_freeze_vision_encoder", parser)
        _apply_if_explicit(m, "train_expert_only", args, "pi05_train_expert_only", parser)
        _apply_if_explicit(m, "gradient_checkpointing", args, "pi05_gradient_checkpointing", parser)
        _apply_if_explicit(m, "empty_cameras", args, "pi05_empty_cameras", parser)
        if not m.pretrained_path:
            parser.error(
                "--pi05-pretrained-path is required when --policy-type pi05 (either as a CLI flag or "
                "in --config-file's model section)"
            )

    # --- data: action-space/image-normalization -- CLI wins per-key for the
    # two dict fields (a config-file's other camera entries are preserved,
    # not wholesale replaced); the scalar fields use the same
    # _apply_if_explicit as everything else above.
    run_cfg.data.image_normalization = {**run_cfg.data.image_normalization, **image_normalization}
    run_cfg.data.image_normalization_max = {**run_cfg.data.image_normalization_max, **image_normalization_max}
    _apply_if_explicit(run_cfg.data, "default_image_normalization", args, "image_normalization_default", parser)
    _apply_if_explicit(run_cfg.data, "action_space", args, "action_space", parser)
    _apply_if_explicit(run_cfg.data, "action_representation", args, "action_representation", parser)
    # argparse choices= doesn't cover a --config-file YAML value -- validate the final result.
    if run_cfg.data.default_image_normalization not in IMAGE_NORMALIZATION_MODES:
        parser.error(f"default image normalization must be one of {IMAGE_NORMALIZATION_MODES}")
    if run_cfg.data.action_representation not in ("absolute", "delta"):
        parser.error("action representation must be 'absolute' or 'delta'")
    _apply_if_explicit(run_cfg.data, "action_delta_exclude", args, "action_delta_exclude", parser)

    _apply_if_explicit(run_cfg, "storage_root", args, "storage_root", parser)
    run_cfg.storage_root = os.path.abspath(run_cfg.storage_root)
    _apply_if_explicit(run_cfg, "run_name", args, "run_name", parser)
    if not run_cfg.run_name:
        run_cfg.run_name = f"{run_cfg.policy_type}-{'-'.join(run_cfg.tasks)}-{time.strftime('%Y%m%d-%H%M%S')}"
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
    original_robot = source.robot
    if run_cfg.data.action_space is not None and run_cfg.data.action_space not in original_robot.action_space_components:
        raise SystemExit(
            f"--action-space {run_cfg.data.action_space!r} isn't available for dataset_source "
            f"{dataset_source!r} -- available: {sorted(original_robot.action_space_components) or '(none)'}"
        )
    run_cfg.data.robot = original_robot.select_action_space(run_cfg.data.action_space)

    for cam, mode in run_cfg.data.image_normalization.items():
        if mode not in IMAGE_NORMALIZATION_MODES:
            raise SystemExit(f"image-normalization {cam}={mode}: unknown mode, must be one of {IMAGE_NORMALIZATION_MODES}")
    # Depth cameras are looked up as "depth_<key>", not the bare RobotSchema name.
    available_cams = set(original_robot.camera_keys) | {f"depth_{k}" for k in original_robot.depth_camera_keys}
    unknown_cams = set(run_cfg.data.image_normalization) - available_cams
    if unknown_cams:
        raise SystemExit(
            f"--image-normalization refers to unknown camera(s) {sorted(unknown_cams)} -- "
            f"available: {sorted(available_cams)}"
        )
    needs_max = {cam for cam, mode in run_cfg.data.image_normalization.items() if mode in ("depth", "log")}
    if run_cfg.data.default_image_normalization in ("depth", "log"):
        needs_max |= available_cams - set(run_cfg.data.image_normalization)
    missing_max = needs_max - set(run_cfg.data.image_normalization_max)
    if missing_max:
        raise SystemExit(
            f"image-normalization mode 'depth'/'log' needs --image-normalization-max for camera(s) "
            f"{sorted(missing_max)}"
        )
    non_positive_max = {cam: v for cam, v in run_cfg.data.image_normalization_max.items() if v <= 0}
    if non_positive_max:
        raise SystemExit(f"--image-normalization-max values must be positive, got {non_positive_max}")

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
    raw_ds = build_lerobot_v3_dataset(
        v3_root, run_cfg.model.chunk_size, depth_camera_keys=run_cfg.data.robot.depth_camera_keys,
    )
    raw_ds = select_action_space(raw_ds, original_robot, run_cfg.data.robot, name=run_cfg.run_name)
    if run_cfg.data.action_representation == "delta":
        raw_ds = to_relative_action_space(
            raw_ds, run_cfg.data.robot, exclude_components=run_cfg.data.action_delta_exclude,
            name=run_cfg.run_name,
        )

    # Held-out eval split (--eval-split-fraction, opt-in) -- held out of
    # TRAINING entirely, including normalization stats below (computing
    # stats from episodes the eval pass then scores against would leak
    # information the held-out set is supposed to be free of). Real
    # eval_episode_indices computed directly from meta/episodes (the same
    # deterministic hash split_by_episode uses, done here without touching
    # Ray Data at all -- cheap, driver-side) so train_loop.py's GIF
    # sampling can pick from it without re-querying the dataset.
    eval_ds = None
    eval_episode_indices = None
    if run_cfg.train.eval_split_fraction:
        from training.data.ray_dataset import _episode_in_eval, split_by_episode
        from training.vendor.lerobot_datasource import LeRobotDatasourceMetadata

        meta = LeRobotDatasourceMetadata(v3_root)
        all_episode_indices = meta.episodes.column("episode_index").to_pylist()
        eval_episode_indices = [
            idx for idx in all_episode_indices
            if _episode_in_eval({"episode_index": idx}, seed=0, eval_fraction=run_cfg.train.eval_split_fraction)
        ]
        # A valid fraction can still hash every episode of a small dataset to one side.
        if not eval_episode_indices or len(eval_episode_indices) == len(all_episode_indices):
            raise SystemExit(
                f"--eval-split-fraction {run_cfg.train.eval_split_fraction} hashed all "
                f"{len(all_episode_indices)} episode(s) to one side (0 held out for eval) -- "
                f"try a different fraction, or use more episodes."
            )
        print(
            f"  eval split: {len(eval_episode_indices)}/{len(all_episode_indices)} episodes held out "
            f"of training (--eval-split-fraction {run_cfg.train.eval_split_fraction})"
        )
        raw_ds, eval_raw_ds = split_by_episode(raw_ds, run_cfg.train.eval_split_fraction, seed=0)

    print("\n=== compute normalization stats ===")
    # Sampled from raw_ds (pre-transpose, HWC), AFTER action-space selection/
    # delta-conversion/eval-split above -- so stats reflect whatever's
    # actually trained on (computing stats on absolute actions then
    # converting to delta afterwards would normalize deltas using
    # absolute-action mean/std, which is wrong; same reasoning for
    # excluding the held-out eval episodes). See training/data/stats.py.
    dataset_stats = compute_dataset_stats(raw_ds, run_cfg.data)

    ds = transpose_for_training(
        raw_ds, run_cfg.data.robot.camera_keys, name=run_cfg.run_name,
        depth_camera_keys=run_cfg.data.robot.depth_camera_keys,
    )
    if eval_episode_indices is not None:
        eval_ds = transpose_for_training(
            eval_raw_ds, run_cfg.data.robot.camera_keys, name=f"{run_cfg.run_name}-eval",
            depth_camera_keys=run_cfg.data.robot.depth_camera_keys,
        )

    if run_cfg.policy_type == "molmoact2" and run_cfg.model.offload_tokenization:
        print("\n=== offload MolmoAct2 preprocessing (tokenizer + image processor) to Ray Data ===")
        concurrency = args.molmoact2_offload_concurrency or max(1, int(resources.get("CPU", 1)) // 2)
        ds = offload_molmoact2_preprocessing(
            ds, run_cfg.data, run_cfg.model, run_cfg.train, dataset_stats,
            concurrency=concurrency, name=run_cfg.run_name,
        )
        if eval_ds is not None:
            eval_ds = offload_molmoact2_preprocessing(
                eval_ds, run_cfg.data, run_cfg.model, run_cfg.train, dataset_stats,
                concurrency=concurrency, name=f"{run_cfg.run_name}-eval",
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

    datasets = {"train": ds}
    if eval_ds is not None:
        datasets["eval"] = eval_ds

    print(f"\n=== launch Ray Train: {run_cfg.run_name} ===")
    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "run_cfg": run_cfg, "dataset_stats": dataset_stats,
            # v3_root/eval_episode_indices aren't part of RunConfig's own
            # schema (internal plumbing for training/wandb_logging.py's
            # sample_episode_frames, not user-facing config) -- threaded
            # through train_loop_config the same way dataset_stats already is.
            "v3_root": v3_root, "eval_episode_indices": eval_episode_indices,
        },
        scaling_config=ray.train.ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=ray.train.RunConfig(
            name=run_cfg.run_name,
            storage_path=run_cfg.storage_root,
            failure_config=ray.train.FailureConfig(max_failures=1),
            checkpoint_config=checkpoint_config,
        ),
        datasets=datasets,
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
    if run_cfg.train.tensorboard:
        print(f"View TensorBoard: tensorboard --logdir {os.path.join(run_cfg.storage_root, run_cfg.run_name, 'tensorboard')}")


if __name__ == "__main__":
    main()
