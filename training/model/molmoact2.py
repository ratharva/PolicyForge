"""Build lerobot's MolmoAct2Config/MolmoAct2Policy from data-derived dims,
mirroring training/model/act.py's contract.

MolmoAct2 is bundled in stock PyPI `lerobot>=0.6.1`, same package/
environment as ACT -- no fork or separate environment needed.
"""
from __future__ import annotations

import json
import os

import torch

from training.common.config import DataConfig
from training.config import MolmoAct2ConfigOverrides, TrainConfig


def build_molmoact2_config(
    data_cfg: DataConfig, overrides: MolmoAct2ConfigOverrides, train_cfg: TrainConfig, device: str = "cpu",
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    if not overrides.setup_type or not overrides.control_mode:
        raise ValueError(
            "MolmoAct2ConfigOverrides.setup_type and .control_mode are required free-text prompt "
            "fields describing this robot's embodiment/control convention -- set via "
            "--molmoact2-setup-type / --molmoact2-control-mode. They condition MolmoAct2's VLM "
            "prompt (MolmoAct2PackInputsProcessorStep hard-requires them), unlike ACT which has no "
            "language-conditioning input at all."
        )
    if overrides.distributed_strategy == "fsdp2" and overrides.train_mode != "fft":
        raise ValueError(
            f"distributed_strategy='fsdp2' only makes sense with train_mode='fft' "
            f"(got train_mode={overrides.train_mode!r}) -- LoRA's whole point is fitting "
            "under plain DDP without FSDP2's added complexity; see README's FSDP2 section."
        )

    # train_mode is our own CLI-level convenience concept, translated to the
    # real enable_lora_vlm/enable_lora_action_expert/train_action_expert_only
    # booleans:
    #   "lora"   -> LoRA on the VLM only; action expert stays fully trainable.
    #   "fft"    -> full fine-tune, no LoRA, no freezing.
    #   "freeze" -> VLM frozen, only the action expert trains (requires
    #               action_mode="continuous").
    if overrides.train_mode == "lora":
        enable_lora_vlm, enable_lora_action_expert, train_action_expert_only = True, False, False
    elif overrides.train_mode == "fft":
        enable_lora_vlm, enable_lora_action_expert, train_action_expert_only = False, False, False
    elif overrides.train_mode == "freeze":
        enable_lora_vlm, enable_lora_action_expert, train_action_expert_only = False, False, True
    else:
        raise ValueError(f"train_mode must be one of 'lora', 'fft', 'freeze', got {overrides.train_mode!r}")

    from training.model.image_normalization import visual_input_features

    input_features = visual_input_features(data_cfg)
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(data_cfg.robot.state_dim,)
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(data_cfg.robot.action_dim,))
    }

    # image_keys (distinct from input_features, which just declares dims)
    # is the ONLY thing that controls which cameras MolmoAct2's own
    # processor actually extracts and feeds to the vision tower/VLM prompt
    # -- confirmed by reading processor_molmoact2.py's _resolve_image_keys
    # directly. The default below (RGB camera_keys only) silently drops
    # any depth camera: it's still declared as an input_features VISUAL
    # entry, so it gets normalized/transferred for nothing. Fail fast
    # instead of guessing this is safe -- an explicit --config-file
    # model.image_keys (including depth) is the deliberate opt-in once
    # someone's verified it end to end against a real checkpoint.
    if overrides.image_keys is None and data_cfg.robot.depth_camera_keys:
        raise ValueError(
            f"this dataset has depth camera(s) {data_cfg.robot.depth_camera_keys} but MolmoAct2's "
            f"image_keys would default to RGB-only cameras {data_cfg.robot.camera_keys}, silently "
            f"never feeding depth to the model -- pass model.image_keys explicitly via --config-file "
            f"(including observation.images.depth_<key> for each one) to opt in, or use a dataset "
            f"without depth cameras for --policy-type molmoact2."
        )
    image_keys = overrides.image_keys or [f"observation.images.{k}" for k in data_cfg.robot.camera_keys]

    return MolmoAct2Config(
        input_features=input_features,
        output_features=output_features,
        device=device,
        checkpoint_path=overrides.checkpoint_path,
        chunk_size=overrides.chunk_size,
        n_action_steps=overrides.n_action_steps,
        action_mode=overrides.action_mode,
        num_flow_timesteps=overrides.num_flow_timesteps,
        enable_lora_vlm=enable_lora_vlm,
        enable_lora_action_expert=enable_lora_action_expert,
        train_action_expert_only=train_action_expert_only,
        lora_rank=overrides.lora_rank,
        lora_alpha=overrides.lora_alpha,
        lora_dropout=overrides.lora_dropout,
        lora_bias=overrides.lora_bias,
        gradient_checkpointing=overrides.gradient_checkpointing,
        model_dtype=overrides.dtype,
        image_keys=image_keys,
        setup_type=overrides.setup_type,
        control_mode=overrides.control_mode,
        # STATE/ACTION: MEAN_STD (not the real default IDENTITY/QUANTILES) so
        # training/data/stats.py's mean/std-only compute_dataset_stats works
        # unchanged. VISUAL: IDENTITY -- training/model/image_normalization.py
        # now owns all image scaling instead (applied in train_loop.py/
        # offload_molmoact2_preprocessing before this policy's preprocessor
        # runs), so lerobot's own per-FeatureType normalizer must not also
        # scale images.
        normalization_mapping={"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        optimizer_lr=train_cfg.lr,
        optimizer_vit_lr=overrides.optimizer_vit_lr or train_cfg.lr,
        optimizer_connector_lr=overrides.optimizer_connector_lr or train_cfg.lr,
        optimizer_action_expert_lr=overrides.optimizer_action_expert_lr or train_cfg.lr,
        # Not consumed by this pipeline's optimizer construction (train_loop.py
        # builds AdamW directly from get_optim_params() + train_cfg, bypassing
        # get_optimizer_preset()) -- set for documentation/future-proofing only.
        optimizer_weight_decay=train_cfg.weight_decay,
    )


def build_policy_and_processor(
    data_cfg: DataConfig, overrides: MolmoAct2ConfigOverrides, train_cfg: TrainConfig,
    dataset_stats: dict, device: str = "cpu",
):
    """Returns (policy, preprocessor) -- same contract as act.py's function
    of the same name. Apply `preprocessor(batch)` to every batch before
    `policy(batch)`, unless overrides.offload_tokenization is set, in which
    case train_loop.py skips it because the Ray Data pipeline already did it
    upstream."""
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
    from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors

    cfg = build_molmoact2_config(data_cfg, overrides, train_cfg, device=device)
    policy = MolmoAct2Policy(cfg)
    preprocessor, _postprocessor = make_molmoact2_pre_post_processors(cfg, dataset_stats=dataset_stats)
    return policy, preprocessor


def forward_loss(policy, inputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics = policy(inputs)
    return loss, {k: float(v) for k, v in metrics.items()}


def enable_training_optimizations(policy, overrides: MolmoAct2ConfigOverrides) -> None:
    """post_build_hook: called right after construction, before any
    distributed wrapping -- gradient checkpointing must be enabled before
    DDP/FSDP2 registers hooks over the final module structure."""
    if overrides.gradient_checkpointing and hasattr(policy, "_enable_gradient_checkpointing"):
        policy._enable_gradient_checkpointing()


# ---------------------------------------------------------------------------
# Full-finetune / FSDP2 path (overrides.distributed_strategy == "fsdp2")
# ---------------------------------------------------------------------------
# Only reachable when train_mode="fft" (validated in build_molmoact2_config
# above). Uses accelerate's own FSDP2 support rather than
# ray.train.torch.prepare_model's "fsdp" option, which wraps legacy FSDP1.


def wrap_for_training(policy, optimizer, overrides: MolmoAct2ConfigOverrides, device):
    """Returns (policy, optimizer, accelerator). Use accelerator.unwrap_model()
    in place of the DDP `.module` pattern -- accelerate doesn't proxy custom
    methods like get_optim_params() through its wrapper either."""
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    # MolmoAct2DecoderLayer/MolmoAct2PostNormDecoderLayer are mutually
    # exclusive per checkpoint. accelerate's class-name matching is exact
    # type(module).__name__ ==, not isinstance-based, so listing a name
    # that isn't present raises -- filter the candidates down to what's
    # actually present on the constructed model instead of hardcoding.
    present_classes = {type(m).__name__ for m in policy.modules()}
    candidate_wrap_classes = [
        "MolmoAct2DecoderLayer", "MolmoAct2PostNormDecoderLayer",
        "MolmoAct2VisionBlock", "ActionExpertBlock",
    ]
    transformer_cls_names_to_wrap = [c for c in candidate_wrap_classes if c in present_classes]
    if not transformer_cls_names_to_wrap:
        raise RuntimeError(
            f"None of the candidate FSDP2 wrap classes {candidate_wrap_classes} were found on the "
            f"constructed policy (found classes: {sorted(present_classes)}) -- the installed "
            "MolmoAct2 model's internal class names have likely changed; update candidate_wrap_classes "
            "in wrap_for_training to match."
        )

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,  # accelerate defaults this to 1 (legacy FSDP1) -- must be explicit
        reshard_after_forward=True,  # bool for fsdp_version=2, NOT a torch.distributed.fsdp.ShardingStrategy
        cpu_offload=overrides.fsdp_cpu_offload,
        state_dict_type=overrides.fsdp_state_dict_type,  # "sharded_state_dict" (lowercase)
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=transformer_cls_names_to_wrap,
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin, mixed_precision="bf16")
    policy, optimizer = accelerator.prepare(policy, optimizer)
    return policy, optimizer, accelerator


def save_checkpoint(accelerator, out_dir: str, epoch: int, step: int, epoch_complete: bool) -> None:
    """Every rank must call this -- FSDP2-sharded save needs all ranks'
    shards, unlike ACT's rank-0-only pickle save.

    epoch_complete distinguishes a step-windowed (mid-epoch) checkpoint from
    an end-of-epoch one; train_loop.py's resume logic needs this to compute
    the correct start_epoch."""
    accelerator.save_state(out_dir)
    if accelerator.is_main_process:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump({"epoch": epoch, "step": step, "epoch_complete": epoch_complete}, f)


def load_checkpoint(accelerator, in_dir: str) -> dict:
    accelerator.load_state(in_dir)
    with open(os.path.join(in_dir, "meta.json")) as f:
        return json.load(f)
