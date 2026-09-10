"""Build lerobot's PI05Config/PI05Policy from data-derived dims, mirroring
training/model/act.py's/molmoact2.py's contract.

Bundled in the same `lerobot==0.6.1` install as ACT/MolmoAct2 -- no fork, no
separate environment.

Real differences from MolmoAct2 worth knowing before touching this file:
  - No `checkpoint_path` config field -- pretrained weights load via
    `PreTrainedPolicy.from_pretrained(path, config=cfg)`, NOT `PI05Policy(cfg)`
    (which would randomly initialize).
  - `get_optim_params()` returns `self.parameters()` directly -- flat and
    ungrouped, no per-component LR split.
  - Gradient checkpointing auto-wires from `config.gradient_checkpointing`
    at construction time -- no post_build_hook needed, unlike MolmoAct2.
  - No LoRA/peft support -- only freeze_vision_encoder/train_expert_only
    reduce the trainable surface.
"""
from __future__ import annotations

import torch

from training.common.config import DataConfig
from training.config import Pi05ConfigOverrides, TrainConfig


def build_pi05_config(
    data_cfg: DataConfig, overrides: Pi05ConfigOverrides, train_cfg: TrainConfig, device: str = "cpu",
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if not overrides.pretrained_path:
        raise ValueError(
            "Pi05ConfigOverrides.pretrained_path is required -- 'finetuning' implies starting "
            "from real pretrained weights, and there's no verified real public checkpoint repo "
            "id to default to here. Find a real pi05 checkpoint on the HF Hub and pass it via "
            "--pi05-pretrained-path."
        )

    from training.model.image_normalization import visual_input_features

    h, w = data_cfg.image_size
    input_features = visual_input_features(data_cfg)
    input_features["observation.state"] = PolicyFeature(
        type=FeatureType.STATE, shape=(data_cfg.robot.state_dim,)
    )
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(data_cfg.robot.action_dim,))
    }

    return PI05Config(
        input_features=input_features,
        output_features=output_features,
        device=device,
        chunk_size=overrides.chunk_size,
        n_action_steps=overrides.n_action_steps,
        freeze_vision_encoder=overrides.freeze_vision_encoder,
        train_expert_only=overrides.train_expert_only,
        gradient_checkpointing=overrides.gradient_checkpointing,
        empty_cameras=overrides.empty_cameras,
        image_resolution=(h, w),
        dtype=overrides.dtype,
        # STATE/ACTION: MEAN_STD (not the real default IDENTITY/QUANTILES),
        # same deliberate simplification as MolmoAct2, so
        # training/data/stats.py's mean/std-only compute_dataset_stats
        # works unchanged. VISUAL: IDENTITY -- training/model/
        # image_normalization.py now owns all image scaling instead
        # (applied in train_loop.py before this policy's preprocessor
        # runs), so lerobot's own per-FeatureType normalizer must not also
        # scale images.
        normalization_mapping={"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        optimizer_lr=train_cfg.lr,
        # Not consumed by this pipeline's optimizer construction (train_loop.py
        # builds AdamW directly from get_optim_params() + train_cfg, bypassing
        # get_optimizer_preset()) -- set for documentation/future-proofing only.
        optimizer_weight_decay=train_cfg.weight_decay,
    )


def build_policy_and_processor(
    data_cfg: DataConfig, overrides: Pi05ConfigOverrides, train_cfg: TrainConfig,
    dataset_stats: dict, device: str = "cpu",
):
    """Returns (policy, preprocessor) -- same contract as act.py's/
    molmoact2.py's function of the same name. Loads pretrained weights via
    PI05Policy.from_pretrained(...), NOT PI05Policy(cfg) directly (which
    would randomly initialize instead of finetuning)."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

    cfg = build_pi05_config(data_cfg, overrides, train_cfg, device=device)
    policy = PI05Policy.from_pretrained(overrides.pretrained_path, config=cfg)
    preprocessor, _postprocessor = make_pi05_pre_post_processors(cfg, dataset_stats=dataset_stats)
    return policy, preprocessor


def forward_loss(policy, inputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
    loss, metrics = policy(inputs)
    # PI05Policy.forward() returns loss_per_dim as a real list (one entry per
    # action dim), not a scalar -- confirmed via its real installed source
    # (modeling_pi05.py). Expand any list/tuple-valued metric into named
    # per-index scalars instead of dropping it; it's genuinely useful
    # (which action dims are harder to predict), and this handles any future
    # list-valued metric generically, not just this one key.
    scalars: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):
            for i, vi in enumerate(v):
                scalars[f"{k}_{i}"] = float(vi)
        else:
            scalars[k] = float(v)
    return loss, scalars


# ---------------------------------------------------------------------------
# Full-finetune / FSDP2 path (overrides.distributed_strategy == "fsdp2")
# ---------------------------------------------------------------------------
# Uses accelerate's own FSDP2 support, same mechanism as model/molmoact2.py.
# Unlike MolmoAct2, NOT restricted to any particular
# freeze_vision_encoder/train_expert_only combination -- PI05 has no LoRA
# to already solve memory (which is why MolmoAct2 restricts fsdp2 to
# train_mode=="fft"), and DDP always fully replicates the whole model
# regardless of what's frozen, so FSDP2's memory-sharding benefit applies
# to every PI05 training mode.


def wrap_for_training(policy, optimizer, overrides: Pi05ConfigOverrides, device):
    """Returns (policy, optimizer, accelerator). Use accelerator.unwrap_model()
    in place of the DDP `.module` pattern."""
    from accelerate import Accelerator
    from accelerate.utils import FullyShardedDataParallelPlugin

    # _PiGemmaDecoderLayerBase covers BOTH the PaliGemma VLM's language_model
    # layers and the separately-instantiated gemma_expert.model's layers --
    # both are built by the same locally-scoped factory
    # (lerobot/policies/pi05/pi_gemma.py's _get_pi_gemma_decoder_layer_base),
    # so type(m).__name__ is identical for both despite being different
    # Python class objects per call -- confirmed via the real installed
    # lerobot==0.6.1 source, not guessed. SiglipEncoderLayer covers the
    # vision tower (a real HF SiglipVisionModel). Filtered against what's
    # actually present, same defensive pattern as molmoact2.py's
    # wrap_for_training -- accelerate's class-name matching is exact
    # type(module).__name__ ==, not isinstance-based.
    present_classes = {type(m).__name__ for m in policy.modules()}
    candidate_wrap_classes = ["_PiGemmaDecoderLayerBase", "SiglipEncoderLayer"]
    transformer_cls_names_to_wrap = [c for c in candidate_wrap_classes if c in present_classes]
    if not transformer_cls_names_to_wrap:
        raise RuntimeError(
            f"None of the candidate FSDP2 wrap classes {candidate_wrap_classes} were found on the "
            f"constructed policy (found: {sorted(present_classes)}) -- the installed PI05 model's "
            "internal class names have likely changed; update candidate_wrap_classes."
        )

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        cpu_offload=overrides.fsdp_cpu_offload,
        state_dict_type=overrides.fsdp_state_dict_type,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=transformer_cls_names_to_wrap,
    )
    # No mixed_precision= kwarg, unlike MolmoAct2's hardcoded
    # mixed_precision="bf16" -- PI05 already hand-casts a mixed bf16/fp32
    # scheme onto individual params at construction time
    # (PaliGemmaWithExpertModel.to_bfloat16_for_selected_params, called
    # from build_policy_and_processor before this ever runs, keeping
    # vision_tower/multi_modal_projector/layernorms in float32 for
    # stability). Passing mixed_precision="bf16" here too would
    # blanket-recast everything back to bf16 and undo that split.
    # UNVERIFIED (no GPU-capable dev environment here): that these
    # fp32-designated params actually survive accelerator.prepare() still
    # showing dtype=torch.float32 -- confirm via named_parameters() before
    # trusting a real run, see training/README.md's FSDP2 section.
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    return policy, optimizer, accelerator


# save_checkpoint/load_checkpoint: fully generic over accelerator/paths, no
# PI05-specific logic -- shared with molmoact2.py's own FSDP2 path instead
# of duplicated.
from training.model.fsdp2_checkpoint import load_checkpoint, save_checkpoint  # noqa: E402,F401
