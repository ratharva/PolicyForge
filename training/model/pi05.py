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

    h, w = data_cfg.image_size
    input_features = {
        f"observation.images.{k}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, h, w))
        for k in data_cfg.robot.camera_keys
    }
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
        # MEAN_STD (not the real default IDENTITY/QUANTILES), same
        # deliberate simplification as MolmoAct2, so
        # training/data/stats.py's mean/std-only compute_dataset_stats
        # works unchanged.
        normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
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
    return loss, {k: float(v) for k, v in metrics.items()}
