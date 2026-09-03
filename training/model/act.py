"""Build lerobot's ACTConfig/ACTPolicy from data-derived dims.

lerobot 0.6.1 separates normalization into a processor pipeline --
ACTPolicy.forward() does not normalize internally, so
build_policy_and_processor() returns both the policy and the preprocessor
that must be applied to every batch before it's fed to the policy.
"""
from __future__ import annotations

import torch

from training.config import ACTConfigOverrides, DataConfig, TrainConfig


def build_act_config(
    data_cfg: DataConfig, overrides: ACTConfigOverrides, train_cfg: TrainConfig, device: str = "cpu",
):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig

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

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=device,  # explicit to avoid lerobot's implicit GPU auto-resolution
        chunk_size=overrides.chunk_size,
        n_action_steps=overrides.n_action_steps,
        vision_backbone=overrides.vision_backbone,
        pretrained_backbone_weights=overrides.pretrained_backbone_weights,
        dim_model=overrides.dim_model,
        n_heads=overrides.n_heads,
        dim_feedforward=overrides.dim_feedforward,
        n_encoder_layers=overrides.n_encoder_layers,
        n_decoder_layers=overrides.n_decoder_layers,
        use_vae=overrides.use_vae,
        latent_dim=overrides.latent_dim,
        n_vae_encoder_layers=overrides.n_vae_encoder_layers,
        kl_weight=overrides.kl_weight,
        dropout=overrides.dropout,
        # Must be set here for policy.get_optim_params() to honor CLI
        # --lr/--lr-backbone/--weight-decay overrides.
        optimizer_lr=train_cfg.lr,
        optimizer_lr_backbone=train_cfg.lr_backbone,
        optimizer_weight_decay=train_cfg.weight_decay,
    )


def build_policy_and_processor(
    data_cfg: DataConfig, overrides: ACTConfigOverrides, train_cfg: TrainConfig,
    dataset_stats: dict, device: str = "cpu",
):
    """Returns (policy, preprocessor). Apply `preprocessor(batch)` to every
    batch before `policy(batch)`."""
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    cfg = build_act_config(data_cfg, overrides, train_cfg, device=device)
    policy = ACTPolicy(cfg)
    preprocessor, _postprocessor = make_act_pre_post_processors(cfg, dataset_stats)
    return policy, preprocessor


def forward_loss(policy, inputs: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """loss_dict values are already plain floats. 'kld_loss' is present only
    when config.use_vae is True."""
    loss, loss_dict = policy(inputs)
    return loss, {"l1_loss": loss_dict.get("l1_loss", 0.0), "kld_loss": loss_dict.get("kld_loss", 0.0)}
