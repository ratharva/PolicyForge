"""Training-time configuration: hyperparameters and per-policy overrides.
DataConfig (the dataset/robot description shared with data prep) lives in
training/common/config.py -- see that module and
training/data_prep/schema_loader.py for how a --dataset-source populates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from training.common.config import DataConfig


@dataclass
class ACTConfigOverrides:
    """Values passed to lerobot's ACTConfig -- field names must match the
    installed lerobot version; see training/model/act.py."""
    chunk_size: int = 100
    n_action_steps: int = 100
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    dropout: float = 0.1


@dataclass
class MolmoAct2ConfigOverrides:
    """Values passed to lerobot's MolmoAct2Config; see training/model/molmoact2.py."""
    checkpoint_path: str = "allenai/MolmoAct2"
    chunk_size: int = 30          # MolmoAct2Config's own default -- NOT ACT's 100
    n_action_steps: int = 30
    action_mode: str = "continuous"  # real default is "both"; narrowed to skip the
                                      # discrete FAST-tokenizer dependency entirely
    num_flow_timesteps: int = 8
    # "lora" | "fft" | "freeze" -- our own CLI-level convenience concept, not a
    # real MolmoAct2Config field. Translated to the real enable_lora_vlm/
    # enable_lora_action_expert/train_action_expert_only booleans in
    # model/molmoact2.py's build_molmoact2_config.
    train_mode: str = "lora"
    lora_rank: int = 64    # MolmoAct2Config's own default
    lora_alpha: int = 16   # MolmoAct2Config's own default (alpha < rank here)
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    gradient_checkpointing: bool = True   # our own default; real MolmoAct2Config default is False
    image_keys: list[str] | None = None   # None -> derived from data_cfg.robot.camera_keys at build time
    setup_type: str = ""     # REQUIRED free-text embodiment prompt, validated non-empty at build time
    control_mode: str = ""   # REQUIRED free-text control-mode prompt, validated non-empty at build time

    # Per-group learning rates -- MolmoAct2's get_optim_params() returns 4
    # groups (vlm/vit/connector/action_expert), unlike ACT's 2. None falls
    # back to TrainConfig.lr at build time.
    optimizer_vit_lr: float | None = None
    optimizer_connector_lr: float | None = None
    optimizer_action_expert_lr: float | None = None

    # --- Full-finetune / FSDP2 -- only sane with train_mode="fft" ---
    distributed_strategy: str = "ddp"   # "ddp" | "fsdp2"
    fsdp_cpu_offload: bool = False      # trades speed for fitting on fewer/smaller GPUs
    fsdp_state_dict_type: str = "sharded_state_dict"  # accelerate's own default is "full_state_dict",
                                                       # which gathers the whole model onto one rank

    # --- Ray Data preprocessing offload (training/data/ray_dataset.py) ---
    offload_tokenization: bool = False


@dataclass
class Pi05ConfigOverrides:
    """Values passed to lerobot's PI05Config; see training/model/pi05.py.

    Deliberately no single "train_mode" convenience string like
    MolmoAct2ConfigOverrides -- PI05Config exposes two independent booleans
    directly (freeze_vision_encoder, train_expert_only) and has no LoRA/peft
    support.
    """
    # REQUIRED, no safe default -- find a real pretrained checkpoint on the
    # HF Hub and pass it via --pi05-pretrained-path.
    pretrained_path: str = ""
    chunk_size: int = 50          # PI05Config's own default
    n_action_steps: int = 50      # PI05Config's own default
    freeze_vision_encoder: bool = False   # freezes the vision tower only
    train_expert_only: bool = False       # only the action expert trains
    gradient_checkpointing: bool = True   # our own default; real PI05Config default is False
    # Pads input_features with dummy observation.images.empty_camera_{i}
    # VISUAL features up to a target camera count, for when a pretrained
    # checkpoint expects more camera slots than this dataset has.
    empty_cameras: int = 0


@dataclass
class TrainConfig:
    num_epochs: int = 1
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-5
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    max_train_steps: int | None = None
    # Step-windowed reporting: finer-grained than per-epoch, so early
    # stopping/best-checkpoint scoring get more than one data point even
    # with num_epochs=1.
    eval_every_steps: int = 200
    # Stop once the windowed loss hasn't improved for this many windows in a
    # row. None disables early stopping.
    early_stop_patience: int | None = 5
    # False (default): a checkpoint is written on every report, scored by
    # loss. Ray Train's checkpoint manager keeps the best N by score PLUS
    # the single most recently written one, which is what a resume picks up
    # from. True: only checkpoint when the loss improves -- less write I/O,
    # but a resume can lose progress back to the last improvement.
    save_only_on_improvement: bool = False
    checkpoint_max_to_keep: int = 3

    # Opt-in perf instrumentation (training/perf_logging.py) -- off by
    # default so a normal run pays zero cost: accurate step-timing needs
    # torch.cuda.synchronize() calls, which serialize async CUDA work and
    # cost real throughput whenever they're on. See train_loop.py.
    log_perf_metrics: bool = False
    # (start_step, end_step) inclusive range to capture a real
    # torch.profiler trace for -- only meaningful when log_perf_metrics is
    # also True. None disables profiling entirely.
    profile_steps: tuple[int, int] | None = None


@dataclass
class RunConfig:
    tasks: list[str] = field(default_factory=list)
    max_episodes_per_task: int = 300
    run_name: str | None = None
    # Kept inside training/ deliberately -- train.py's runtime_env excludes
    # it from the working_dir upload by its resolved path (see
    # build_runtime_env in train.py); a path outside the repo root can miss
    # that exclude and sweep checkpoints into the upload past Ray's 512MiB
    # package cap. On a shared cluster, override to shared storage (e.g. a
    # mounted network filesystem) so a worker restart on a different node
    # can still see checkpoints.
    storage_root: str = "training/runs"
    # "act" (default), "molmoact2", or "pi05" -- selects which policy
    # model/registry.py builds; train.py's main() overwrites `model` with
    # the right override type once --policy-type is parsed.
    policy_type: str = "act"
    data: DataConfig = field(default_factory=DataConfig)
    model: ACTConfigOverrides | MolmoAct2ConfigOverrides | Pi05ConfigOverrides = field(default_factory=ACTConfigOverrides)
    train: TrainConfig = field(default_factory=TrainConfig)
