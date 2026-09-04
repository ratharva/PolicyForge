# Training -- `train.py`

Reads an already-prepared LeRobot v3 dataset and trains a policy on it.
Assumes `training/prepare_data.py` has already been run for `--tasks` --
this script does no discover/download/convert of its own; if no prepared
dataset is found at `--v3-root`, it exits with the `prepare_data.py`
command to run first. See [`training/data_prep/README.md`](data_prep/README.md)
for that step.

```bash
python -m training.prepare_data --tasks arrange_the_flowers --max-episodes-per-task 300
python -m training.train --tasks arrange_the_flowers --max-train-steps 50
```

## General flags

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | must match what `prepare_data.py` was run with |
| `--run-name` | `<policy_type>-<tasks>-<timestamp>` | also the resume key -- see below |
| `--storage-root` | `TrainConfig` default | where run output (checkpoints, TensorBoard, history) is written |
| `--num-epochs` | `1` | |
| `--batch-size` | `8` | |
| `--max-train-steps` | none (full epoch) | cap total steps -- for a smoke run |
| `--eval-every-steps` | `200` | report/checkpoint/early-stop-check granularity |
| `--early-stop-patience` | `5` | stop after this many eval windows with no loss improvement; `0` or negative disables it |
| `--save-only-on-improvement` | off | only checkpoint when loss improves (less write I/O, but a resume can lose progress back to the last improvement) |
| `--checkpoint-max-to-keep` | `3` | keeps the N best-scoring checkpoints plus the single most recent one (native Ray Train `CheckpointConfig` behavior) |
| `--num-workers` | live GPU count | Ray Train DDP worker count |
| `--v3-root` | matches `prepare_data.py`'s own default for these `--tasks` | must already exist |
| `--dataset-source` | read from the prepared dataset's own `conversion_params.json` | override only needed for a hand-built `--v3-root` with no `conversion_params.json` |
| `--policy-type` | `act` | `act`, `molmoact2`, or `pi05` |

**Not currently CLI-configurable**: `lr`/`lr_backbone`/`grad_accum`/
`weight_decay` are fixed at `TrainConfig`'s dataclass defaults (`1e-5`,
`1e-5`, `1`, `1e-4`) -- there is no `--lr` flag today, despite the
MolmoAct2 per-group LR flags' help text referencing one as a fallback. Set
these by editing `TrainConfig` directly if you need to change them.

## Resuming a run

There's no `--resume` flag -- re-running with the **same `--run-name`**
(and `--storage-root`) resumes automatically: Ray Train detects the
existing run directory and its checkpoints and picks up from the most
recent one. Pass `--run-name` explicitly if you want to resume later
(the default name embeds a timestamp, so a fresh invocation without it
always starts a new run).

```bash
python -m training.train --tasks dress_the_teddy_bear --run-name teddy_bear_act_v1 --max-train-steps 500
# ...interrupted or crashed...
python -m training.train --tasks dress_the_teddy_bear --run-name teddy_bear_act_v1 --max-train-steps 500
```

## ACT (default)

```bash
# Smoke run
python -m training.train --tasks dress_the_teddy_bear --max-train-steps 20

# Full run over the data, larger batch
python -m training.train --tasks dress_the_teddy_bear --batch-size 16 --num-epochs 5
```

## MolmoAct2

`--policy-type molmoact2` -- a ~8B-param VLA. `--molmoact2-setup-type` and
`--molmoact2-control-mode` are **required**.

| Flag | Default | Meaning |
|---|---|---|
| `--molmoact2-checkpoint-path` | `allenai/MolmoAct2` | HF repo id or local path the VLM backbone loads from |
| `--molmoact2-setup-type` | -- (required) | free-text embodiment prompt, e.g. `"dual-arm robot with wrist and top cameras"` |
| `--molmoact2-control-mode` | -- (required) | free-text control-mode prompt, e.g. `"delta joint position"` |
| `--molmoact2-action-mode` | `continuous` | `continuous`, `discrete`, or `both` -- real default is `both`; narrowed here to skip the discrete FAST-tokenizer setup |
| `--molmoact2-train-mode` | `lora` | `lora` (LoRA on the VLM, action expert stays fully trainable -- fits under plain DDP, ~20GB at batch 8), `fft` (full fine-tune, needs `--molmoact2-distributed-strategy fsdp2` to be practical, ~60GB/GPU at batch 32 under plain DDP otherwise), or `freeze` (VLM frozen, only the action expert trains -- requires `--molmoact2-action-mode continuous`) |
| `--molmoact2-lora-rank` / `-alpha` / `-dropout` | `64` / `16` / `0.05` | `MolmoAct2Config`'s own real defaults (yes, alpha < rank) |
| `--molmoact2-no-gradient-checkpointing` | off (checkpointing on) | disable only with confirmed memory headroom |
| `--molmoact2-vit-lr` / `-connector-lr` / `-action-expert-lr` | none -> falls back to the fixed base `lr` | MolmoAct2's `get_optim_params()` returns 4 LR groups (vlm/vit/connector/action_expert), not ACT's flat params |
| `--molmoact2-distributed-strategy` | `ddp` | `ddp` (only sane with `lora`) or `fsdp2` (accelerate-driven FSDP2 sharding inside the Ray Train worker, for `fft` at scale) |
| `--molmoact2-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |
| `--molmoact2-offload-tokenization` | off | run MolmoAct2's tokenizer+image-processor as a Ray Data stage instead of inline in the training loop |
| `--molmoact2-offload-concurrency` | auto (from live CPU count) | Ray Data actor-pool size for the above |

```bash
# LoRA on one GPU (the starting point -- get this working before fft)
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --batch-size 8 --max-train-steps 50

# VLM frozen, only the action expert trains
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode freeze --molmoact2-action-mode continuous \
    --batch-size 8

# Full fine-tune with FSDP2 across multiple GPUs on one node, plus the Ray
# Data preprocessing offload -- both carry real unverified risk (checkpoint
# resume under FSDP2, and Ray-Data-Arrow round-tripping tokenizer output)
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-offload-tokenization --batch-size 32

# fft that doesn't fit even under FSDP2 -- trade speed for VRAM headroom
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-fsdp-cpu-offload --batch-size 32

# Custom per-group learning rates
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-vit-lr 1e-6 --molmoact2-connector-lr 5e-5 --molmoact2-action-expert-lr 1e-4
```

## π0.5

`--policy-type pi05` -- a PaliGemma-based VLM (~2.3B params total).
`--pi05-pretrained-path` is **required** -- "finetuning" implies starting
from real pretrained weights, not random init, and this project never
identified/verified a specific real π0.5 checkpoint repo id for you (the
standing rule here is to never guess a repo id) -- find a real one on the
HF Hub yourself first.

| Flag | Default | Meaning |
|---|---|---|
| `--pi05-pretrained-path` | -- (required) | HF repo id or local path to a real pretrained π0.5 checkpoint |
| `--pi05-freeze-vision-encoder` | off | freeze the vision tower; language model + action expert still train. No LoRA/peft exists in this integration (`PI05Config.use_peft` is a dead field) -- this and the next flag are the only ways to reduce the trainable surface |
| `--pi05-train-expert-only` | off | only the action expert trains, everything else frozen -- combinable with `--pi05-freeze-vision-encoder` |
| `--pi05-no-gradient-checkpointing` | off (checkpointing on) | disable only with confirmed memory headroom -- auto-wires from `PI05Config`, no extra wiring needed unlike MolmoAct2 |
| `--pi05-empty-cameras` | `0` | pad `input_features` with dummy camera slots -- for when the pretrained checkpoint expects more cameras than this dataset has |

DDP only -- no FSDP2 path exists for π0.5 in this pipeline (no real VRAM
number ever justified building one; extending `model/pi05.py` to FSDP2
later is a mechanical repeat of `model/molmoact2.py`'s pattern if a real
run shows DDP doesn't fit).

```bash
# Full fine-tune on one GPU
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --batch-size 8 --max-train-steps 50

# Freeze the vision encoder, only train the action expert -- reduces the
# trainable/optimizer-state footprint without needing FSDP2
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-freeze-vision-encoder --pi05-train-expert-only --batch-size 8

# Checkpoint expects more camera slots than this dataset provides
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-empty-cameras 2
```

## Overriding the dataset source / v3 root

```bash
# A hand-built v3 root with no conversion_params.json needs an explicit
# --dataset-source so camera_keys/state_dim/action_dim/tick_fps resolve
python -m training.train --tasks my_tasks --v3-root /data/hand_built_v3 --dataset-source abc130k
```

## Checkpoint retention

```bash
# Keep only checkpoints where loss improved (fewer writes, but a resume can
# lose progress back to the last improvement -- see the flag table above)
python -m training.train --tasks dress_the_teddy_bear --save-only-on-improvement

# Keep more/fewer checkpoints on disk
python -m training.train --tasks dress_the_teddy_bear --checkpoint-max-to-keep 10
```

## After training

```bash
python -m training.history --storage-root <storage-root>
tensorboard --logdir <storage-root>/<run-name>/tensorboard
```

## See also

- Root [`README.md`](../README.md) -- install, quick start, repo layout
- [`training/data_prep/README.md`](data_prep/README.md) -- preparing a dataset before training
- [`docs/extending.md`](../docs/extending.md) -- adding a new policy, dataset, or robot
