# Training -- `train.py`

Reads an already-prepared LeRobot v3 dataset and trains a policy on it.
Assumes `training/prepare_data.py` has already been run for `--tasks` --
this script does no discover/download/convert of its own; if no prepared
dataset is found at `--v3-root`, it exits with the `prepare_data.py`
command to run first. See [`training/data_prep/README.md`](data_prep/README.md)
for that step.

```bash
python -m training.prepare_data --tasks arrange_the_flowers --dataset-source abc130k --max-episodes-per-task 300
python -m training.train --tasks arrange_the_flowers --dataset-source abc130k --policy-type act --max-train-steps 50
```

## General flags

Neither `--dataset-source` nor `--policy-type` has a default -- both are
either required outright (`--policy-type`) or required unless `--v3-root`
is given explicitly (`--dataset-source`, see the table row below).

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | must match what `prepare_data.py` was run with |
| `--config-file` | none | optional YAML file providing a base `RunConfig` -- every flag in this doc still works and takes precedence; see "Config files" below |
| `--run-name` | `<policy_type>-<tasks>-<timestamp>` | also the resume key -- see below |
| `--storage-root` | `TrainConfig` default | where run output (checkpoints, TensorBoard, history) is written |
| `--num-epochs` | `1` | |
| `--batch-size` | `8` | |
| `--lr` | `1e-5` | base learning rate -- the fallback every MolmoAct2 per-group LR flag (`--molmoact2-vit-lr` etc.) uses when left unset |
| `--lr-backbone` | `1e-5` | ACT-specific backbone LR group -- unused by MolmoAct2/pi05, which have their own LR-group flags |
| `--grad-accum` | `1` | gradient accumulation steps |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--max-train-steps` | none (full epoch) | cap total steps -- for a smoke run |
| `--eval-every-steps` | `200` | report/checkpoint/early-stop-check granularity |
| `--early-stop-patience` | `5` | stop after this many eval windows with no loss improvement; `0` or negative disables it |
| `--save-only-on-improvement` | off | only checkpoint when loss improves (less write I/O, but a resume can lose progress back to the last improvement) |
| `--checkpoint-max-to-keep` | `3` | keeps the N best-scoring checkpoints plus the single most recent one (native Ray Train `CheckpointConfig` behavior) |
| `--num-workers` | live GPU count | Ray Train DDP worker count |
| `--v3-root` | derived from `--dataset-source` + `--tasks` (prepare_data.py's own default path) | must already exist. If omitted, `--dataset-source` is required so the default path can be derived at all |
| `--dataset-source` (required unless `--v3-root` given) | read from the prepared dataset's own `conversion_params.json` if `--v3-root` is given and omits it | which schema to resolve camera_keys/state_dim/action_dim/tick_fps from |
| `--policy-type` (required) | -- | `act`, `molmoact2`, or `pi05` |

```bash
# Custom base LR, ACT backbone LR, gradient accumulation, and weight decay
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --lr 5e-5 --lr-backbone 1e-5 --grad-accum 4 --weight-decay 1e-3
```

## Config files

`--config-file PATH` loads a YAML file into a base `RunConfig`, parsed with
[`draccus`](https://github.com/dlwh/draccus) -- the same library lerobot's
own `ACTConfig`/`PI05Config`/`MolmoAct2Config` are built on (already an
installed dependency, pulled in transitively by `lerobot==0.6.1`). It's
purely additive: **every named flag in this doc still works exactly as it
always has, and takes precedence over anything the YAML sets** --
`--config-file` only fills in values nothing else was explicitly passed
for. `--tasks` and `--policy-type` are still required on the command line
even when the file also sets them (there's no way to make a required flag
optional only when a config file exists without changing the CLI's shape
for everyone, which this was deliberately kept from doing).

Precedence, low to high: dataclass defaults (`training/config.py`) <
`--config-file`'s YAML < a named CLI flag actually typed on the command
line. One accepted limitation: a CLI flag whose value happens to equal its
own default is indistinguishable from not having passed it at all, so the
config-file's value (if any) wins in that case -- if you need to force a
value back to its default while using a config file, remove it from the
file instead of relying on the flag.

The YAML mirrors `RunConfig`'s real shape (`training/config.py`,
`training/common/config.py`'s `DataConfig`) -- top-level `tasks`/
`policy_type`/`run_name`/`storage_root`, nested `train:`/`data:` sections,
and a `model:` section whose `type: act|molmoact2|pi05` key selects which
of the three `*ConfigOverrides` dataclasses the rest of that section's
fields apply to (draccus's "choice registry" mechanism -- if `model.type`
disagrees with the resolved `--policy-type`, `train.py` exits with a clear
error rather than guessing which one you meant).

Five real, draccus-verified examples in
[`training/configs/`](configs/README.md) -- one per policy, plus one
showing the `data:` section (action-space selection, delta actions,
per-camera image normalization including a depth camera):

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml

# A named flag still overrides whatever the file sets
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k \
    --policy-type act --config-file training/configs/act_example.yaml --batch-size 32
```

## Action space & per-camera image normalization

General flags -- apply regardless of `--policy-type`, applied once to the
Ray Dataset before normalization stats are computed (not a per-policy
concern).

| Flag | Default | Meaning |
|---|---|---|
| `--action-space` | none (every action component) | which named subset of the dataset's `action_components` to train on -- e.g. `joint` or `end_effector` for `agibot_alpha`, which records both. Valid names are dataset-specific (the schema's `action_space_components` keys); most datasets declare none, so this only applies to `agibot_alpha` today |
| `--action-representation` | `absolute` | `absolute` (unchanged) or `delta`: `action -= observation.state` per dim, using each action component's **same-named** state component as the reference point -- components with no same-named state component (e.g. `agibot_alpha`'s `robot/velocity`, which has no state counterpart) stay absolute, there's no reference point to use |
| `--action-delta-exclude` | none | action component names to keep absolute even under `--action-representation delta` (e.g. a gripper component) |
| `--image-normalization` | none | per-camera mode, `CAMERA=MODE` pairs, e.g. `--image-normalization top=unit01 depth_head=depth`. Modes: `mean_std` (default, dataset-computed mean/std, current behavior), `unit01` (`x/255`), `unit_pm1` (`x/127.5 - 1`), `depth` (`clip(x,0,max)/max`), `log` (`log1p(x)/log1p(max)`) |
| `--image-normalization-default` | `mean_std` | mode for any camera not covered by `--image-normalization` |
| `--image-normalization-max` | none | `CAMERA=VALUE` pairs -- **required** for any camera using `depth`/`log` (the raw-value ceiling to clip/scale by, e.g. max depth in millimeters); no guessed default |

A single-channel camera (a depth camera, in practice) is automatically
replicated to 3 channels after normalization, so it flows through the same
vision backbone every RGB camera does -- see
`training/model/image_normalization.py`.

```bash
# Train agibot_alpha on joint-space actions instead of the full 36-dim
# action vector (which mixes joint- and end-effector-space components)
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --action-space joint

# Delta (relative-to-state) actions, keeping one component absolute --
# needs a dataset whose action/state components are actually same-named
# (abc130k's aren't: e.g. "/left-arm-action" vs "/left-arm-state", so
# --action-representation delta would leave every dim absolute there --
# see the --action-representation row above). agibot_alpha's are:
# "effector/position" appears in both state_components and
# action_components, so it gets a real delta unless excluded like this.
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --action-representation delta --action-delta-exclude effector/position

# Per-camera normalization: one RGB camera to [0,1], another to [-1,1],
# a depth camera clipped/scaled by its real max range (millimeters)
python -m training.train --tasks fridge --dataset-source agibot_alpha --policy-type act \
    --image-normalization top=unit01 wrist=unit_pm1 depth_head=depth \
    --image-normalization-max depth_head=5000
```

## Performance instrumentation

General flags -- apply regardless of `--policy-type`. Off by default:
accurate step timing needs `torch.cuda.synchronize()` calls, which
serialize async CUDA work and cost real throughput whenever they're on, so
this is opt-in rather than always-on.

| Flag | Default | Meaning |
|---|---|---|
| `--log-perf-metrics` | off | logs GPU utilization/VRAM, per-step timing (`perf/data_wait_s`/`preprocess_s`/`compute_s`/`optimizer_step_s`), effective batch size, throughput (`perf/samples_per_sec`), and an IO-bound-vs-compute-bound ratio (`perf/io_bound_fraction`) to TensorBoard under `perf/*`. Needs `nvidia-ml-py` installed for GPU compute-utilization % (`perf/gpu_util_pct`, `perf/gpu_mem_util_pct`) -- VRAM stats (`perf/vram_*`) work without it; missing `nvidia-ml-py` just skips those two, logged once as a warning, not a crash |
| `--profile-steps START:END` | none | requires `--log-perf-metrics`. Captures a real `torch.profiler` trace for steps `START..END` (inclusive) into the run's `tensorboard/` dir -- viewable in TensorBoard's PyTorch Profiler tab, same `tensorboard --logdir` command as everything else. Rank 0 only. Keep the range small (10-20 steps is usually plenty) -- traces get large fast; a range over 50 steps prints a warning |

`perf/io_bound_fraction` is `data_wait_s / (data_wait_s + preprocess_s +
compute_s)` per window -- loosely, high (>0.3-0.5) + low `gpu_util_pct`
means IO-bound (more Ray Data actors/CPU, bigger prefetch buffer, faster
decode is the fix); low + high `gpu_util_pct` means compute-bound (bigger
batch, mixed precision, model-side work is the fix). `perf/checkpoint_save_s`
is logged whenever a checkpoint actually writes -- without it, a periodic
step-time spike every `--eval-every-steps` looks like unexplained noise
instead of "checkpointing is slow."

```bash
# Find out whether a run is IO- or compute-bound
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --log-perf-metrics --max-train-steps 200

# Same, plus a kernel-level trace of steps 50-65 for a deeper look
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --log-perf-metrics --profile-steps 50:65 --max-train-steps 200
```

## Resuming a run

There's no `--resume` flag -- re-running with the **same `--run-name`**
(and `--storage-root`) resumes automatically: Ray Train detects the
existing run directory and its checkpoints and picks up from the most
recent one. Pass `--run-name` explicitly if you want to resume later
(the default name embeds a timestamp, so a fresh invocation without it
always starts a new run).

```bash
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --run-name teddy_bear_act_v1 --max-train-steps 500
# ...interrupted or crashed...
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act \
    --run-name teddy_bear_act_v1 --max-train-steps 500
```

## ACT

```bash
# Smoke run
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --max-train-steps 20

# Full run over the data, larger batch
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --batch-size 16 --num-epochs 5
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
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --batch-size 8 --max-train-steps 50

# VLM frozen, only the action expert trains
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode freeze --molmoact2-action-mode continuous \
    --batch-size 8

# Full fine-tune with FSDP2 across multiple GPUs on one node, plus the Ray
# Data preprocessing offload -- both carry real unverified risk (checkpoint
# resume under FSDP2, and Ray-Data-Arrow round-tripping tokenizer output)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-offload-tokenization --batch-size 32

# fft that doesn't fit even under FSDP2 -- trade speed for VRAM headroom
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-fsdp-cpu-offload --batch-size 32

# Custom per-group learning rates
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type molmoact2 \
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
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --batch-size 8 --max-train-steps 50

# Freeze the vision encoder, only train the action expert -- reduces the
# trainable/optimizer-state footprint without needing FSDP2
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-freeze-vision-encoder --pi05-train-expert-only --batch-size 8

# Checkpoint expects more camera slots than this dataset provides
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-empty-cameras 2
```

## Overriding the dataset source / v3 root

```bash
# A hand-built v3 root with no conversion_params.json needs an explicit
# --dataset-source so camera_keys/state_dim/action_dim/tick_fps resolve
python -m training.train --tasks my_tasks --v3-root /data/hand_built_v3 --dataset-source abc130k --policy-type act
```

## Checkpoint retention

```bash
# Keep only checkpoints where loss improved (fewer writes, but a resume can
# lose progress back to the last improvement -- see the flag table above)
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --save-only-on-improvement

# Keep more/fewer checkpoints on disk
python -m training.train --tasks dress_the_teddy_bear --dataset-source abc130k --policy-type act --checkpoint-max-to-keep 10
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
