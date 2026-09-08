# 4. Train

`train.py` assumes `prepare_data.py` (step 3) already ran for these
`--tasks` -- it does no discover/download/convert of its own, and exits
with the exact `prepare_data.py` command to run if nothing's prepared yet.

```bash
python -m training.train --tasks arrange_the_flowers box_folding \
    --dataset-source abc130k --policy-type act \
    --max-train-steps 50   # smoke run first
```

`--policy-type` is always required (no default). Locating the dataset also
always needs `--dataset-source` or `--v3-root` (either one, to derive/give
the path) -- `train.py` exits immediately with a clear error naming
whichever's missing.

Drop `--max-train-steps` for a full run once the smoke run's loss is moving
and a checkpoint lands under `training/runs/<run-name>/` (see
[5. View results](05-view-results.md)).

Once the dataset is located, `train.py` resolves its `RobotSchema` (camera
keys, state/action dims, tick rate) automatically from the prepared
dataset's own `conversion_params.json` -- so if `--v3-root` already points
at a real prepared dataset, `--dataset-source` isn't needed a second time
just for this. It's only needed as an override for a hand-built `--v3-root`
with no `conversion_params.json`.

## Choosing a policy

`--policy-type {act, molmoact2, pi05}` selects which policy trains --
`act` (default, lightweight, trains from random init) is the place to
start. MolmoAct2 and π0.5 are much bigger VLAs with their own required
flags and real tradeoffs (LoRA vs. full fine-tune, FSDP2, etc.) -- see
[Customizing policies](customizing-policies.md) for the full picture before
using either.

## Command reference

Reads whatever's already at `--v3-root` and errors out (with the
`prepare_data.py` command to run) if nothing's there -- see "Using data you
already have" in [3. Prepare data](03-prepare-data.md#using-data-you-already-have).
No data-prep flags here; those all live on `prepare_data.py`.

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | one or more task-name substrings -- must match what `prepare_data.py` was run with, since it also determines the default `--v3-root` |
| `--dataset-source` | none (read from `conversion_params.json`) | override which schema's `RobotSchema` to use -- only needed for a hand-built `--v3-root` with no `conversion_params.json`; the common case needs no flag at all |
| `--run-name` | `<policy>-<tasks>-<timestamp>` | run identifier; also the TensorBoard/checkpoint/history subdirectory name |
| `--storage-root` | `training/runs` | where checkpoints/TensorBoard/history for this run land |
| `--num-epochs` | `1` | full passes over the dataset |
| `--batch-size` | `8` | |
| `--max-train-steps` | none (full run) | cap total steps -- use for a smoke run |
| `--eval-every-steps` | `200` | report/checkpoint/early-stop-check granularity, in steps |
| `--early-stop-patience` | `5` | stop after this many eval windows with no loss improvement; `0` or negative disables it |
| `--save-only-on-improvement` | off (checkpoint every report) | only write a checkpoint when the loss improves -- less write I/O, but a resume after an interruption can lose progress back to the last improvement |
| `--checkpoint-max-to-keep` | `3` | max checkpoints kept on disk -- with the default (every report checkpointed), this is the N best-scoring PLUS the single most recent checkpoint (for resuming), even when it isn't among the N best -- native Ray Train behavior, see [5. View results](05-view-results.md) |
| `--num-workers` | live GPU count | Ray Train DDP workers |
| `--v3-root` | `training/lerobot_v3/<dataset-source>/<sorted tasks>` (needs `--dataset-source` to derive this) | where the already-converted dataset is read from -- must already exist |
| `--policy-type` (required) | -- | `act`, `molmoact2`, or `pi05` -- see [Customizing policies](customizing-policies.md); no separate environment needed for any of them (see [1. Setup](01-setup.md)) |

Policy-specific flags (`--molmoact2-*`, `--pi05-*`) are documented in full in
[Customizing policies](customizing-policies.md).

```bash
# Already have a LeRobot v3 dataset (from anywhere -- hand-built, downloaded
# pre-converted, prepare_data.py from a prior run) and just want to train on it
# -- --v3-root alone locates it, --policy-type is still always required
python -m training.train --tasks arrange_the_flowers --v3-root /path/to/existing/dataset --policy-type act

# Keep more checkpoints, and only write one when the loss improves (less write
# I/O than the default, at the cost of a resume potentially losing progress
# back to the last improvement)
python -m training.train --tasks arrange_the_flowers --dataset-source abc130k --policy-type act \
    --checkpoint-max-to-keep 10 --save-only-on-improvement

# Full run, tuned: longer patience, non-default storage location
python -m training.train --tasks arrange_the_flowers box_folding \
    --dataset-source abc130k --policy-type act \
    --num-epochs 3 --batch-size 16 --eval-every-steps 100 --early-stop-patience 8 \
    --storage-root /mnt/shared_storage/act_training --run-name flowers-boxes-v2
```

On a shared multi-node cluster, override `--storage-root` to shared storage
(e.g. a mounted network filesystem) so a worker restarting on a different
node can still see checkpoints.

## Next

[5. View results](05-view-results.md)
