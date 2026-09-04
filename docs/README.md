# PolicyForge docs

Ray Data + Ray Train pipeline, split into two independent halves: **data
prep** (`training/data_prep/` + `training/prepare_data.py`) discovers/
downloads/converts a raw dataset into LeRobot v3; **training**
(`training/train.py` + `training/train_loop.py` + `training/model/`) reads
an already-prepared v3 root and trains a policy. `train.py` never
discovers/downloads/converts anything itself.

## Start here -- one file per step, in order

1. [Setup](01-setup.md) -- install, `HF_TOKEN`, repo layout
2. [Verify decode](02-verify-data.md) -- sanity-check one episode before converting anything
3. [Prepare data](03-prepare-data.md) -- discover, download, convert to LeRobot v3
4. [Train](04-train.md) -- run `train.py`, full CLI reference
5. [View results](05-view-results.md) -- history, TensorBoard, what a run produces

## Deep dives -- for the parts you're expected to customize

- [Customizing datasets](customizing-datasets.md) -- the three built-in
  datasets (abc130k, droid, agibot_alpha), the schema YAML format, and how
  to add a new one
- [Customizing policies](customizing-policies.md) -- ACT, MolmoAct2, π0.5:
  every policy-specific flag, LoRA vs. full fine-tune, FSDP2, normalization
- [Extending: new dataset / robot / policy](extending.md) -- condensed,
  procedural checklist for all three (the deep dives above have the *why*)

## Also worth reading before a real run

- [Verification status](verification-status.md) -- what's been run against
  real data and what's still open, across the whole pipeline

## Repo layout

```
training/
  common/        RobotSchema, DataConfig, ray_setup.py -- shared by data
                 prep and training
  config.py      TrainConfig, RunConfig, per-policy overrides (train-time only)
  data_prep/     discover -> decode -> align -> convert -> write. The ONLY
                 place that talks to a raw dataset source.
    schemas/*.yaml       one file per dataset
    strategies/*.py      one file per raw ingestion format
  data/          train-TIME dataset consumption (reading an already-
                 prepared v3 root) -- ray_dataset.py, stats.py
  model/         one file per policy (act.py, molmoact2.py, pi05.py) + registry.py
  vendor/        LeRobotDatasource / NumpyToTorchCollate, reused as-is
  train_loop.py, train.py, prepare_data.py, history.py, requirements.txt
```

Full detail on each piece lives in the step/deep-dive docs above, not here.
