# OpenPolicyKernel docs

Ray Data + Ray Train pipeline, split into two independent halves: **data
prep** (`training/data_prep/` + `training/prepare_data.py`) discovers/
downloads/converts a raw dataset into LeRobot v3; **training**
(`training/train.py` + `training/train_loop.py` + `training/model/`) reads
an already-prepared v3 root and trains a policy. `train.py` never
discovers/downloads/converts anything itself.

See the root [`README.md`](../README.md) for install/quick-start commands.

## Extending the pipeline

- [Extending: new dataset / robot / policy](extending.md) -- condensed,
  procedural checklist for adding any of the three

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

Full detail on each piece lives in the code itself -- every module here has
a docstring explaining its role and how it fits together.
