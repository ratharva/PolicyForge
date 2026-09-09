# OpenPolicyKernel

A Ray Data + Ray Train pipeline for finetuning robot-learning policies on
robot teleop data. Multi-policy and multi-dataset by design -- ACT,
MolmoAct2, and π0.5 today, on XDOF/ABC-130k, AgiBot World Alpha, or DROID --
built so adding another policy, dataset, or robot is a small, additive
change, not a rework of existing code.

## Install

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat hf_tok.txt)   # after accepting a gated dataset's terms on huggingface.co
```

## Quick start

Two independent steps: `prepare_data.py` discovers/downloads/converts a raw
dataset into LeRobot v3; `train.py` reads an already-prepared v3 root and
trains a policy on it -- it never discovers, downloads, or converts
anything itself. A few representative examples (full flag reference and
every scenario -- other datasets, other policies, resuming a run, FSDP2,
LoRA, etc. -- in the linked docs below):

`--dataset-source` and `--policy-type` have no default -- both must be
given explicitly on every run.

```bash
# Prepare abc130k, then train ACT on it
python -m training.prepare_data --tasks arrange_the_flowers --dataset-source abc130k --max-episodes-per-task 20
python -m training.train --tasks arrange_the_flowers --policy-type act --max-train-steps 50

# A gated dataset (AgiBot World Alpha) -- downloads whole tar shards to
# local disk first, real sizes vary a lot by task, check before running
python -m training.prepare_data --tasks fridge --dataset-source agibot_alpha --max-episodes-per-task 2

# Finetune a VLA policy (π0.5) instead of ACT -- needs a real pretrained
# checkpoint you've found on the HF Hub yourself, see training/README.md
python -m training.train --tasks arrange_the_flowers --dataset-source abc130k --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> --max-train-steps 50
```

- [`training/data_prep/README.md`](training/data_prep/README.md) -- every
  `prepare_data.py`/`discover.py`/`verify_decode.py` flag and scenario, per
  dataset source
- [`training/README.md`](training/README.md) -- every `train.py` flag and
  scenario, per policy (ACT, MolmoAct2, π0.5)

## Layout

```
training/
  common/        RobotSchema, DataConfig, ray_setup.py -- shared by data prep and training
  config.py      TrainConfig, RunConfig, per-policy config overrides
  data_prep/     discover -> decode -> align -> convert -> write
    schemas/*.yaml     one file per dataset (camera keys, state/action layout, tick rate)
    strategies/*.py    one file per raw ingestion format (mcap, hf_lerobot_mirror, agibot_hdf5)
  data/          train-time dataset consumption (an already-prepared v3 root) -- ray_dataset.py, stats.py
  model/         one file per policy (act.py, molmoact2.py, pi05.py) + registry.py
  vendor/        proven, load-bearing third-party-adjacent infra, reused as-is
  train_loop.py, train.py, prepare_data.py, history.py, requirements.txt
```

## More detail

- [`docs/extending.md`](docs/extending.md) -- condensed, procedural
  checklist for adding a new dataset, robot, or policy
