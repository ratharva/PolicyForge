# 1. Setup

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat hf_tok.txt)   # after accepting a gated dataset's terms on huggingface.co
```

Run everything from the repo root (parent of `training/`), as
`python -m training.<module>` -- the package imports (`from training.config
import ...`, `from training.vendor.lerobot_datasource import ...`) need it
on the path.

`HF_TOKEN` is only needed by `training/prepare_data.py` (and
`training/data_prep/verify_decode.py`) -- `train.py` never talks to a raw
dataset source, only an already-converted local/cloud LeRobot v3 root, so it
never needs a token.

## Environment variables

Every real env var this pipeline (or a library it calls) reads, in one
place -- not every run needs all of these.

**Required, situationally:**

| Variable | When | Notes |
|---|---|---|
| `HF_TOKEN` | `prepare_data.py`/`verify_decode.py` against a gated `hf://` source | see above -- `train.py` never needs it for the dataset itself |
| `HF_TOKEN` | `--policy-type pi05` at train time, regardless of dataset | π0.5's tokenizer loads from `google/paligemma-3b-pt-224`, a *separate* gated repo from whatever `--pi05-pretrained-path` checkpoint you use -- accept its license on huggingface.co too, or the run fails with a real `GatedRepoError` deep inside the first training step |
| `WANDB_API_KEY` (or run `wandb login` once) | `--wandb` with the default `--wandb-mode online` | not read by this project's own code -- the `wandb` library itself requires it. Skip with `--wandb-mode offline` for a no-login smoke test |

**Optional:**

| Variable | Default if unset | Notes |
|---|---|---|
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `~/.cache/huggingface` | redirect the HF download cache to a bigger volume before a large `prepare_data.py` run |
| `WANDB_MODE` | wandb's own default (`online`) | same effect as `--wandb-mode`; the CLI flag wins if both are set |
| `WANDB_PROJECT` / `WANDB_ENTITY` | none | fallback for `--wandb-project`/`--wandb-entity` when those flags aren't passed -- wandb's own env-var convention, not this project's |
| `LEROBOT_S3_ANON` | unset (real credentials expected) | set to `1`/`true`/`yes` for anonymous, no-credential access to a public `s3://` dataset source (read in `training/vendor/lerobot_datasource.py`) |
| `OPENPOLICYKERNEL_DASHBOARD_HOST` | `127.0.0.1` (loopback-only) | exposes Ray's dashboard beyond localhost -- it has no built-in auth, so this prints a warning; SSH-tunnel instead (`ssh -L 8265:localhost:8265 <host>`) unless you specifically need this |
| `RAY_DEDUP_LOGS` | `1` (Ray's own default -- dedupes identical log lines across workers) | set to `0` if you want every worker's identical log line printed separately instead of collapsed -- a real Ray feature, not something this project adds |

## One environment covers every policy and every dataset

**No separate environment needed for MolmoAct2, π0.5, or any of the three
datasets.** `pip install -r training/requirements.txt` (the same install
ACT uses) is enough for all of it -- `MolmoAct2Config`/`MolmoAct2Policy` are
bundled in stock PyPI `lerobot>=0.6.1`, confirmed directly against the real
public PyPI wheel (`lerobot/policies/molmoact2/` is really in there) and by
actually constructing a real `MolmoAct2Config` in this environment, not just
an import check.

Two optional extras, commented out in `requirements.txt`:
- `accelerate` -- only for `--molmoact2-distributed-strategy fsdp2` (full
  fine-tuning at scale). `--molmoact2-train-mode lora` (the default) needs
  nothing beyond the base install.
- `s3fs`/`gcsfs` -- only if `--source-uri`/`--v3-root` uses `s3://` or
  `gs://` instead of `hf://` or a local path.

**Correction, kept here because it was a real mistake worth learning from**:
earlier versions of this doc said MolmoAct2 needed a separate fork
(`allenai/lerobot@molmoact2-policy`) in its own environment, with a whole
`requirements-molmoact2.txt` and a two-environment split. That was based on
research against the fork's own branch, which genuinely did have MolmoAct2
before mainline did -- but by the time this pipeline pinned `lerobot==0.6.1`
in `requirements.txt`, mainline had already absorbed the same policy, and
that was never cross-checked against what was actually installed in the dev
environment already being used for ACT. Running
`python -c "from lerobot.policies.molmoact2... import MolmoAct2Config"`
against the environment already sitting there would have caught this
immediately -- the general lesson: verify against what's *actually
pinned/installed*, not just against wherever the research happened to look.

## Repo layout

```
training/
  common/        RobotSchema, DataConfig, ray_setup.py -- shared by data
                 prep and training
  config.py      TrainConfig, RunConfig, per-policy overrides (train-time only)
  data_prep/     discover -> decode -> align -> convert -> write. The ONLY
                 place that talks to a raw dataset source.
    schemas/*.yaml       one file per dataset -- see customizing-datasets.md
    strategies/*.py      one file per raw ingestion format
  data/          train-TIME dataset consumption (reading an already-
                 prepared v3 root) -- ray_dataset.py, stats.py
  model/         one file per policy (act.py, molmoact2.py, pi05.py) + registry.py
  vendor/        LeRobotDatasource / NumpyToTorchCollate, reused as-is
  train_loop.py, train.py, prepare_data.py, history.py, requirements.txt
```

## Next

[2. Verify decode](02-verify-data.md)
