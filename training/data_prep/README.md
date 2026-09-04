# Data prep -- `prepare_data.py`, `discover.py`, `verify_decode.py`

Discovers, downloads, and converts a raw dataset into LeRobot v3.
`training/train.py` never does any of this itself -- run this first.

```bash
export HF_TOKEN=$(cat hf_tok.txt)   # only needed for hf:// sources, and only if the repo is gated
python -m training.prepare_data --tasks <task-name-substrings> --dataset-source <abc130k|droid|agibot_alpha> --max-episodes-per-task 20
```

`--dataset-source` has no default -- it's required on every run. Which raw
format it uses (and which flags below actually apply) is decided by its
`ingestion_strategy`, declared in
`training/data_prep/schemas/<dataset_source>.yaml`:

| `--dataset-source` | ingestion strategy | `--tasks` filters? |
|---|---|---|
| `abc130k` | `mcap` | yes |
| `droid` | `hf_lerobot_mirror` | **no** -- see below |
| `agibot_alpha` | `agibot_hdf5` | yes |

## Full flag reference

| Flag | Default | Applies to | Meaning |
|---|---|---|---|
| `--tasks` (required) | -- | all | task-name substrings, matched against real discovered task names |
| `--dataset-source` (required) | -- | all | which schema/strategy to use |
| `--max-episodes-per-task` | `300` | all | cap per task (fewer if a task has less) |
| `--source-uri` | schema's `default_source_uri` | `mcap` only | override where the raw dataset lives -- `hf_lerobot_mirror` always uses its schema's `full_repo_id` instead |
| `--v3-root` | `training/lerobot_v3/<dataset_source>/<sorted task names>` | all | where to write the converted dataset -- local path, `s3://bucket/prefix`, or `gs://bucket/prefix` (S3/GCS unverified against real buckets in this project) |
| `--reconvert` | off | all | rebuild even if a matching dataset already exists at `--v3-root` |
| `--refresh-listing` | off | `mcap` only | re-scan the full source tree instead of the cached task/episode listing (`training/.cache/<dataset_source>_listing.json`) |
| `--mode` | `stream` | `mcap` only | `stream`: read over HTTP, nothing local but the output. `download`: pull each raw episode to disk first, then convert |
| `--sequential-camera-decode` | off (parallel) | `mcap` only | decode one camera at a time -- slower per episode, lower peak memory |
| `--max-concurrent` | auto (from live CPU/memory) | `mcap` only | override concurrent episode conversions -- too high risks OOM, set deliberately |

## Scenarios

### abc130k -- MCAP, XDOF/ABC-130k

```bash
# First run -- fresh conversion
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20

# Same tasks, more episodes -- incrementally reuses the 20 already converted,
# only fetches/decodes the 10 new ones (matched via the episode manifest,
# not re-fetched/re-decoded)
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 30

# Multiple tasks in one v3 root
python -m training.prepare_data --tasks dress_the_teddy_bear arrange_the_flowers --dataset-source abc130k --max-episodes-per-task 20

# Force a full rebuild even though a matching dataset already exists
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --reconvert

# Re-scan the source tree instead of the cached listing (new tasks/episodes
# appeared upstream since the last run)
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --refresh-listing

# Download raw episodes to disk first instead of streaming over HTTP
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --mode download

# Lower peak memory per episode (slower) -- useful on a smaller/shared machine
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --sequential-camera-decode

# Pin conversion concurrency explicitly instead of the auto-computed value
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --max-concurrent 4

# Custom output location
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --v3-root /data/abc130k_teddy_bear

# Point at a different raw source (mcap strategy only)
python -m training.prepare_data --tasks dress_the_teddy_bear --dataset-source abc130k --max-episodes-per-task 20 --source-uri hf://datasets/some/other-mcap-mirror
```

### droid -- DROID (Franka Panda), via lerobot's own v2.1->v3.0 migration

`--tasks`/`--max-episodes-per-task` are accepted (still required by the CLI)
but **not applied as a filter** -- this strategy downloads and converts its
schema's `full_repo_id` (`cadene/droid_1.0.1`, ~95,600 episodes) as a whole,
via lerobot's own official `convert_dataset_v21_to_v30.py`. Real
per-source task-filtering for a monolithic mirror like this is unresolved
future work, not implemented -- see `strategies/hf_lerobot_mirror.py`'s
`prepare()` docstring.

```bash
# --tasks is still required by the CLI even though it isn't used to filter
python -m training.prepare_data --tasks droid --dataset-source droid --v3-root /data/droid_v3
```

`cadene/droid_1.0.1` is large (18.4GB+) -- for a cheap smoke test of the
mechanism itself, a schema override pointing at the small
`lerobot/droid_100` sample (2GB, already `v3.0`, a real fast-path no-op
migration) is what this project's own DROID verification used.

### agibot_alpha -- AgiBot World Alpha (mobile dual-arm, gated)

Requires accepting AgiBot World Alpha's gated terms on huggingface.co first,
with `HF_TOKEN` belonging to that account.

Downloads whole tar shards to local disk, task by task, then scans them
locally (no per-episode network streaming) -- see
`strategies/agibot_hdf5.py`'s module docstring for the full reasoning. Two
real, load-bearing consequences:

- **Observation shards download per-task, in full** -- not just the
  shard(s) covering the episodes you asked for. Real sizes vary a lot by
  task (checked directly against the repo): as little as ~54GB, as much as
  ~285GB for a single task's observation shards.
- **`proprio_stats` downloads once, shared across every task** -- a single
  ~48GB tar covering the whole dataset, downloaded the first time any task
  needs it and reused (via `huggingface_hub`'s own local cache) for every
  task after, in this run or a later one.

Check real shard sizes for a task before running this for real:

```bash
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
entries = list(api.list_repo_tree('agibot-world/AgiBotWorld-Alpha', repo_type='dataset', path_in_repo='observations/<task_id>/', token=True))
print(sum(e.size for e in entries) / 1e9, 'GB across', len(entries), 'shard(s)')
"
```

```bash
python -m training.prepare_data --tasks fridge --max-episodes-per-task 2 \
    --dataset-source agibot_alpha --v3-root /data/agibot_v3
```

By default, `huggingface_hub` downloads into `~/.cache/huggingface/hub` --
point `HF_HOME` or `HUGGINGFACE_HUB_CACHE` at a larger volume first if your
default disk doesn't have room for a task's full shard set plus the shared
48GB proprio tar.

## Before converting anything -- `discover.py` and `verify_decode.py`

Both also require `--dataset-source` explicitly -- no default.

```bash
# List every real task name/id this dataset source has (mcap strategy only)
python -m training.data_prep.discover --dataset-source abc130k

# Sanity-check the decode pipeline on ONE real episode before trusting it for
# a full conversion -- downloads one episode, decodes + aligns it, prints
# state/action ranges and action-chunk padding, writes a few PNG frames per
# camera for visual inspection (mcap strategy only)
python -m training.data_prep.verify_decode --task dress_the_teddy_bear --dataset-source abc130k
```

`verify_decode.py`'s full flag set: `--task` (required, task-name
substring), `--dataset-source` (required), `--n-frames` (frames per
camera to dump), `--chunk-size`, `--out-dir`, `--image-size H W`.

## See also

- Root [`README.md`](../../README.md) -- install, quick start, repo layout
- [`training/README.md`](../README.md) -- running `train.py` once a dataset is prepared
- [`docs/extending.md`](../../docs/extending.md) -- adding a new dataset, robot, or ingestion strategy
