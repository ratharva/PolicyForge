# Customizing datasets

`--dataset-source {abc130k, agibot_alpha, droid}` on `prepare_data.py`/
`train.py` selects which YAML file
(`training/data_prep/schemas/<dataset_source>.yaml`) and which ingestion
strategy (`training/data_prep/strategies/<ingestion_strategy>.py`) runs.
Choices are **discovered** from whatever schema files exist -- not a
hardcoded list -- so adding a dataset that reuses an existing strategy is a
new YAML file, not new Python.

## The three built-in datasets

### abc130k (default) -- XDOF/ABC-130k, MCAP

The original, fully working dataset -- unchanged behavior from before this
was generalized to a schema file, verified with a real end-to-end regression
(real discover -> real stream -> real MCAP decode -> real convert -> real
train, all through the schema-driven path) after the refactor.

```bash
python -m training.prepare_data --tasks arrange_the_flowers --max-episodes-per-task 20
python -m training.train --tasks arrange_the_flowers --max-train-steps 50
```

### droid -- DROID (Franka Panda, 18-institution consortium)

Uses the `hf_lerobot_mirror` ingestion strategy: downloads an existing HF
LeRobot-format mirror and runs lerobot's own official
`convert_dataset_v21_to_v30.py` migration script -- no new raw decode code.

**A real correction worth knowing, found by downloading real `meta/info.json`
files instead of trusting a fetched dataset-card summary** (this project's
recurring lesson, caught again here): the schema originally pointed at
`IPEC-COMMUNITY/droid_lerobot` assuming it was a same-schema "full" version
of the small `lerobot/droid_100` sample. Downloading both directly showed
they're NOT interchangeable -- `lerobot/droid_100` is already
`codebase_version: v3.0` (state_dim 7), while `IPEC-COMMUNITY/droid_lerobot`
is `v2.0` (state_dim 8) -- and lerobot's own migration script hard-validates
for `v2.1` specifically, rejecting v2.0 outright
(`validate_local_dataset_version`, read directly from the installed
package). `cadene/droid_1.0.1` (95,600 episodes, confirmed real `v2.1`,
matches the official porting guide's own reference `DROID_FEATURES` schema)
is used instead -- confirmed by downloading its real `meta/info.json`, not
assumed. See `training/data_prep/schemas/droid.yaml`'s comment for the full
story.

**Verified**: `training/data_prep/strategies/hf_lerobot_mirror.py`'s
`convert_dataset(repo_id, root, push_to_hub=False)` call and its real
signature/version-validation behavior were confirmed by reading the
installed `lerobot` package's actual source
(`lerobot/scripts/convert_dataset_v21_to_v30.py`), not the porting-guide
docs alone -- it also already checks for an existing v3.0 hub tag before
attempting any conversion.

**A real dependency conflict was found and fixed by actually importing this
code path**, not just reading it: `lerobot.scripts.convert_dataset_v21_to_v30`
needs lerobot's own `"dataset"` extra (`jsonlines`, `datasets` -- neither
were in `requirements.txt`, confirmed missing by real
`ImportError`/`ModuleNotFoundError`), and that extra's own `av-dep` pin is
`av<16.0.0,>=15.0.0` -- but this project's `requirements.txt` had `av`
completely unpinned, which resolves to the latest (18.1.0). Installed
that way, `import lerobot.scripts.convert_dataset_v21_to_v30` fails with
`AttributeError: module 'av' has no attribute 'option'`
(`lerobot/datasets/pyav_utils.py` uses an `av.option.Option` type that
doesn't exist in later PyAV releases). Downgrading to `av>=15.0.0,<16.0.0`
fixes it -- verified this doesn't break this project's *own* av usage
either: re-ran both a real MCAP conversion (`video_decode.py`'s raw
elementary-stream decode) and a real training run reading the converted v3
dataset back (`vendor/lerobot_datasource.py`'s `av.open()`) against real
data with `av==15.1.0`, both passed. `requirements.txt` now pins
`av>=15.0.0,<16.0.0` and adds `jsonlines`/`datasets` explicitly, with
comments explaining why.

The dispatch/glue code was smoke-tested with `convert_dataset` monkeypatched
first (no download), then **a real full run against `lerobot/droid_100`
(2GB, has a real `v2.1` tag as well as its native `v3.0`) completed
successfully end to end**: real download, real
`convert_dataset_v21_to_v30.py` migration (100/100 data files, 100/100
videos per camera, real parquet output), and the result read back correctly
by our own `training/vendor/lerobot_datasource.py` (100 episodes, correct
video keys).

**A real bug was caught and fixed by this run**, not visible from
reading the code alone: `training/data/ray_dataset.py`'s
`build_lerobot_v3_dataset()` unconditionally renamed every video key to
`observation.images.<key>`, which is correct for our own writer's short
keys (e.g. `"top"`) but double-prefixed a dataset's columns that already
arrive fully-qualified -- exactly what `convert_dataset_v21_to_v30.py`
produces (`observation.images.exterior_image_1_left`, not
`exterior_image_1_left`). Real symptom before the fix:
`observation.images.observation.images.exterior_image_1_left` columns.
Fixed with a guard that skips already-prefixed keys.

**Still not run**: the full `cadene/droid_1.0.1` (18.4GB, the dataset the
schema's `full_repo_id` actually points at) -- `droid_100`'s dims (state 7)
differ from `cadene/droid_1.0.1`'s (state 8, per the schema), so this was a
strategy/mechanism smoke test, not a claim that `cadene/droid_1.0.1`
itself has been converted or trained on.

### agibot_alpha -- AgiBot World Alpha (mobile dual-arm, gated)

Uses the `agibot_hdf5` ingestion strategy. **Gated access was obtained and
the real data structure fully confirmed** -- no more placeholders in the
schema, and the core decode transform is real, implemented, and verified
against real downloaded data (not just written and assumed correct).

Real repo structure, confirmed via unauthenticated file listing:
`observations/<task_id>/<range>.tar` (per-task video, ~25-49GB each),
`proprio_stats/<range>.tar` (ONE ~48GB tar for the *entire* dataset),
`task_info/task_<id>.json` (a list of episode dicts with sub-action
segments/labels). None of these tars have an index -- a member can only be
reached by streaming sequentially from the start of its tar.

With real gated access, one full episode's data was extracted (streaming
just far enough into each tar to reach it -- about 1.2MB out of the 48GB
proprio tar, since the target happened to be an early entry, and ~29MB for
one camera's video -- not a full download) and inspected directly:
- **Real HDF5 structure confirmed exactly matches** the third-party
  BETA-sourced draft this schema started from (previously flagged as "not
  confirmed identical to Alpha's own" -- now confirmed, via real data, that
  it is): `state/<component>`/`action/<component>` datasets, all the
  populated ones sharing one length T (1136 or 1430 depending on episode)
  aligned to a shared `timestamp` dataset.
- **Real tick rate: 30.0 fps**, confirmed twice independently (the
  `timestamp` array's real median spacing, 33.339ms, AND the real video's
  own encoded `average_rate`, both 30).
- **Real camera keys confirmed** (8, all present as real
  `<key>_color.mp4` files): `head`, `head_center_fisheye`,
  `head_left_fisheye`, `head_right_fisheye`, `hand_left`, `hand_right`,
  `back_left_fisheye`, `back_right_fisheye`. Video files are genuine muxed
  mp4 containers, confirmed by real `av.open()` decode -- not raw
  elementary streams like MCAP.
- **No floor-align step needed** -- confirmed by comparing the SAME
  episode's real video frame count against its real proprio row count:
  both exactly 1136. State/action/video are already positionally aligned
  by construction.

`training/data_prep/schemas/agibot_alpha.yaml` now has real, non-placeholder
values throughout. `training/data_prep/strategies/agibot_hdf5.py`'s
transform logic (HDF5 component extraction + video decode, no alignment
needed) was tested against the real downloaded episode and confirmed
correct: real state (1136, 55) and action (1136, 36) arrays, real video
(1136, 224, 224, 3) uint8 frames -- then fed through the real
`write_episode`/`finalize_dataset` and read back correctly by
`training/vendor/lerobot_datasource.py`. This is the first time any
AgiBot data has gone through this pipeline end to end, from raw HDF5+mp4
to a trainable LeRobot v3 dataset.

**What's still open, a real scope boundary rather than a blocker**: fetching
one episode by name works (proven above), but `decode_and_align()`'s
`rel_path` currently expects the caller to already know which tar shard
covers a given episode -- there's no `discover.py`-style listing yet that
maps a task's requested episodes to their real tar shard URLs (each task's
observations tar filenames encode an episode-id range, e.g.
`653277-674257.tar`, which such a listing would need to parse). More
importantly, since neither tar has an index, converting many episodes from
the same tar one-by-one (the way `convert_to_lerobot_v3`'s per-episode Ray
tasks work for MCAP) would each re-pay the streaming-scan cost for
everything before their target -- a real, flagged inefficiency, not
something to discover the hard way at conversion scale. A real
implementation should scan each required tar once per run, extracting
every requested episode encountered along the way. Not built here.

## The schema YAML format

Every dataset is one file at `training/data_prep/schemas/<dataset_source>.yaml`.
Top-level fields are the same for every dataset regardless of ingestion
strategy:

```yaml
dataset_source: my_dataset          # must match the filename (my_dataset.yaml)
ingestion_strategy: mcap            # which strategies/*.py module interprets this file
default_source_uri: "hf://datasets/org/repo"

tick_fps: 30.0                      # common alignment/tick rate

camera_keys: [top, left_wrist, right_wrist]   # fixed-arity logical camera names

# (component name, dim) pairs, concatenated in order into one state/action
# vector -- state_dim/action_dim are derived by summing these, not set
# directly. What "component name" means is up to the ingestion strategy
# (an MCAP topic name, an HDF5 group path, a LeRobot column name).
state_components:
  - ["some_component_name", 6]
  - ["another_component_name", 1]
action_components:
  - ["some_component_name", 6]

# A block named after ingestion_strategy, read only by that strategy's
# build_ingestion_config(). Shape differs per strategy -- see below.
mcap:
  ...
```

`training/data_prep/schema_loader.py` parses this into two objects: a
generic `RobotSchema` (`training/common/robots.py` -- `camera_keys`,
`state_dim`/`action_dim` derived from the components lists, `tick_fps`) and
a strategy-specific ingestion-config dataclass built from the named block.
`RobotSchema` is never subclassed -- it's one concrete dataclass, always
populated from a schema file.

### The `mcap:` block (ingestion_strategy: mcap)

```yaml
mcap:
  state_value_field: position    # decoded protobuf message's value field
  camera_data_field: data        # decoded camera message's raw-bytes field
  camera_format_field: format    # decoded camera message's codec-name field
  top_camera_candidates: ["/top-camera", "/top-left-camera"]
  wrist_camera_topics:
    left_wrist: /left-wrist-camera
    right_wrist: /right-wrist-camera
  path_prefix: data               # "<path_prefix>/{split}/{task}/.../<episode_filename>"
  episode_filename: episode.mcap
```

Here, `state_components`/`action_components`' "component name" is the
literal MCAP topic string (e.g. `/left-arm-state`) -- `mcap.py`'s
`_extract()` reads each topic's decoded protobuf message and pulls
`state_value_field` off it via `getattr()`.

A different MCAP-based robot with the *same shape* of schema (protobuf
state/action with a position-like value field, video messages with a
data+format-like pair, a similar directory convention) is **just a new YAML
file** -- copy `abc130k.yaml`, change the topic names/field names/camera
keys/tick rate. A robot with a genuinely different message shape (not
protobuf, or e.g. ROS `sensor_msgs`) needs new extraction logic in
`strategies/mcap.py`'s `_extract()` itself -- the schema's fields cover
naming differences within the same message shape, not message-shape
differences.

### The `hf_lerobot_mirror:` block (ingestion_strategy: hf_lerobot_mirror)

```yaml
hf_lerobot_mirror:
  full_repo_id: cadene/droid_1.0.1
  source_format_version: v2.1     # informational -- the real script hard-validates this itself
  camera_key_map:                  # this mirror's real column name -> our logical camera_keys
    exterior_1_left: observation.images.exterior_1_left
    exterior_2_left: observation.images.exterior_2_left
    wrist_left: observation.images.wrist_left
```

Here, `state_components`/`action_components`' "component name" is the
mirror's real flat column name (e.g. `observation.state`) -- this strategy
doesn't decode anything itself, it downloads `full_repo_id` and runs
lerobot's own official `convert_dataset_v21_to_v30.py` migration to produce
a v3 root directly.

A different dataset that already has an HF LeRobot-format mirror (OXE,
RoboCasa, LIBERO, GR00T variants are increasingly common) is **just a new
YAML file** -- point `full_repo_id` at it, and fill in real
`state_components`/`action_components`/`camera_keys`/`tick_fps` from that
mirror's own real `meta/info.json`. **Download that file yourself and read
it before writing the schema** -- see the droid correction above for why a
fetched/cached summary isn't trustworthy enough on its own, and note that
`convert_dataset_v21_to_v30.py` specifically requires `codebase_version:
v2.1` (it hard-validates and rejects v2.0, and a dataset already at v3.0
needs no conversion at all -- the script checks for that itself).

### The `agibot_hdf5:` block (ingestion_strategy: agibot_hdf5)

```yaml
agibot_hdf5:
  observations_tar_prefix: observations
  proprio_stats_tar_prefix: proprio_stats
  task_info_dir: task_info
  state_hdf5_paths: {effector/position: effector/position, ...}   # component -> real "state/<path>" suffix
  action_hdf5_paths: {effector/position: effector/position, ...}  # component -> real "action/<path>" suffix
  video_filename_template: "{camera_key}_color.mp4"   # real, confirmed filename inside <episode_id>/videos/
```

The one ingestion strategy in this pipeline that needed genuinely new raw
decode work (not reusable off the shelf the way `mcap`/`hf_lerobot_mirror`
are for a same-shape dataset) -- but that work is now real and verified
against real downloaded data, not just scaffolding. See "agibot_alpha"
above for what's confirmed, and
`training/data_prep/strategies/agibot_hdf5.py`'s docstring for the real,
still-open tar-shard-discovery/scan-efficiency concern.

## Adding a genuinely new ingestion strategy

Only needed for a raw format that isn't MCAP, an HF LeRobot mirror, or
HDF5+tar. Add `training/data_prep/strategies/<name>.py` with:
- `build_ingestion_config(spec: dict) -> YourIngestionConfig` -- parses the
  schema's `<name>:` block
- Either `decode_and_align(source_uri, rel_path, robot, ingestion_cfg,
  token, convert_cfg, image_size) -> dict[str, np.ndarray]` (if this
  strategy uses the shared `convert_to_lerobot_v3` Ray-parallel
  orchestration -- see `mcap.py`), or your own `prepare(...)` entirely (if
  it doesn't, like `hf_lerobot_mirror.py`'s download+migrate)

Then register it in `training/data_prep/strategies/registry.py`'s
`_STRATEGY_MODULES` dict, and write a schema file naming it. The write side
(`training/data_prep/lerobot_v3_writer.py`) needs no changes -- it only
ever consumes the generic `dict[str, np.ndarray]` shape
(`{"state": (T, state_dim), "action": (T, action_dim),
"image.<camera_key>": (T, H, W, 3) uint8}`), regardless of what raw format
produced it.
