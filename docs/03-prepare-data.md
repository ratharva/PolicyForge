# 3. Prepare data

`training/prepare_data.py` is the **only** script that discovers, downloads,
or converts anything -- `train.py` (step 4) just reads whatever it produces.
`--dataset-source` picks which dataset: `abc130k` (default), `agibot_alpha`,
or `droid`. See [Customizing datasets](customizing-datasets.md) for the
full status of each and how to add a new one -- this page covers the
mechanics of running the step.

```bash
# See real task names + episode counts for the mcap strategy (abc130k)
python -m training.data_prep.discover --dataset-source abc130k

# Prepare data -- discover, download, convert to LeRobot v3, then stop.
# No GPU needed. Needs HF_TOKEN (only this script talks to the raw source).
python -m training.prepare_data --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300
```

Re-running for the same `--tasks`/`--max-episodes-per-task` is cheap
(`prepare_data.py`'s own `is_prepared()` check skips reconverting if nothing
changed). Pass `--reconvert` to force a rebuild -- e.g. after changing
`--max-episodes-per-task` or a schema's camera/tick settings.

## Data source: HF, S3, or GCS

`--source-uri` (mcap strategy only -- `hf_lerobot_mirror` always uses its
schema's own `full_repo_id`) identifies where the raw dataset lives:
- `hf://datasets/<repo_id>` (default: each schema's own `default_source_uri`,
  e.g. `hf://datasets/XDOF/ABC-130k`) -- Hugging Face Hub, needs `HF_TOKEN`
  if the repo is gated.
- `s3://<bucket>/<prefix>` or `gs://<bucket>/<prefix>` -- relies on ambient
  credentials (the standard boto3/gcloud credential chain), no token needed.

Streaming reads (`training/data_prep/strategies/mcap.py`'s
`read_raw_episode_streaming`) go through `fsspec.core.url_to_fs()` for all
three -- the same pattern `training/vendor/lerobot_datasource.py` already
uses for the *converted* LeRobot v3 side, extended here to the raw MCAP
input. Listing (`discover.py`) and download (`--mode download`,
`verify_decode.py`) deliberately keep `hf://` on `huggingface_hub`'s own
`HfApi`/`hf_hub_download` rather than routing it through the same generic
path -- that's what's proven fast and correct against the real 176k-file
XDOF repo. `s3://`/`gs://` use fsspec's generic `fs.find()`/`fs.get()`
instead.

**Only `hf://` has been run against real data.** `s3://`/`gs://` are
implemented but unverified -- no credentials were available in this
environment to test against a real bucket.

**The dataset schema is a YAML file, not hardcoded pipeline code.** See
[Customizing datasets](customizing-datasets.md) for the full format --
`training/data_prep/schemas/abc130k.yaml` for MCAP's own topic names,
message field names, tick rate, and directory layout. A different MCAP
dataset with the *same shape* of schema (protobuf state/action with a
position-like value field, video messages with a data+format-like pair, a
similar directory convention) is a new YAML file, not new Python. A raw
format that isn't protobuf-over-MCAP at all needs a new ingestion strategy
module (also covered in that doc).

## Streaming conversion (no local raw file)

`prepare_data.py` streams each episode directly from `--source-uri`'s
backend over the network (mcap strategy: `strategies/mcap.py`'s
`read_raw_episode_streaming`) -- nothing lands on local disk except the
converted LeRobot v3 output (parquet + mp4). Verified live against the real
gated ABC-130k repo, with per-step timing, not just "it worked": streaming +
protobuf-parsing a full episode took ~15s; video decode -- H.264 decode of 3
cameras -- was the real bottleneck at ~156s of a ~172s total (91%), not the
network. Cameras now decode concurrently (a thread pool inside each
episode's Ray task) instead of one after another -- measured ~83s end to
end after that change, near the theoretical ~3x on the decode portion since
PyAV releases the GIL during actual codec work.

One real gotcha found and worked around: `mcap`'s `make_reader()`
auto-detects `SeekingReader` vs. `NonSeekingReader` by calling `.seekable()`
on the stream, but `huggingface_hub`'s streaming file object reports
`seekable() == True` while `.seek()` actually raises -- so `NonSeekingReader`
is forced explicitly rather than trusting that auto-detection.

Trade-off versus a download-then-convert path: no local raw-file storage
cost (~170MB/episode for MCAP), but no HF download-cache reuse either --
reconverting the same episodes re-streams them over the network every time.
`discover.py`'s `download_episodes()` still exists (only `verify_decode.py`
uses it now) if you want a local copy for some other reason.

## Incremental conversion

Asking for more episodes than a previous conversion (e.g.
`--max-episodes-per-task 15` after an earlier `10`) reuses the
already-converted episodes and only streams/decodes the new ones
(`meta/episode_manifest.json` tracks which source episode maps to which
local `episode_index`) -- it used to either silently serve the stale
smaller dataset, or (once that was fixed) blindly reconvert everything
including what was already done. Dropping episodes/tasks instead (shrinking
the request) falls back to a full wipe + rebuild: LeRobot v3 requires
contiguous 0-based episode indices, so removing one in place would mean
renumbering and renaming every later episode's files -- not implemented.
The wipe also fixes what used to happen there: a plain reconvert only
overwrote indices `0..N-1` and left any higher-numbered episodes' files
orphaned on disk when the count shrank.

`task_index` is handled carefully across incremental runs: it's baked
permanently into each episode's data file at write time and never touched
again, so it has to stay stable (existing tasks keep their index; only a
genuinely new task gets a new one, appended, not resorted) -- recomputing it
fresh from a sorted task list every run would silently relabel old episodes
if the task set ever changed.

This surfaced a real bug during verification, now fixed: `finalize_dataset`'s
global-row-index patch used to read whatever was in a data file's `index`
column and add the new offset to it -- correct for a freshly-written episode
(whose `index` is still a fresh local 0-based value), wrong for a *reused*
one, whose `index` already holds a previous run's global offset from last
time. Reusing an episode across generations double-applied the offset,
corrupting its row range and silently truncating it on read. Fixed by
deriving the local position from `frame_index` (written once, never touched
again) instead of `index` (mutated every run) -- verified with a real
2-episode-then-3-episode run against the live dataset, confirming every row
of every episode (including the reused ones) reads back correctly.

## Configurable conversion: stream vs. download, parallel vs. sequential decode

`ConvertConfig` (`training/data_prep/config.py`) exposes both, as CLI flags
(mcap strategy only -- the only strategy that does its own Ray-parallel
per-episode conversion):

- `--mode stream` (default) / `--mode download`: stream reads the raw
  episode directly over HTTP, nothing local but the converted output.
  download pulls it to `huggingface_hub`'s normal local cache first (reused
  automatically on a repeat download-mode run), then reads it locally. We
  measured streaming+extract at only ~15s of a ~83s total episode, so
  download-first mainly buys network/compute overlap across episodes, not a
  fix for the actual bottleneck (decode) -- verified both modes produce
  identical, correctly-readable output.
- `--sequential-camera-decode` (default: parallel): decode a episode's
  cameras one after another instead of concurrently. Measured (3 cameras):
  ~172s/episode sequential vs. ~83s parallel -- slower per episode, but the
  memory estimate is correspondingly lower too (~4.7GB vs ~7.0GB
  budgeted/episode). Since `max_concurrent` is usually memory-bound in
  practice, sequential decode can let more episodes convert at once and win
  on **aggregate** throughput despite being slower per episode --
  situational, worth measuring on your own machine.
- `--max-concurrent`: override the auto-computed concurrency bound directly,
  if you've measured your machine's real headroom and want to be more
  aggressive than the conservative default. Set deliberately -- the default
  bound exists specifically because of a real 29GB OOM from before it did.

## Where the converted LeRobot v3 dataset lives: local, S3, or GCS

`--v3-root` (both `train.py` and `prepare_data.py`) accepts a local path,
`s3://<bucket>/<prefix>`, or `gs://<bucket>/<prefix>` -- the converted
parquet + mp4 dataset can be written straight to cloud storage instead of
local disk, and Ray Data reads it back the same way regardless.

The read side needed no changes to support this:
`training/vendor/lerobot_datasource.py` already went through
`fsspec.core.url_to_fs()` for both metadata and video reads (including PyAV
decoding straight from a remote file handle) before any of this project's
own code existed. `training/data_prep/lerobot_v3_writer.py` goes through the
same `fsspec` call on the write side (`training/data_prep/source.py`'s
`open_fs()`) for every parquet and JSON file -- these are safe to write
directly to a remote handle since neither format needs to seek backward into
what it's already written. Video is the one exception: encoding always goes
to a real local temp file first, then gets uploaded via `fs.put()` once
finished, rather than muxing directly onto a remote write stream -- standard
(non-fragmented) mp4 muxing patches byte offsets into an already-written
header once the final size is known, which needs a seekable output, and
fsspec's remote write handles (`s3fs`, `gcsfs`) are forward-only upload
streams, not seekable.

**Only local `--v3-root` has been run against real data** (full convert +
read-back verified, including real video decode). `s3://`/`gs://` are
implemented the same way -- same `fsspec` call the already-proven read side
uses -- but unverified against a real bucket: `s3fs`/`gcsfs` aren't even
installed by default (`pip install s3fs gcsfs` to use them).

## Using data you already have

`train.py` never discovers/converts anything itself -- it just reads
whatever LeRobot v3 dataset is already at `--v3-root`, from any source
(built by `prepare_data.py`, hand-built, downloaded pre-converted), and
errors out with the `prepare_data.py` command to run if nothing's there. No
`HF_TOKEN` is needed to run `train.py` for this reason.

This means `train.py` performs no staleness check -- it trusts
`--v3-root`'s contents as-is, unlike `prepare_data.py`'s own `is_prepared()`
check (which requires an exact match to a conversion `prepare_data.py`
itself tracked, specifically to avoid silently reconverting stale/mismatched
data). If you change `--max-episodes-per-task` or a schema's camera/tick
settings, re-run `prepare_data.py --reconvert` -- `train.py` has no way to
know your data is stale.

## Command reference

### `training/data_prep/discover.py --dataset-source abc130k` -- mcap strategy only

Lists real task names + episode counts (from the cached listing, or a fresh
scan on first run). Also writes/refreshes
`training/.cache/<dataset_source>_listing.json`.

```bash
python -m training.data_prep.discover --dataset-source abc130k
```

### `training/prepare_data.py` -- data prep only, no GPU needed

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | one or more task-name substrings |
| `--dataset-source` | `abc130k` | `abc130k`, `agibot_alpha`, or `droid` -- see [Customizing datasets](customizing-datasets.md); choices come from `training/data_prep/schemas/*.yaml` |
| `--max-episodes-per-task` | `300` | cap per task (fewer if a task has less) |
| `--source-uri` | the chosen `--dataset-source`'s own `default_source_uri` | `hf://datasets/<repo_id>`, `s3://<bucket>/<prefix>`, or `gs://<bucket>/<prefix>` -- mcap strategy only |
| `--v3-root` | `training/lerobot_v3/<dataset-source>/<sorted tasks>` | where the converted dataset lives |
| `--reconvert` | off | wipe and rebuild from scratch, ignoring any existing/incremental data |
| `--refresh-listing` | off | re-scan the full source tree instead of the cached listing -- mcap strategy only |
| `--mode` | `stream` | `stream` (nothing local but the output) or `download` (caches the raw file locally first) -- mcap strategy only |
| `--sequential-camera-decode` | off (parallel) | decode a episode's cameras one at a time instead of concurrently -- mcap strategy only |
| `--max-concurrent` | auto (from live CPU/memory) | override the concurrent-episode-conversion bound directly -- mcap strategy only |

```bash
python -m training.prepare_data --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300 --mode stream --max-concurrent 4

# Full run, tuned: download mode, sequential decode for more concurrency
python -m training.prepare_data --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300 --mode download --sequential-camera-decode --max-concurrent 6

# Same dataset mirrored to S3 instead of HF -- no HF_TOKEN needed
# (uses ambient AWS credentials, unverified against a real bucket -- see above)
python -m training.prepare_data --tasks arrange_the_flowers --source-uri s3://my-bucket/abc-130k-mirror
```

## Next

[4. Train](04-train.md)
