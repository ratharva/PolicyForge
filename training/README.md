# PolicyForge -- ACT / MolmoAct2 / π0.5 training on XDOF/ABC-130k

Ray Data + Ray Train pipeline: pick task names, cap episodes per task, stream
MCAP directly from the gated HF dataset and convert to LeRobot v3 on the fly
(no local .mcap ever written), then train a policy with DDP across every GPU
using `training/vendor/lerobot_datasource.py`'s already-proven streaming
Datasource. Three selectable policies (`--policy-type`), all bundled in the
same `lerobot==0.6.1` install -- no separate environment for any of them
(see Setup below): lerobot's ACT (default -- tens of millions of params,
DDP, this repo's original target), MolmoAct2 (`allenai/molmoact2`'s ~8B-param
VLA -- LoRA/DDP or full-finetune/FSDP2, see "MolmoAct2" below), and π0.5
(`lerobot.policies.pi05`'s ~2.3B-param VLA, finetuned from a real pretrained
checkpoint via `from_pretrained` -- DDP only, see "π0.5" below). Adding each
new policy didn't touch the previous ones' code paths or defaults;
everything policy-specific is dispatched through `training/model/registry.py`.

## Layout

```
training/
  config.py            all tunables (data, ACT hyperparams, training)
  data/
    discover.py         list real task names on HF, cap episodes (listing cached);
                          download_episodes() still exists but only verify_decode.py uses it
    decode.py            MCAP -> raw per-topic message streams (protobuf decode);
                           read_raw_episode_streaming() sources bytes straight from the
                           Hub over HTTP, read_raw_episode() reads a local file (verify_decode.py)
    video_decode.py       raw H.264/H.265 elementary stream -> RGB frames (used during conversion)
    align.py               floor-align independently-timed streams to a common tick grid
    episode.py               orchestrates one episode: raw -> aligned
    lerobot_v3_writer.py      writes one aligned episode into LeRobot v3 layout
    convert.py                orchestrates conversion: Ray-parallel per episode (streamed) + a
                               cheap sequential pass for global indices + dataset metadata
    prepare.py                 shared discover -> convert pipeline,
                                used by both prepare_data.py and train.py
    stats.py                  approximate normalization stats, sampled from the built dataset
    ray_dataset.py            build_lerobot_v3_dataset() (train.py's default) +
                               build_dataset_direct() (debug fallback, no conversion)
  model/
    act.py               builds lerobot's ACTConfig/ACTPolicy + normalization pipeline
    molmoact2.py          builds lerobot's MolmoAct2Config/MolmoAct2Policy + normalization
                           pipeline, mirrors act.py's contract; also the FSDP2/accelerate
                           wrap_for_training + sharded save/load_checkpoint (see "MolmoAct2" below)
    pi05.py               builds lerobot's PI05Config/PI05Policy from a real pretrained
                           checkpoint (from_pretrained) + normalization pipeline, mirrors
                           act.py's/molmoact2.py's contract; DDP only, no wrap_for_training/
                           save_checkpoint/load_checkpoint (see "π0.5" below)
    registry.py            PolicyAdapter dispatch table (get_adapter) train_loop.py uses instead
                            of hardcoding which policy it's running -- lazy per-policy imports,
                            so an ACT-only environment never touches molmoact2.py at all
  train_loop.py         Ray Train per-worker loop -- policy-agnostic (dispatches through
                         registry.py's PolicyAdapter), DDP or FSDP2, checkpointing
  prepare_data.py       standalone: discover -> stream -> convert, nothing else. Run
                         this ahead of time to decouple data prep from training.
  train.py              entrypoint: (prepare, if not already done) -> build dataset -> launch Ray Train
  verify_decode.py      RUN THIS FIRST on one real episode, before ever converting/training
                         (this one still downloads locally -- convenient for repeated inspection)
```

## Setup

```bash
pip install -r training/requirements.txt
export HF_TOKEN=$(cat ../hf_tok.txt)   # after accepting XDOF/ABC-130k's gated terms
```

Run everything from the repo root (parent of this `training/` directory), as
`python -m training.<module>` -- the package imports (`from training.config
import ...`, `from training.vendor.lerobot_datasource import ...`) need it on the path.

### MolmoAct2 environment

**No separate environment needed.** `pip install -r training/requirements.txt`
(the same install ACT uses) is enough for `--policy-type molmoact2` too --
`MolmoAct2Config`/`MolmoAct2Policy` are bundled in stock PyPI
`lerobot>=0.6.1`, confirmed directly against the real public PyPI wheel
(`lerobot/policies/molmoact2/` is really in there) and by actually
constructing a real `MolmoAct2Config` in this environment, not just an
import check.

The only extra dependency is `accelerate` (commented-out in
`requirements.txt`), and only for `--molmoact2-distributed-strategy fsdp2`
(full fine-tuning at scale) -- `--molmoact2-train-mode lora` (the default)
needs nothing beyond the base install.

**Correction, kept here because it was a real mistake worth learning from**:
earlier versions of this section said MolmoAct2 needed a separate fork
(`allenai/lerobot@molmoact2-policy`) in its own environment, with a whole
`requirements-molmoact2.txt` and a two-environment split. That was based on
research against the fork's own branch, which genuinely did have MolmoAct2
before mainline did -- but by the time this pipeline pinned
`lerobot==0.6.1` in `requirements.txt`, mainline had already absorbed the
same policy, and that was never cross-checked against what was actually
installed in the dev environment already being used for ACT. Running
`python -c "from lerobot.policies.molmoact2... import MolmoAct2Config"`
against the environment already sitting there would have caught this
immediately -- the general lesson (echoed in `CLAUDE.md`): verify against
what's *actually pinned/installed*, not just against wherever the research
happened to look.

## Why convert to LeRobot v3 instead of training directly off MCAP

Decoding MCAP (protobuf parse + floor-alignment + raw H.264/H.265 elementary-
stream decode) is real work that a direct-MCAP pipeline pays on **every**
training run. Converting once instead:
- pays that cost a single time, cached by task set under `training/lerobot_v3/`
- lets training reuse `training/vendor/lerobot_datasource.py` -- a proven,
  partitioned, memory-disciplined streaming reader -- instead of a
  bespoke `flat_map`
- writes real mp4 containers (not raw elementary streams), so the read side
  is the same well-tested `av.open(path)` container path notebooks 01/02 use,
  not the riskier raw-stream decode conversion itself still has to do once

This is a net win once you'll train more than once or twice on the same
episode set. For a single one-shot smoke test, converting first is strictly
more work, not less -- `build_dataset_direct()` in `ray_dataset.py` is kept
around for exactly that case (not wired into `train.py` by default).

## Data source: HF, S3, or GCS

`DataConfig.source_uri` (`--source-uri` on `train.py`/`prepare_data.py`)
identifies where the raw MCAP dataset lives:
- `hf://datasets/<repo_id>` (default: `hf://datasets/XDOF/ABC-130k`) -- Hugging
  Face Hub, needs `HF_TOKEN` if the repo is gated.
- `s3://<bucket>/<prefix>` or `gs://<bucket>/<prefix>` -- relies on ambient
  credentials (the standard boto3/gcloud credential chain), no token needed.

Streaming reads (`training/data/decode.py`'s `read_raw_episode_streaming`) go
through `fsspec.core.url_to_fs()` for all three -- the same pattern
`training/vendor/lerobot_datasource.py` already uses for the *converted* LeRobot
v3 side, extended here to the raw MCAP input, which was hardcoded to
`huggingface_hub.HfFileSystem` before. Listing (`discover.py`) and download
(`--mode download`, `verify_decode.py`) deliberately keep `hf://` on
`huggingface_hub`'s own `HfApi`/`hf_hub_download` rather than routing it
through the same generic path -- that's what's proven fast and correct
against the real 176k-file XDOF repo, and there's no reason to risk a
regression on an already-working path for uniformity. `s3://`/`gs://` use
fsspec's generic `fs.find()`/`fs.get()` instead, since there's no prior
working code there to preserve.

**Only `hf://` has been run against real data.** `s3://`/`gs://` are
implemented but unverified -- no credentials were available in this
environment to test against a real bucket. The refactor that added them
*was* re-verified against the real gated HF repo end-to-end (stream a real
episode through the new `source_uri`-based code path, convert, read back) to
confirm no regression on the path that was already working.

**The dataset schema lives per-robot, in `training/robots.py`, not hardcoded
in the pipeline code.** Each robot's teleop-station export convention is one
`RobotSchema` subclass: topic names (`state_topics`, `action_topics`,
`top_camera_candidates`, `wrist_camera_topics`), the within-message field
names `data/decode.py`'s `_extract()` reads via `getattr()`
(`state_value_field`, `camera_data_field`, `camera_format_field`), the
recording tick rate (`tick_fps`), and the episode-relative-path ->
`(split, task)` parser (`path_to_split_and_task()`, a method -- used by
`discover.py`'s `list_episodes_by_task()`). `DataConfig.robot` holds the
active instance; everything downstream reads schema values off
`cfg.robot.X`, not off `DataConfig` directly.

The only robot implemented so far is `XDOFABCRobot` (`state_value_field=
"position"`, `camera_data_field="data"`, `camera_format_field="format"`,
`tick_fps=30.0`, `data/{split}/{task}/episode_{uuid}/episode.mcap` layout --
XDOF/ABC-130k's own protobuf schema and directory convention). A different
robot dataset with the *same shape* of schema (protobuf state/action with a
position-like value field, video messages with a data+format-like pair, a
similar directory convention) is a new `RobotSchema` subclass in
`robots.py` -- swap `DataConfig(robot=YourRobot())` in, no other code change.
A robot with a genuinely different message shape -- not protobuf, or e.g.
ROS `sensor_msgs` instead of a custom protobuf schema -- needs new extraction
logic in `decode.py`'s `_extract()` itself; `RobotSchema`'s fields cover
naming differences within the same message shape, not message-shape
differences.

## Streaming conversion (no local MCAP)

`convert.py` streams each episode.mcap directly from `source_uri`'s backend
over the network (`training/data/decode.py`'s `read_raw_episode_streaming`)
-- nothing lands on local disk except the converted LeRobot v3 output
(parquet + mp4). Verified
live against the real gated repo, with per-step timing, not just "it worked":
streaming + protobuf-parsing a full episode took ~15s; video decode -- H.264
decode of 3 cameras -- was the real bottleneck at ~156s of a ~172s total
(91%), not the network. Cameras now decode concurrently
(`concurrent.futures.ThreadPoolExecutor` inside each episode's Ray task,
`CONVERT_CPUS_PER_EPISODE=3` accounted for in both the per-task CPU request
and the cross-episode concurrency bound) instead of one after another --
measured ~83s end to end after that change, near the theoretical ~3x on the
decode portion since PyAV releases the GIL during actual codec work.

One real gotcha found and worked around: `mcap`'s `make_reader()`
auto-detects `SeekingReader` vs. `NonSeekingReader` by calling `.seekable()`
on the stream, but `huggingface_hub`'s streaming file object reports
`seekable() == True` while `.seek()` actually raises -- so `NonSeekingReader`
is forced explicitly rather than trusting that auto-detection.

Trade-off versus the old download-then-convert path: no local MCAP storage
cost (previously ~170MB/episode budgeted below), but no HF download-cache
reuse either -- reconverting the same episodes re-streams them over the
network every time rather than reading an already-downloaded local copy.
`discover.py`'s `download_episodes()` still exists (only `verify_decode.py`
uses it now) if you want a local copy for some other reason.

## Incremental conversion

Asking for more episodes than a previous conversion (e.g. `--max-episodes-per-task 15`
after an earlier `10`) reuses the already-converted episodes and only
streams/decodes the new ones (`meta/episode_manifest.json` tracks which HF
episode maps to which local `episode_index`) -- it used to either silently
serve the stale smaller dataset, or (once that was fixed) blindly reconvert
everything including what was already done. Dropping episodes/tasks instead
(shrinking the request) falls back to a full wipe + rebuild: LeRobot v3
requires contiguous 0-based episode indices, so removing one in place would
mean renumbering and renaming every later episode's files -- not implemented.
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

`ConvertConfig` (`config.py`) exposes both, as CLI flags on `train.py` and
`prepare_data.py`:

- `--mode stream` (default) / `--mode download`: stream reads MCAP directly
  from HF over HTTP, nothing local but the converted output. download pulls
  each episode.mcap to huggingface_hub's normal local cache first (reused
  automatically on a repeat download-mode run), then reads it locally. We
  measured streaming+extract at only ~15s of a ~83s total episode, so
  download-first mainly buys network/compute overlap across episodes, not a
  fix for the actual bottleneck (decode, see below) -- verified both modes
  produce identical, correctly-readable output.
- `--sequential-camera-decode` (default: parallel): decode a episode's 3
  cameras one after another instead of concurrently. Measured: ~172s/episode
  sequential vs. ~83s parallel -- slower per episode, but the memory estimate
  is correspondingly lower too (~4.7GB vs ~7.0GB budgeted/episode -- see
  `_estimate_episode_memory_bytes`'s docstring for the reasoning, which is a
  reasoned adjustment, not independently measured peak memory). Since
  `max_concurrent` is usually memory-bound in practice (see the OOM incident
  above), sequential decode can let more episodes convert at once and win on
  **aggregate** throughput despite being slower per episode -- situational,
  worth measuring on your own machine, not a strict improvement either way.
- `--max-concurrent`: override the auto-computed concurrency bound directly,
  if you've measured your machine's real headroom and want to be more
  aggressive than the conservative default. Set deliberately -- the default
  bound exists specifically because of a real 29GB OOM from before it did.

## Using data you already have

`--use-existing-v3` trusts whatever's at `--v3-root` as-is and skips
discover/convert entirely (no `HF_TOKEN` needed) -- for a LeRobot v3 dataset
from any source: hand-built, downloaded pre-converted, or a prior run of this
pipeline. Unlike the normal `is_prepared()` staleness check (which requires
an exact match to a conversion this pipeline tracked itself, specifically to
prevent silently training on stale/mismatched data -- see the "why is it only
converting 1 concurrent" and "10 vs 15 episodes" incidents above),
`--use-existing-v3` is an explicit opt-out for data you're vouching for
yourself, not a replacement for that check.

## Where the converted LeRobot v3 dataset lives: local, S3, or GCS

`--v3-root` (both `train.py` and `prepare_data.py`) accepts a local path,
`s3://<bucket>/<prefix>`, or `gs://<bucket>/<prefix>` -- the converted
parquet + mp4 dataset can be written straight to cloud storage instead of
local disk, and Ray Data reads it back the same way regardless.

The read side needed no changes to support this: `training/vendor/lerobot_datasource.py`
already went through `fsspec.core.url_to_fs()` for both metadata and video
reads (including PyAV decoding straight from a remote file handle) before any
of this project's own code existed. `training/data/lerobot_v3_writer.py` now
goes through the same `fsspec` call on the write side (`training/data/source.py`'s
`open_fs()`) for every parquet and JSON file -- these are safe to write
directly to a remote handle since neither format needs to seek backward into
what it's already written. Video is the one exception: encoding always goes
to a real local temp file first, then gets uploaded via `fs.put()` once
finished, rather than muxing directly onto a remote write stream -- standard
(non-fragmented) mp4 muxing patches byte offsets into an already-written
header once the final size is known, which needs a seekable output, and
fsspec's remote write handles (`s3fs`, `gcsfs`) are forward-only upload
streams, not seekable. This only affects where the finished file lands, not
its contents.

**Only local `--v3-root` has been run against real data** (full convert +
read-back verified, including real video decode). `s3://`/`gs://` are
implemented the same way -- same `fsspec` call the already-proven read side
uses -- but unverified against a real bucket: `s3fs`/`gcsfs` aren't even
installed in this environment (`pip install s3fs gcsfs` to use them), so
there was no way to test beyond confirming the local path still works
identically after this change.

## Training on a remote GPU provider (e.g. Thunder Compute)

Thunder Compute (thundercompute.com) is a GPU rental provider -- SSH-accessible
instances via its `tnr` CLI, not a cloud with its own S3/GCS-equivalent object
storage. That changes the efficient transfer story slightly from a same-cloud
setup: Thunder isn't AWS or GCP, so the "same region as your bucket = free
egress" point from the cost discussion above doesn't apply -- every byte read
from S3/GCS into a Thunder instance is billed at that provider's egress rate,
not $0. Everything below is reasoned from Thunder's own published CLI docs
(github.com/Thunder-Compute/thunder-compute-documentation), not run against a
real Thunder account from this environment -- verify the exact commands
against `tnr --help` before relying on them.

**Code**: this repo has no data in `training/` itself (only under `.cache/`,
`lerobot_v3/`, `runs/`, all already gitignored/excluded via
`build_runtime_env()`'s excludes) -- small enough that either works:
- `tnr scp -r training <instance_id>:~/ray_test/` -- direct copy,
  uses `rsync` under the hood on Linux/macOS (progress display, recursive).
- Push to a git remote and `git clone` on the instance instead, if you expect
  to iterate -- avoids re-copying on every change, and `tnr connect
  <instance_id>` drops you into a normal shell where `git pull` works exactly
  like any other machine.

**Data**: for anything more than a few GB, Thunder's own docs recommend
skipping `tnr scp` (which round-trips through your local machine's upload
bandwidth) in favor of cloud storage the instance downloads directly --
exactly what `--v3-root s3://...`/`gs://...` (see above) is for. Two ways to
use it once the instance has AWS/GCP credentials (env vars, or
`aws configure`/`gcloud auth` on the instance) and `s3fs`/`gcsfs` installed:
- **Point `--v3-root` straight at the bucket** and let Ray Data stream it in
  at training time -- simplest, but every epoch's worth of reads is billed
  egress since Thunder isn't in AWS/GCP. Fine for `num_epochs=1` (this repo's
  default) or a quick check; adds up over many epochs.
- **Sync to the instance's local disk once**, then point `--v3-root` at that
  local path instead: `aws s3 sync s3://bucket/prefix ~/lerobot_v3/` or
  `gcloud storage cp -r gs://bucket/prefix ~/lerobot_v3/`. One egress charge
  for the whole dataset instead of one per epoch. The instance's disk
  persists across stop/start via Thunder's snapshots (`tnr snapshot create`),
  so this only needs to happen once per instance, not once per run --
  Thunder's own docs note snapshots are for convenience/fast restore, not
  guaranteed long-term durability, so the S3/GCS copy stays the source of
  truth either way.

**Ray Dashboard**: `ray.init()`'s dashboard binds to `127.0.0.1` (loopback)
by default -- unreachable from outside the instance no matter what, so
`training/ray_setup.py`'s `connect_ray()` now passes
`dashboard_host="0.0.0.0"` when it starts a local Ray instance (not needed if
you're connecting to an already-running external cluster started with
`ray start --head --dashboard-host=0.0.0.0`). With that, forward the port
through Thunder's CLI from your own machine, not the instance:
```bash
tnr ports forward <instance_id> --add 8265
```
The dashboard is then reachable at `https://<instance-uuid>-8265.thundercompute.net`
(HTTPS, from any browser, no SSH tunnel needed) -- per Thunder's docs, port
forwarding only speaks HTTP/gRPC, which the dashboard's web UI already is.
`tnr ports list` shows everything currently forwarded.

## MolmoAct2

`--policy-type molmoact2` trains lerobot's `MolmoAct2Policy` -- a
~8B-parameter VLA (Qwen2-7B-scale LLM + SigLIP ViT + a flow-matching action
expert), a much bigger jump from ACT than the flag name suggests. Same
environment as ACT, no separate install (see Setup above).

**Verification status, precisely**: `MolmoAct2Config`/`MolmoAct2Policy`'s
field names, `get_optim_params()`'s real 4-group structure,
`forward()`'s signature, the real `train_mode`-to-boolean-flags mapping
(`enable_lora_vlm`/`enable_lora_action_expert`/`train_action_expert_only`),
and the gradient-checkpointing method name were all confirmed by direct
introspection AND real construction calls against the installed package in
this environment -- not just import success, actual
`MolmoAct2Config(...)` calls with real validation passing for all three
`train_mode` values. This caught several concrete mismatches from earlier
fork-based research (no such field as `train_mode_vlm`; the real
gradient-checkpointing method is `_enable_gradient_checkpointing`, not
`gradient_checkpointing_enable`; `lora_rank`/`lora_alpha`'s real defaults
are 64/16, not 16/32) -- all now fixed. What's still NOT verified: actually
constructing a real `MolmoAct2Policy` (needs downloading the ~22GB
checkpoint, not done here -- disk headroom and GPU memory on this dev
machine are both tight, see the cost/hardware notes below) and running a
real forward/backward pass. Re-verify against a live install if you bump
the lerobot version:

```bash
python -c "from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config; \
    import dataclasses; [print(f.name) for f in dataclasses.fields(MolmoAct2Config)]"
```

### LoRA (default) vs full fine-tune

`--molmoact2-train-mode lora` (default) enables LoRA on the VLM only (the
action expert stays fully trainable -- matches the MolmoAct2 team's own
training recipe) and fits under plain DDP (`ray.train.torch.prepare_model`,
same mechanism ACT uses) on one modern GPU -- ~20GB VRAM at batch 8 by
real-world reports. This is the starting point; get it working before
considering `fft`. `--molmoact2-train-mode freeze` is a third option: VLM
completely frozen, only the action expert trains (requires
`--molmoact2-action-mode continuous`, enforced by `MolmoAct2Config`'s own
validation with a clear error otherwise).

`--molmoact2-train-mode fft` (full fine-tune) needs
`--molmoact2-distributed-strategy fsdp2` to be practical (~60GB/GPU at
batch 32 under plain DDP otherwise -- FSDP2 shards
parameters/gradients/optimizer state across GPUs instead). This does NOT
use Ray Train's own `prepare_model(parallel_strategy="fsdp")`, which wraps
the legacy FSDP1 API (`torch.distributed.fsdp.FullyShardedDataParallel`) --
verified from installed Ray source. Instead, `training/model/molmoact2.py`'s
`wrap_for_training()` constructs an `accelerate.Accelerator` with
`FullyShardedDataParallelPlugin(fsdp_version=2, ...)` inside the Ray Train
worker function itself, the same mechanism lerobot's own official
`accelerate launch -m lerobot.scripts.lerobot_train` training path uses.
This composes cleanly with Ray Train: verified by reading both
`accelerate.state.PartialState.__init__` (checks
`if not torch.distributed.is_initialized()` before calling
`init_process_group`, i.e. it reuses an existing process group rather than
fighting it) and Ray Train's own worker-group startup sequence
(`ReplicaGroup.start_training()` runs backend setup callbacks, including
`TorchBackend.on_start()`'s process-group init, synchronously via
`ray.get(...)` *before* dispatching the training function to any worker) --
so the process group Ray Train sets up is unconditionally ready before
`accelerate.Accelerator()` is ever constructed.

`auto_wrap_policy="transformer_based_wrap"` needs real transformer layer
class names to find anything to wrap. `wrap_for_training()` determines
these **dynamically from the actual constructed `policy`** rather than
hardcoding a fixed list -- a real, live-tested reason why: the LLM
backbone has two decoder-layer variants,
`MolmoAct2DecoderLayer`/`MolmoAct2PostNormDecoderLayer`
(`MolmoAct2PostNormDecoderLayer` subclasses the other), and exactly one is
chosen per checkpoint at construction time
(`config.norm_after`) -- **never both**. `accelerate`'s own class-matching
(`get_module_class_from_name`) is exact `type(module).__name__ ==` string
comparison, not `isinstance`-based (verified by reading its source), so
listing both unconditionally raised a real, reproducible
`ValueError: Could not find the transformer layer class ... in the model`
against a fake-policy integration test that only had one of the two.
`wrap_for_training()` now filters its candidate class list down to
whichever ones are actually present on `policy.modules()` before building
the FSDP plugin, so either checkpoint variant works without needing to know
which one in advance. The four real candidate names
(`MolmoAct2DecoderLayer`/`MolmoAct2PostNormDecoderLayer`/
`MolmoAct2VisionBlock`/`ActionExpertBlock`) were found by reading the
installed package's source directly, not guessed.

### FSDP2 checkpointing

FSDP2-sharded `policy.state_dict()` returns per-rank shards/DTensors, not a
normal full state dict, so the DDP path's pickle-one-rank-0-state-dict
approach doesn't work here. `training/model/molmoact2.py`'s
`save_checkpoint`/`load_checkpoint` use `accelerate.Accelerator.save_state`/
`load_state` instead. Their own docstring scopes them to *"restoring the
state in the same environment"* -- a Ray Train worker restart
(`ray.train.get_checkpoint()` after a failure) is a different process,
potentially a different node, not obviously "the same environment" in that
sense.

**Verified live**, not just reasoned about: a fake-policy integration test
(same interface real `MolmoAct2Policy` needs -- `forward`, `get_optim_params`,
`_enable_gradient_checkpointing` -- but a tiny model, since the real ~8B model
isn't installed here and wouldn't fit this dev machine's 12GB GPU) exercised
the real `wrap_for_training`/`save_checkpoint`/`load_checkpoint` code through
a deliberately-simulated mid-run worker crash (`FailureConfig(max_failures=1)`
+ an injected exception). Ray Train relaunched a genuinely new worker
process, which successfully loaded the sharded FSDP2 checkpoint ("All model
weights loaded successfully", "All optimizer states loaded successfully")
and correctly resumed training to completion. This confirms the specific
single-node worker-restart scenario works; `save_state`'s broader "same
environment" scoping (different GPU topology, genuine multi-node) remains
unverified. `train_loop.py`'s FSDP2 checkpoint flow's other defensive
detail -- every rank writing to one fixed, deterministic shared directory
with a `torch.distributed.barrier()` before any rank can overwrite it for
the next checkpoint -- is exercised by this same test (multiple checkpoints
saved successfully across the run) but not independently stress-tested
under real multi-GPU contention.

**A real, more serious bug was caught and fixed by this test**, unrelated to
FSDP2 itself and affecting ACT equally: `train_loop_per_worker`'s resume
logic computed `start_epoch = state["epoch"] + 1` unconditionally, which is
only correct when resuming from an end-of-epoch checkpoint. A step-windowed
(mid-epoch) checkpoint -- the ones this codebase takes every
`--eval-every-steps`, its main checkpointing mechanism -- has `epoch` equal
to whatever epoch was still in progress, so `epoch + 1` always looked like
"the next epoch" and, combined with `num_epochs=1` (this codebase's own
documented typical setting), `start_epoch >= num_epochs` always
false-positived as "run already complete" -- silently discarding all
progress back to epoch start on **any** resume, for both ACT and MolmoAct2,
regardless of distributed strategy. Fixed by threading an `epoch_complete`
flag through the checkpoint state (`_report_with_checkpoint`, both the
pickle and FSDP2 save paths) so resume only advances to the next epoch when
the checkpoint actually represents one finishing. Re-verified with the same
crash+relaunch test (ACT, MolmoAct2/DDP, and MolmoAct2/FSDP2 all now
correctly resume to the original target step count, not zero).

### Ray Data preprocessing offload

MolmoAct2's preprocessor (`make_molmoact2_pre_post_processors`) does real
CPU-bound work per batch -- its `MolmoAct2PackInputsProcessorStep` builds a
natural-language prompt per sample and runs it through a real HF tokenizer +
image processor. By default this runs inline in `train_loop.py`
(`inputs = preprocessor(batch)`), on the same thread as the GPU
forward/backward -- `--molmoact2-offload-tokenization` instead runs the
*entire* preprocessor as a Ray Data `map_batches` stage
(`training/data/ray_dataset.py`'s `offload_molmoact2_preprocessing`,
`concurrency` defaulting to half the live CPU count the same way
`convert.py`'s `max_concurrent` derives from live resources), so it scales
across the cluster's CPU workers independently of GPU worker count -- the
same principle this pipeline already uses for video decode/transpose.

**Unverified**: whether HF tokenizer/processor output (token ID sequences,
attention masks, possibly padded to a batch-dependent max length)
round-trips correctly through Ray Data's Arrow-backed batch representation
and then into `training/vendor/util.py`'s `NumpyToTorchCollate` was not
confirmed. Construct a real batch, run it through
`offload_molmoact2_preprocessing`, and compare tensor shapes/dtypes against
the non-offloaded path before trusting this for a real run -- and confirm
throughput actually improves (the whole point of this mode) via the Ray
dashboard / `ray_data_stats.txt`.

### `setup_type` / `control_mode`

Required free-text prompt fields MolmoAct2's VLM conditions on --
`MolmoAct2PackInputsProcessorStep` raises `ValueError` if either is missing.
These describe the robot/task, not the MCAP schema, so they live on
`MolmoAct2ConfigOverrides` (via `--molmoact2-setup-type`/
`--molmoact2-control-mode`), not on `training/robots.py`'s `RobotSchema` --
`RobotSchema`'s scope is MCAP/hardware facts independent of which policy
trains on the data (topic names, message fields, tick rate, directory
layout), and forcing every future `RobotSchema` subclass to carry
MolmoAct2-specific prompt fields it can't populate would break that
contract for one policy's benefit. For `XDOFABCRobot`, something like
`--molmoact2-setup-type "dual-arm robot with wrist and top cameras"
--molmoact2-control-mode "delta joint position"` is a reasonable starting
point -- not verified against real training quality.

### Normalization

MolmoAct2's own default is quantile-based normalization for state/action
(`STATE`/`ACTION: QUANTILES`); `training/model/molmoact2.py` overrides this
to `MEAN_STD` for all three (`VISUAL`/`STATE`/`ACTION`) at construction so
`training/data/stats.py`'s existing `compute_dataset_stats` (mean/std only)
works unchanged -- a deliberate simplification, not a claim that it's
optimal. Forcing `MEAN_STD` for `VISUAL` too (the real default there
is `IDENTITY`) construction-succeeds (verified via a real
`MolmoAct2Config(...)` call) but is still unverified to produce numerically
correct results end-to-end inside `make_molmoact2_pre_post_processors` --
that needs a real batch through the real processor, not just successful
construction.

## π0.5

`--policy-type pi05` trains lerobot's `PI05Policy` -- a PaliGemma-based VLM
backbone (`gemma_2b`) + a separate, smaller flow-matching action expert
(`gemma_300m`), roughly ~2.3B params total by variant naming, well below
MolmoAct2's ~8B. Same environment as ACT/MolmoAct2, no separate install (see
Setup above).

Unlike ACT (trains from random init in this pipeline) and MolmoAct2 (loads
its own weights internally via a `checkpoint_path` config field), π0.5 is
the first policy here that goes through lerobot's standard
`PreTrainedPolicy.from_pretrained(pretrained_name_or_path, config=...)`
classmethod -- "finetuning" implies starting from real pretrained weights,
not random init, so `--pi05-pretrained-path` is a **required** flag with no
default. No real public HF checkpoint repo id for a pi05 base/finetunable
checkpoint was looked up or verified in this project (matches the standing
rule against guessing repo ids/URLs) -- find a real one on the HF Hub
yourself before running this for real.

**Verification status, precisely**: `PI05Config`/`PI05Policy`'s field names,
`get_optim_params()`'s real flat (ungrouped) structure, `forward()`'s
signature, `from_pretrained`'s signature, and the gradient-checkpointing
auto-wiring were all confirmed by direct introspection AND a real
`PI05Config(...)` construction call for all three real training modes (full
fine-tune, freeze-vision, expert-only) against the installed package --
applying the "verify against what's actually installed" lesson from the
MolmoAct2 correction (see "MolmoAct2 environment" above) from the start,
which is presumably why this integration had zero real API mismatches on
the first attempt, unlike MolmoAct2's first pass. What's still NOT
verified: an actual `PI05Policy.from_pretrained(...)` call against a real
checkpoint (none was downloaded -- no verified real checkpoint repo id, see
above) and a real forward/backward pass. Re-verify against a live install if
you bump the lerobot version:

```bash
python -c "from lerobot.policies.pi05.configuration_pi05 import PI05Config; \
    import dataclasses; [print(f.name) for f in dataclasses.fields(PI05Config)]"
```

### No LoRA, and no `train_mode` string

`PI05Config.use_peft` exists but is never referenced in `modeling_pi05.py`
(confirmed by grep) -- the same dead-field pattern `MolmoAct2Config.use_peft`
had, so there's no LoRA support in this integration. The only two real,
used training-surface reducers are independent booleans:
`--pi05-freeze-vision-encoder` (freezes the vision tower; language model +
action expert still train) and `--pi05-train-expert-only` (only the action
expert trains, everything else frozen) -- both confirmed consumed in
`PI05Policy.__init__`.

Unlike `MolmoAct2ConfigOverrides.train_mode`, `Pi05ConfigOverrides` exposes
both booleans directly rather than translating a single convenience string
into real config fields. This is deliberate, not an oversight: MolmoAct2's
`train_mode` translation layer was where the earlier `train_mode_vlm`
mismatch lived, and with only two independent real booleans (no evidence in
`__post_init__` of any invalid combination), a hand-maintained translation
layer here would add sync risk without adding real value.

`PI05Policy` has no `.train()` override (confirmed: `'train' not in
PI05Policy.__dict__`) -- freezing is via `requires_grad` only. Unverified:
whether any train/eval-sensitive layers (e.g. dropout) exist in the frozen
submodules such that this matters.

### DDP only -- no FSDP2

MolmoAct2's FSDP2 support was justified by its README's real ~60GB/GPU
full-finetune numbers; no equivalent real VRAM number exists for π0.5 (no
real forward/backward pass has been run -- see above), so fabricating one to
justify building FSDP2 upfront would be exactly the kind of unverified claim
this project's discipline pushes back on. All three π0.5 training modes
(full fine-tune, freeze-vision, expert-only) use plain DDP
(`ray.train.torch.prepare_model`, same mechanism ACT uses,
`find_unused_parameters=True` as the same cheap defensive insurance
MolmoAct2's DDP path uses) -- `PolicyAdapter`'s existing `None`-default
`wrap_for_training`/`save_checkpoint`/`load_checkpoint` are reused unmodified,
same pickle-checkpoint path as ACT. If a real run later shows DDP doesn't
fit π0.5's ~2.3B params, extending to FSDP2 is a mechanical repeat of
`model/molmoact2.py`'s pattern, not new design work.

### Normalization

Same deliberate simplification as MolmoAct2: π0.5's own default
(`VISUAL: IDENTITY`, `STATE`/`ACTION: QUANTILES`) is overridden to
`MEAN_STD` for all three at construction, so `training/data/stats.py`'s
existing mean/std-only `compute_dataset_stats` works unchanged -- construction-
succeeds (verified via a real `PI05Config(...)` call), not verified to be
numerically optimal end-to-end.

## Before you trust this for a real run

Verified end-to-end at least once, not just reasoned about:
- The MCAP schema (topics, protobuf field names, message rates) was
  confirmed live against a real episode with a one-off probe script (not
  part of this repo).
- lerobot's `ACTConfig`/`ACTPolicy`/`make_act_pre_post_processors` API surface
  (field names, `forward()`'s `(loss, loss_dict)` return, the separate
  normalization pipeline) was confirmed by direct inspection of the installed
  package and a real forward+backward pass with dummy tensors.
- The LeRobot v3 writer/reader round-trip (`lerobot_v3_writer.py` ->
  `training/vendor/lerobot_datasource.py`) was verified with synthetic episodes:
  correct row counts, task labels, action chunking/padding, exact state/action
  value round-trip, and a real libx264 encode/decode round-trip.
- A single-worker (`world_size=1`) run surfaced `AttributeError: 'ACTPolicy'
  object has no attribute 'module'` at checkpoint time -- `prepare_model()`
  only wraps in DDP when there's more than one worker. Fixed in
  `train_loop.py` (unwraps defensively rather than assuming DDP).
- After adding MolmoAct2 as a second policy, `train_loop.py`'s optimizer
  construction was switched from a hand-rolled backbone-name-match to
  `policy.get_optim_params()` (a real lerobot API both `ACTPolicy` and
  `MolmoAct2Policy` implement) -- re-ran the ACT smoke test end-to-end
  (`--tasks dress_the_teddy_bear --use-existing-v3 --max-train-steps 25`)
  to confirm identical behavior: same checkpoint-retention pattern (best 3 +
  latest), correct metrics, no regression. Separately verified with a real
  `ACTPolicy` instance that `--lr-backbone` now actually reaches the
  optimizer's backbone param group (a real pre-existing gap this refactor
  fixed -- see `model/act.py`'s `build_act_config`), which it silently
  didn't before.
- **Resume-from-checkpoint, across a real crash and worker relaunch, for
  all three configurations** (ACT, MolmoAct2/LoRA/DDP, MolmoAct2/full-
  finetune/FSDP2): each run deliberately crashed mid-run
  (`FailureConfig(max_failures=1)` + an injected exception after a
  checkpoint had already been saved) and confirmed Ray Train relaunched a
  fresh worker process that resumed correctly to the original target step
  count. This caught and fixed a real bug affecting both policies equally
  (see "MolmoAct2 > FSDP2 checkpointing" above for the detail) -- resuming
  from any step-windowed checkpoint under `num_epochs=1` used to always
  silently discard all progress back to epoch start, for ACT too, not just
  MolmoAct2; nothing before this had ever exercised a real crash+resume to
  surface it.
- MolmoAct2's `accelerate`-based FSDP2 wrapping needed one additional fix
  found the same way: `Accelerator(mixed_precision="bf16")` casts model
  parameters to bf16, but batches stay whatever dtype the collate function
  produces (float32) -- without `accelerator.autocast()` wrapped around the
  forward call, this raised `mat1 and mat2 must have the same dtype` on the
  very first real step. Fixed in `train_loop.py`, scoped to the FSDP2 path
  only (`dist_ctx.autocast()` when `dist_ctx` is set).
- **π0.5 added as a third policy, applying the MolmoAct2-correction lesson
  from the start** (verify against the actually-installed package, not
  research alone): real `PI05Config(...)` construction calls for all three
  training modes, then a fake-policy DDP crash+resume integration test (same
  pattern as MolmoAct2's, adapted for π0.5's flat `get_optim_params()` shape)
  -- passed cleanly on the first attempt, and a real ACT regression run
  (`--max-train-steps 15`) confirmed the shared-code changes
  (`registry.py`/`config.py`) didn't affect ACT.

Still open / worth knowing about:

1. **Video decode during conversion** (`data/video_decode.py`) -- MCAP
   delivers camera streams as raw H.264/H.265 elementary bytes, not a
   container file, so decode goes through `PyAV.CodecContext.parse/decode`
   and frame timestamps are assigned by FIFO-pairing decode order with
   message order. This is wrong if the stream uses B-frames. **Run
   `python -m training.verify_decode --task <substring>` and look at the
   dumped PNGs** before converting anything for real -- they should be sharp
   and change smoothly across sampled frame indices.

2. **Single "top" camera simplification** -- `config.py`'s `DataConfig`
   deliberately keeps ACT's input shape fixed at 3 cameras
   (`top`, `left_wrist`, `right_wrist`) regardless of whether a station is
   mono or stereo, using only the left view when both `/top-left-camera` and
   `/top-right-camera` exist. Revisit if you specifically want stereo input.

3. **Single-node storage assumption, except for `--v3-root`** -- `--v3-root`
   can now point at `s3://`/`gs://` (see "Where the converted LeRobot v3
   dataset lives" above), which sidesteps this on a multi-node cluster. The
   `--mode download` HF cache and `--storage-root` (Ray Train checkpoints)
   still write to local disk read back by absolute path -- fine on a
   single-node setup (what this has been run on) or with shared storage; on a
   real multi-node cluster without a local `--v3-root`, those two still need
   to live on shared storage, e.g. `/mnt/cluster_storage/...`.

4. **`--policy-type molmoact2` -- the real ~8B model has never actually run
   end-to-end**, no GPU-capable environment was available in this
   development environment (12GB GPU here; real-world reports put LoRA
   training at ~20GB and even bf16 inference alone at ~12.1GB). What HAS
   been verified, and it's more than the model-construction level: (i)
   `MolmoAct2Config`/`MolmoAct2Policy` field names/API were confirmed by
   direct introspection AND real `MolmoAct2Config(...)` construction calls
   against the installed package (not just import success) -- this caught
   several concrete mismatches from earlier fork-based research (see
   "MolmoAct2 environment" above), now fixed; (ii) `wrap_for_training`'s
   `transformer_cls_names_to_wrap` uses real class names read from the
   installed source, filtered dynamically to whichever ones are actually
   present on the constructed policy (found and fixed a real mutual-
   exclusivity bug between the two decoder-layer variants this way -- see
   "MolmoAct2 > LoRA vs full fine-tune" above); (iii) a fake-policy
   integration test (same interface -- `get_optim_params`, `forward`,
   `_enable_gradient_checkpointing`, and now a submodule literally named to
   match a real MolmoAct2 class -- but a tiny model) exercised the real
   `registry.py`/`train_loop.py`/`molmoact2.py` code, including a simulated
   crash + Ray-Train-relaunch, for both LoRA/DDP and full-finetune/FSDP2 --
   caught and fixed a real dtype bug (bf16/fp32 mismatch, needed
   `accelerator.autocast()`) and a real resume-logic bug shared with ACT
   (see "MolmoAct2 > FSDP2 checkpointing" above for both). Still open,
   because it genuinely needs the real ~8B model and checkpoint (not
   installable here -- disk headroom and GPU memory both too tight): (a) a
   real forward/backward pass, to catch anything the fake policy's
   simplicity couldn't (e.g. the MEAN_STD/`VISUAL` normalization risk --
   see "MolmoAct2 > Normalization" above -- needs a real batch through the
   real processor); (b) DDP's "did not receive grad for all parameters"
   with the REAL model's actual frozen/trainable parameter split (defended
   against with `find_unused_parameters=True`, not confirmed necessary);
   (c) for `--molmoact2-offload-tokenization`, confirm tensor shapes/dtypes
   reaching `NumpyToTorchCollate` match the non-offloaded path -- not
   exercised by the integration test (used an identity fake preprocessor).

5. **`--policy-type pi05` -- the real ~2.3B model has never actually run
   end-to-end**, same reason as MolmoAct2: no verified real pretrained
   checkpoint repo id was found (this project's standing rule is against
   guessing HF repo ids), so `PI05Policy.from_pretrained(...)` was never
   called against real weights. What HAS been verified: `PI05Config`/
   `PI05Policy` field names/API confirmed by direct introspection AND real
   `PI05Config(...)` construction calls for all three training modes (full
   fine-tune, freeze-vision, expert-only); a fake-policy DDP integration test
   (same interface -- `forward`, `get_optim_params` returning a flat
   iterable -- but a tiny model) exercised the real `registry.py`/
   `train_loop.py`/`pi05.py` code including a simulated crash +
   Ray-Train-relaunch, reaching the target step count. Still open, because it
   genuinely needs a real checkpoint download: (a) `from_pretrained` actually
   loading real weights into `PI05Policy`, not just constructing a config;
   (b) a real forward/backward pass, to catch anything the fake policy's
   simplicity couldn't (e.g. the MEAN_STD/`VISUAL` normalization
   simplification -- see "π0.5 > Normalization" above -- needs a real batch
   through the real processor); (c) whether π0.5's actual ~2.3B model fits
   under plain DDP on this pipeline's target hardware -- no real VRAM number
   exists yet, unlike MolmoAct2's README-sourced ~20GB/~60GB figures, which
   is exactly why FSDP2 wasn't built for it (see "π0.5 > DDP only" above);
   (d) whether any train/eval-sensitive layers exist in the submodules
   `--pi05-freeze-vision-encoder`/`--pi05-train-expert-only` leave frozen,
   given `PI05Policy` has no `.train()` override to force them to eval mode.

## Usage

```bash
# 1. Verify decode on one episode from each task you're considering
python -m training.verify_decode --task arrange_the_flowers

# 2. See real task names + episode counts if you haven't already
python -m training.data.discover

# 3a. Prepare data ahead of time (optional but recommended) -- discover,
#     download, convert to LeRobot v3, then stop. No GPU needed.
python -m training.prepare_data --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300

# 3b. Train. If step 3a was already run for these --tasks, this finds the
#     prepared dataset and skips straight to training -- no HF_TOKEN needed.
#     If not, it prepares inline first (same as 3a would have).
python -m training.train --tasks arrange_the_flowers box_folding \
    --max-train-steps 50   # smoke run first
```

Splitting 3a out is worth it once you're iterating on training config/hparams
without touching the data -- each `train.py` run then skips straight past
discover/download/convert. For a genuine one-shot run, skip 3a and just run
3b with `--max-episodes-per-task` set; it'll prepare inline.

Drop `--max-train-steps` for a full run once the smoke run's loss is moving
and a checkpoint lands under `training/runs/<run-name>/`. Pass `--reconvert`
(to either script) to force rebuilding the LeRobot v3 dataset -- e.g. after
changing `--max-episodes-per-task` or camera/tick settings in `config.py` --
instead of reusing what's cached at `--v3-root`.

On a shared/Anyscale cluster, override `--storage-root` to shared storage
(e.g. `/mnt/cluster_storage/act_training`) so a worker restarting on a
different node can still see checkpoints.

## Command reference

Every script's full flag list, run as `python -m training.<module>` from the
repo root. Defaults shown are what you get by omitting the flag.

### `training/verify_decode.py` -- run this first, per task, before anything else

| Flag | Default | Meaning |
|---|---|---|
| `--task` (required) | -- | task-name substring |
| `--n-frames` | `6` | frames per camera to dump as PNGs |
| `--chunk-size` | `100` | action-chunk size used for the padding-fraction sanity check |
| `--out-dir` | `verify_out` | where the dumped PNGs + printed stats go |

```bash
python -m training.verify_decode --task arrange_the_flowers --n-frames 10 --out-dir verify_out/flowers
```

### `training/data/discover.py` -- no flags

Lists real task names + episode counts (from the cached listing, or a fresh
scan on first run). Also writes/refreshes `training/.cache/abc130k_listing.json`.

```bash
python -m training.data.discover
```

### `training/prepare_data.py` -- data prep only, no GPU needed

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | one or more task-name substrings |
| `--max-episodes-per-task` | `300` | cap per task (fewer if a task has less) |
| `--source-uri` | `config.py`'s `DataConfig.source_uri` | `hf://datasets/<repo_id>`, `s3://<bucket>/<prefix>`, or `gs://<bucket>/<prefix>` -- only `hf://` verified against real data |
| `--v3-root` | `training/lerobot_v3/<sorted tasks>` | where the converted dataset lives |
| `--reconvert` | off | wipe and rebuild from scratch, ignoring any existing/incremental data |
| `--refresh-listing` | off | re-scan the full source tree instead of the cached listing |
| `--mode` | `stream` | `stream` (nothing local but the output) or `download` (caches raw MCAP locally first) |
| `--sequential-camera-decode` | off (parallel) | decode a episode's 3 cameras one at a time instead of concurrently -- slower/episode, lower memory, can mean more episodes convert at once |
| `--max-concurrent` | auto (from live CPU/memory) | override the concurrent-episode-conversion bound directly |

```bash
python -m training.prepare_data --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300 --mode stream --max-concurrent 4
```

### `training/train.py` -- prepare (if needed) then train

| Flag | Default | Meaning |
|---|---|---|
| `--tasks` (required) | -- | one or more task-name substrings |
| `--max-episodes-per-task` | `300` | ignored if reusing an already-prepared `--v3-root` |
| `--source-uri` | `config.py`'s `DataConfig.source_uri` | `hf://datasets/<repo_id>`, `s3://<bucket>/<prefix>`, or `gs://<bucket>/<prefix>` -- only `hf://` verified against real data |
| `--run-name` | `act-<tasks>-<timestamp>` | run identifier; also the TensorBoard/checkpoint/history subdirectory name |
| `--storage-root` | `training/runs` | where checkpoints/TensorBoard/history for this run land |
| `--num-epochs` | `1` | full passes over the dataset |
| `--batch-size` | `8` | |
| `--max-train-steps` | none (full run) | cap total steps -- use for a smoke run |
| `--eval-every-steps` | `200` | report/checkpoint/early-stop-check granularity, in steps |
| `--early-stop-patience` | `5` | stop after this many eval windows with no loss improvement; `0` or negative disables it |
| `--save-only-on-improvement` | off (checkpoint every report) | only write a checkpoint when the loss improves -- less write I/O, but a resume after an interruption can lose progress back to the last improvement |
| `--checkpoint-max-to-keep` | `3` | max checkpoints kept on disk -- with the default (every report checkpointed), this is the N best-scoring PLUS the single most recent checkpoint (for resuming), even when it isn't among the N best -- native Ray Train behavior, see the "Checkpointing" section below |
| `--num-workers` | live GPU count | Ray Train DDP workers |
| `--refresh-listing` | off | re-scan the full source tree instead of the cached listing |
| `--v3-root` | `training/lerobot_v3/<sorted tasks>` | where the converted dataset is read from / written to |
| `--reconvert` | off | wipe and rebuild the LeRobot v3 dataset from scratch |
| `--use-existing-v3` | off | trust whatever's at `--v3-root` as-is, skip discover/convert entirely, no `HF_TOKEN` needed |
| `--mode` | `stream` | `stream` or `download` (only relevant if data isn't already prepared) |
| `--sequential-camera-decode` | off (parallel) | see prepare_data.py's flag above |
| `--max-concurrent` | auto | override the concurrent-episode-conversion bound |
| `--policy-type` | `act` | `act`, `molmoact2`, or `pi05` -- see "MolmoAct2"/"π0.5" sections below; no separate environment needed for any of them (see Setup) |

MolmoAct2-only flags (all ignored/unused when `--policy-type act`; `--molmoact2-setup-type`/`--molmoact2-control-mode` are required when `--policy-type molmoact2`):

| Flag | Default | Meaning |
|---|---|---|
| `--molmoact2-checkpoint-path` | `allenai/MolmoAct2` | HF repo id or local path the VLM backbone loads from |
| `--molmoact2-setup-type` | -- (required) | free-text embodiment prompt, e.g. `"dual-arm robot with wrist and top cameras"` |
| `--molmoact2-control-mode` | -- (required) | free-text control-mode prompt, e.g. `"delta joint position"` |
| `--molmoact2-action-mode` | `continuous` | `continuous`, `discrete`, or `both` -- real default is `both`; narrowed here to skip the discrete FAST-tokenizer dependency/setup |
| `--molmoact2-train-mode` | `lora` | `lora` (LoRA on the VLM only, action expert stays fully trainable -- fits under plain DDP, ~20GB/GPU at batch 8), `fft` (full fine-tune, needs `--molmoact2-distributed-strategy fsdp2` to be practical), or `freeze` (VLM frozen, only the action expert trains -- requires `--molmoact2-action-mode continuous`). Translated internally to `MolmoAct2Config`'s real `enable_lora_vlm`/`enable_lora_action_expert`/`train_action_expert_only` fields |
| `--molmoact2-lora-rank` / `-alpha` / `-dropout` | `64` / `16` / `0.05` | `MolmoAct2Config`'s own real defaults (yes, alpha < rank) |
| `--molmoact2-no-gradient-checkpointing` | off (checkpointing on) | disable only if you've confirmed the memory headroom |
| `--molmoact2-vit-lr` / `-connector-lr` / `-action-expert-lr` | `None` -> falls back to `--lr` | MolmoAct2's `get_optim_params()` returns 4 LR groups (vlm/vit/connector/action_expert), not ACT's 2 |
| `--molmoact2-distributed-strategy` | `ddp` | `ddp` (LoRA) or `fsdp2` (full fine-tune, only with `--molmoact2-train-mode fft`) -- see "MolmoAct2 full fine-tuning" below |
| `--molmoact2-fsdp-cpu-offload` | off | trades speed for fitting on fewer/smaller GPUs -- `fsdp2` only |
| `--molmoact2-offload-tokenization` | off | run MolmoAct2's preprocessor as a Ray Data stage instead of inline in the training loop -- see "MolmoAct2 Ray Data preprocessing offload" below |
| `--molmoact2-offload-concurrency` | auto (from live CPU count) | Ray Data actor-pool size for the above |

π0.5-only flags (all ignored/unused when `--policy-type` is not `pi05`; `--pi05-pretrained-path` is required when `--policy-type pi05`):

| Flag | Default | Meaning |
|---|---|---|
| `--pi05-pretrained-path` | -- (required) | HF repo id or local path to a real pretrained π0.5 checkpoint -- no safe default, see "π0.5" section above |
| `--pi05-freeze-vision-encoder` | off | freeze the vision tower; language model + action expert still train |
| `--pi05-train-expert-only` | off | only the action expert trains, everything else frozen |
| `--pi05-no-gradient-checkpointing` | off (checkpointing on) | disable only if you've confirmed the memory headroom |
| `--pi05-empty-cameras` | `0` | pad `input_features` with dummy camera slots -- for when the pretrained checkpoint expects more cameras than this dataset has |

```bash
# Smoke run, streaming, defaults otherwise
python -m training.train --tasks arrange_the_flowers --max-episodes-per-task 20 --max-train-steps 50

# Full run, tuned: download mode, sequential decode for more concurrency,
# explicit concurrency cap, longer patience, non-default storage location
python -m training.train --tasks arrange_the_flowers box_folding \
    --max-episodes-per-task 300 --num-epochs 3 --batch-size 16 \
    --mode download --sequential-camera-decode --max-concurrent 6 \
    --eval-every-steps 100 --early-stop-patience 8 \
    --storage-root /mnt/cluster_storage/act_training --run-name flowers-boxes-v2

# Already have a LeRobot v3 dataset (from anywhere) and just want to train on it
python -m training.train --tasks arrange_the_flowers --v3-root /path/to/existing/dataset --use-existing-v3

# Keep more checkpoints, and only write one when the loss improves (less write
# I/O than the default, at the cost of a resume potentially losing progress
# back to the last improvement)
python -m training.train --tasks arrange_the_flowers --checkpoint-max-to-keep 10 --save-only-on-improvement

# Same dataset mirrored to S3 instead of HF -- no HF_TOKEN needed, uses ambient
# AWS credentials (unverified against a real bucket -- see README's data-source section)
python -m training.train --tasks arrange_the_flowers --source-uri s3://my-bucket/abc-130k-mirror

# MolmoAct2, LoRA on one GPU (same environment as ACT -- see Setup)
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --batch-size 8 --max-train-steps 50

# MolmoAct2, full fine-tune with FSDP2 across multiple GPUs on one node, plus
# the Ray Data preprocessing offload -- see "MolmoAct2" section below before
# using either of these, both carry real unverified risk
python -m training.train --tasks dress_the_teddy_bear --policy-type molmoact2 \
    --molmoact2-setup-type "dual-arm robot with wrist and top cameras" \
    --molmoact2-control-mode "delta joint position" \
    --molmoact2-train-mode fft --molmoact2-distributed-strategy fsdp2 \
    --molmoact2-offload-tokenization --batch-size 32

# pi05, full fine-tune on one GPU (same environment as ACT/MolmoAct2 -- see
# Setup; --pi05-pretrained-path must point at a real checkpoint you found on
# the HF Hub yourself -- see "π0.5" section above)
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --batch-size 8 --max-train-steps 50

# pi05, freeze the vision encoder and only train the action expert -- reduces
# the trainable/optimizer-state footprint without needing FSDP2
python -m training.train --tasks dress_the_teddy_bear --policy-type pi05 \
    --pi05-pretrained-path <your-real-checkpoint-repo-id> \
    --pi05-train-expert-only --batch-size 8
```

### `training/history.py` -- view past runs

| Flag | Default | Meaning |
|---|---|---|
| `--storage-root` | `./runs` | match whatever `--storage-root` your training runs used |
| `--limit` | `20` | most recent N runs to show |

```bash
python -m training.history --storage-root training/runs --limit 50
```

### What a run produces, all under `<storage_root>/<run_name>/`

- `tensorboard/` -- live loss/l1/kl curves + learning rates:
  `tensorboard --logdir <path>` (printed at the end of every run).
- `ray_data_stats.txt` -- per-operator Ray Data execution stats for what
  that run's shard actually consumed (throughput, block sizes, remote/UDF time).
- Checkpoints -- by default, written on every report and pruned to the
  `--checkpoint-max-to-keep` (default 3) lowest-loss ones, PLUS the single
  most recently written checkpoint even when it isn't among those N -- that
  extra one is what a resumed run (`ray.train.get_checkpoint()`, triggered
  automatically on a worker failure/restart within the same run) picks up
  from, so it's never more than one `--eval-every-steps` window stale
  relative to wherever training actually stopped. Confirmed by reading the
  installed `ray.train.v2` checkpoint manager's pruning logic (it excludes
  `self._latest_checkpoint_result` from its deletion set), not assumed.
  `--save-only-on-improvement` switches to the old behavior instead: a
  checkpoint only written when the loss improves (both window- and
  epoch-level reports share one running best) -- less write I/O, but no
  separate always-fresh checkpoint, so a resume falls back to whichever
  improving checkpoint happened most recently.
- One line appended to `<storage_root>/history.jsonl` per run (success or
  failure) -- view with `python -m training.history --storage-root <path>`.

## Cost/scale note

Streaming means `--max-episodes-per-task 300` no longer costs ~50GB of local
MCAP storage (raw MCAP averages ~173MB/episode) -- that network transfer
still happens, just without landing on disk first. Per-episode wall time is
network (streaming + parse, ~15s) plus CPU-bound video decode (~50-55s per
camera, run concurrently across cameras -- ~70s wall for 3 -- see "Streaming
conversion" above), times however many episodes run concurrently across each
other (`max_concurrent`, printed at the start of a conversion run, derived
from live CPU/memory). Start with `--max-episodes-per-task 20` and
`--max-train-steps 50` to validate the whole pipeline cheaply before
committing to a larger conversion.
