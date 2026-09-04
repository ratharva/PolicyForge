# Adding a new dataset, robot, or policy

A condensed, procedural companion to [Customizing datasets](customizing-datasets.md)
and [Customizing policies](customizing-policies.md) -- those two have the
full narrative/verification history; this one is just "what do I actually
touch." Read the relevant deep-dive too before writing real code -- this
page skips the *why*.

## 1. Add a new dataset

**If it reuses an existing ingestion strategy (`mcap`, `hf_lerobot_mirror`,
`agibot_hdf5`) -- the common case:**

1. Copy an existing schema file that uses the same strategy (e.g.
   `training/data_prep/schemas/abc130k.yaml` for another MCAP dataset) to
   `training/data_prep/schemas/<your_dataset>.yaml`.
2. Set the top-level fields for real, from the real raw data -- **download
   and inspect it yourself first**, don't trust a dataset card or a fetched
   summary (see the droid correction in
   [Customizing datasets](customizing-datasets.md#droid----droid-franka-panda-18-institution-consortium)
   for why that's burned this project before):
   ```yaml
   dataset_source: your_dataset        # must match the filename
   ingestion_strategy: mcap            # or hf_lerobot_mirror / agibot_hdf5
   default_source_uri: "hf://datasets/org/repo"
   tick_fps: 30.0
   camera_keys: [top, left_wrist, right_wrist]
   state_components:
     - ["component_name", 6]
     - ["another_component", 1]
   action_components:
     - ["component_name", 6]
   ```
3. Fill in the strategy-named block (`mcap:` / `hf_lerobot_mirror:` /
   `agibot_hdf5:`) -- field meanings are strategy-specific, see
   [Customizing datasets](customizing-datasets.md#the-schema-yaml-format)
   for each block's real shape.
4. Nothing else to register -- `--dataset-source` choices are *discovered*
   from `training/data_prep/schemas/*.yaml`
   (`schema_loader.available_dataset_sources()`), not a hardcoded list.
5. Verify against real data before trusting it:
   ```bash
   python -m training.prepare_data --tasks <a_real_task_substring> \
       --max-episodes-per-task 2 --dataset-source your_dataset \
       --v3-root /tmp/your_dataset_smoke_test
   ```

**If it needs a genuinely new raw ingestion format** (not MCAP+protobuf, not
an HF LeRobot mirror, not HDF5+tar) -- this is the one case that needs new
Python, not just YAML:

1. Add `training/data_prep/strategies/<name>.py` with:
   - `build_ingestion_config(spec: dict) -> YourIngestionConfig` -- parses
     the schema's `<name>:` block.
   - Either `decode_and_align(source_uri, rel_path, robot, ingestion_cfg,
     token, convert_cfg, image_size) -> dict[str, np.ndarray]` if this
     strategy can use the shared `convert_to_lerobot_v3` Ray-parallel
     per-episode orchestration (see `strategies/mcap.py`) -- each episode
     must be independently, cheaply fetchable for this to fit.
   - Or your own `prepare(...)` entirely if not (see
     `strategies/hf_lerobot_mirror.py`'s download+migrate, or
     `strategies/agibot_hdf5.py`'s own discover -> download shards ->
     scan -> transform -> write orchestration, needed there because many
     episodes share one un-indexed tar).
2. Register it in `training/data_prep/strategies/registry.py`'s
   `_STRATEGY_MODULES` dict (the one place a genuinely new strategy needs a
   code edit -- everything else dispatches through it).
3. Write a schema file naming your new `ingestion_strategy`.

`training/data_prep/lerobot_v3_writer.py` (the write path) needs **zero**
changes either way -- it only ever consumes the generic
`{"state": (T, state_dim), "action": (T, action_dim), "image.<camera_key>":
(T, H, W, 3) uint8}` shape, regardless of what produced it.

## 2. Add a new robot

There is no separate "robot" registry in this codebase, decoupled from a
dataset -- `RobotSchema` (`training/common/robots.py`: `camera_keys`,
`state_components`, `action_components`, `tick_fps`) is **always** derived
from one dataset's schema YAML by `schema_loader.build_robot_schema()`,
never hand-subclassed, and not shared/reused across multiple YAML files.

So mechanically, "adding a new robot" **is** "add a new dataset" (section 1
above) -- the fields that actually describe the robot are a subset of that
same YAML:

| Field | What it means for the robot |
|---|---|
| `camera_keys` | fixed-arity logical camera names this robot exposes |
| `state_components` | `(component name, dim)` pairs, concatenated in order into the state vector |
| `action_components` | same, for the action vector |
| `tick_fps` | the common rate every episode is resampled/aligned to |

"Component name" meaning is strategy-specific, not a robot-level concept:
an MCAP topic string (`mcap`), an HDF5 group path suffix (`agibot_hdf5`), or
a LeRobot column name (`hf_lerobot_mirror`) -- so the same physical robot
ingested two different ways would still need two schema files, one per
strategy, because the component names differ even if the physical
state/action layout doesn't.

Two real sub-cases:

- **Same message/file shape as an existing dataset, different robot** (a
  different MCAP-based robot with the same protobuf shape and a
  data+format-style camera message) -- copy the closest existing schema
  YAML, change `camera_keys`/`state_components`/`action_components`/topic
  names/`tick_fps` to the new robot's real values. No Python changes.
- **Genuinely different message/file shape** (not protobuf, e.g. ROS
  `sensor_msgs`, or a different HDF5 layout than `agibot_hdf5` assumes) --
  this is section 1's "new ingestion strategy" case, not a schema-only
  change. `RobotSchema`'s fields describe *what* the robot's data looks
  like generically; a shape difference needs new decode logic
  (`strategies/mcap.py`'s `_extract()`, or an entirely new strategy
  module), not new schema fields.

As with datasets: get every field from the real raw data (a real decoded
message, a real HDF5 file, a real `meta/info.json`), not assumed --
verified twice over in this project already (the AgiBot HDF5-structure
confirmation, the droid `meta/info.json` correction).

## 3. Add a new policy

Full detail: [Customizing policies](customizing-policies.md#adding-a-new-policy)
and CLAUDE.md's "How to add a new policy" section (verified gotchas, the
`get_optim_params()` LR-wiring trap, the fake-policy test pattern). Short
version:

1. **`training/model/<policy>.py`** implementing the adapter contract
   (mirror `act.py` first):
   - `build_policy_and_processor(data_cfg, overrides, train_cfg,
     dataset_stats, device) -> (policy, preprocessor)`
   - `forward_loss(policy, inputs) -> (loss: Tensor, metrics: dict[str, float])`
   - Optional: `post_build_hook(policy, overrides)` (e.g. gradient
     checkpointing)
   - Optional, only if the policy needs more than plain DDP:
     `wrap_for_training(policy, optimizer, overrides, device) -> (policy,
     optimizer, dist_ctx)`, `save_checkpoint(...)`, `load_checkpoint(...)`
     -- see `model/molmoact2.py`'s FSDP2 path for the real shape.
2. **New `<Policy>ConfigOverrides` dataclass** in `training/config.py`.
   Must include `chunk_size`/`n_action_steps` (the shared v3 dataset reads
   these off `run_cfg.model` generically for action chunking). If
   `get_optim_params()` returns per-component LR groups, add whatever LR
   fields it needs and **wire them from `train_cfg` in `build_*_config`** --
   easy to miss, and silently no-ops if skipped (this exact bug shipped for
   ACT's `--lr-backbone` before it was caught). Check the real shape with
   `inspect.getsource(YourPolicy.get_optim_params)` first -- don't assume
   grouped vs. flat.
3. **Lazy factory** in `training/model/registry.py`: a
   `_<policy>_adapter()` returning a `PolicyAdapter`, plus a branch in
   `get_adapter()`. Import the new module **inside** the factory function,
   not at module top level -- lets environments without this policy's
   deps skip it entirely and fail fast if wrongly selected.
4. **CLI flags** in `train.py`, `--<policy>-*` prefix convention, plus
   post-parse validation for anything conditionally required.
5. **`needs_task=True`** if the policy is language-conditioned -- `task`
   already flows through `LeRobotDatasource` into every batch by default.
6. **Normalization**: default to `MEAN_STD` (matches
   `training/data/stats.py`'s `compute_dataset_stats`) unless the policy's
   own defaults genuinely need something else.
7. **Check the environment before assuming a new one is needed**:
   ```bash
   python -c "import lerobot.policies as p; import pkgutil; \
       print([m.name for m in pkgutil.iter_modules(p.__path__)])"
   python -c "from lerobot.policies.<new_policy>.configuration_<new_policy> import <Config>"
   ```
   Most lerobot-shipped policies need nothing extra. Only reach for a
   separate environment if a real, currently-installed check shows a
   genuine conflict -- not because an external fork's README says so.
8. **Test with a fake-policy integration test before the real model**
   (monkeypatch only `build_policy_and_processor`, passed into the Ray
   Train worker function -- not patched driver-side, workers get a fresh
   import). Catches dtype mismatches, wrap-order bugs, checkpoint dispatch
   bugs without needing real weights or a big GPU.
9. **Update `training/README.md`** (command reference table) and
   `CLAUDE.md` if the addition changes any pattern above.

`train_loop.py` (data iteration, DDP/FSDP wrap, checkpoint/resume, early
stopping, TensorBoard) needs **no changes** -- it's ~90% identical across
every policy by design; only model construction and the forward/loss shape
actually differ.
