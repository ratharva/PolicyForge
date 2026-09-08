# 2. Verify decode

Before converting or training on anything, sanity-check that one real
episode decodes correctly. This catches the kind of bug that's expensive to
discover after a full conversion or a training run: a wrong field mapping,
a misaligned video decode, a schema that doesn't match the real data.

`training/data_prep/verify_decode.py` is **mcap-ingestion-strategy only**
today (it errors out clearly for `--dataset-source droid`/`agibot_alpha`,
which don't decode raw episodes this pipeline's own way -- see
[Customizing datasets](customizing-datasets.md)).

```bash
export HF_TOKEN=$(cat hf_tok.txt)
python -m training.data_prep.verify_decode --task arrange_the_flowers
```

Downloads one episode, decodes + aligns it, prints state/action ranges and
the action-chunk padding fraction, and writes a handful of frames per camera
as PNGs for visual inspection.

| Flag | Default | Meaning |
|---|---|---|
| `--task` (required) | -- | task-name substring |
| `--dataset-source` | `abc130k` | which schema to verify decode against -- must use the mcap strategy |
| `--n-frames` | `6` | frames per camera to dump as PNGs |
| `--chunk-size` | `100` | action-chunk size used for the padding-fraction sanity check |
| `--out-dir` | `verify_out` | where the dumped PNGs + printed stats go |

```bash
python -m training.data_prep.verify_decode --task arrange_the_flowers --n-frames 10 --out-dir verify_out/flowers
```

**Look at the dumped PNGs**: frames should be sharp (not garbled/green-block
artifacts from a misaligned decode) and change smoothly across the sampled
indices (not jump around, which would indicate the FIFO frame<->timestamp
pairing in `training/data_prep/strategies/mcap.py`/`video_decode.py` is
wrong for this stream -- see "Video decode" in
[Verification status](verification-status.md#video-decode-during-conversion)
for exactly what's unverified there and why).

Also check the printed ranges: gripper dims should sit in `[0, 1]`, arm dims
in a plausible radian range. If the padding fraction printed for the action
chunk is over 50%, your `--chunk-size` is probably too large for this
episode's length.

## Why convert to LeRobot v3 instead of training directly off the raw source

Decoding MCAP (protobuf parse + floor-alignment + raw H.264/H.265
elementary-stream decode) is real work that a direct-MCAP pipeline pays on
**every** training run. Converting once instead:
- pays that cost a single time, cached by task set under `training/lerobot_v3/`
- lets training reuse `training/vendor/lerobot_datasource.py` -- a proven,
  partitioned, memory-disciplined streaming reader -- instead of a bespoke
  `flat_map`
- writes real mp4 containers (not raw elementary streams), so the read side
  is the same well-tested `av.open(path)` container path, not the riskier
  raw-stream decode conversion itself still has to do once

This is a net win once you'll train more than once or twice on the same
episode set. For a single one-shot smoke test, converting first is strictly
more work, not less -- `build_dataset_direct()` in `training/data/ray_dataset.py`
is kept around for exactly that case (mcap-only, not wired into `train.py`
by default).

## Next

[3. Prepare data](03-prepare-data.md)
