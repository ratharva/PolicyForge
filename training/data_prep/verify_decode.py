"""Sanity-check the mcap decode pipeline on one real episode before trusting
it for a training run. Downloads one episode, decodes + aligns it, prints
state/action ranges and the action-chunk padding fraction, and writes a
handful of frames per camera as PNGs for visual inspection.

Only meaningful for a --dataset-source whose ingestion_strategy is "mcap"
today -- errors out clearly for any other strategy rather than pretending
to verify something it doesn't know how to decode.

Usage (run from the repo root):
    export HF_TOKEN=hf_...
    python -m training.data_prep.verify_decode --task arrange_the_flowers
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

from training.data.action_chunk import chunk_actions
from training.data_prep.discover import download_episodes, list_episodes_by_task, select_episodes
from training.data_prep.prepare import require_token
from training.data_prep.strategies.mcap import align_episode, path_to_split_and_task, read_raw_episode
from training.data_prep.strategies.registry import get_dataset_source
from training.data_prep.video_decode import decode_camera_stream


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task-name substring")
    parser.add_argument("--dataset-source", required=True)
    parser.add_argument("--n-frames", type=int, default=6, help="frames per camera to dump")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--out-dir", default="verify_out")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    args = parser.parse_args()

    source = get_dataset_source(args.dataset_source)
    if source.ingestion_strategy != "mcap":
        sys.exit(
            f"--dataset-source {args.dataset_source!r} uses the {source.ingestion_strategy!r} "
            f"ingestion strategy -- this tool only verifies mcap decode."
        )
    robot, mcap_cfg = source.robot, source.ingestion_config
    image_size = tuple(args.image_size)

    token = require_token(source.default_source_uri)
    by_task = list_episodes_by_task(
        token, source.default_source_uri, args.dataset_source,
        path_to_split_and_task=lambda p: path_to_split_and_task(p, mcap_cfg),
        episode_filename=mcap_cfg.episode_filename,
    )
    selected = select_episodes(by_task, [args.task], 1)
    local = download_episodes(token, source.default_source_uri, selected)
    task, paths = next(iter(local.items()))
    mcap_path = paths[0]
    print(f"\ndecoding {mcap_path}")

    raw = read_raw_episode(mcap_path, robot, mcap_cfg)
    for k, msgs in raw.state.items():
        print(f"  state  {k:20s} {len(msgs):6d} msgs  sample={msgs[0][1]}")
    for k, msgs in raw.action.items():
        print(f"  action {k:20s} {len(msgs):6d} msgs  sample={msgs[0][1]}")
    for k, frames in list(raw.cameras.items()):
        print(f"  camera {k:20s} {len(frames):6d} msgs  format={frames[0][2]}")
        raw.cameras[k] = decode_camera_stream(frames, image_size)
        print(f"           -> decoded {len(raw.cameras[k])} frames")

    aligned = align_episode(raw, robot)
    t = aligned["state"].shape[0]
    print(f"\naligned to {t} ticks at {robot.tick_fps} fps ({t / robot.tick_fps:.1f}s)")
    print(f"state  min={aligned['state'].min(axis=0)}")
    print(f"state  max={aligned['state'].max(axis=0)}")
    print(f"action min={aligned['action'].min(axis=0)}")
    print(f"action max={aligned['action'].max(axis=0)}")
    print("  (gripper dims should sit in [0, 1]; arm dims in a plausible radian range)")

    action_chunks, is_pad = chunk_actions(aligned["action"], args.chunk_size)
    pad_frac = is_pad.mean()
    print(f"\nchunked actions: shape={action_chunks.shape}  pad_fraction={pad_frac:.3f}")
    if pad_frac > 0.5:
        print(
            f"  WARNING: over half of chunk positions are padding -- chunk_size="
            f"{args.chunk_size} may be too large for this episode's length ({t} ticks)."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    idxs = np.linspace(0, t - 1, args.n_frames, dtype=int)
    written = 0
    for cam_key in robot.camera_keys:
        for i in idxs:
            img = aligned[f"image.{cam_key}"][i]
            Image.fromarray(img).save(os.path.join(args.out_dir, f"{cam_key}_frame{i:05d}.png"))
            written += 1
    print(f"\nwrote {written} frames to {args.out_dir}/")
    print(
        "LOOK AT THEM: frames should be sharp (not garbled/green-block artifacts from a\n"
        "misaligned decode) and change smoothly across the sampled indices (not jump\n"
        "around, which would indicate the FIFO frame<->timestamp pairing in\n"
        "video_decode.py is wrong for this stream). Only trust `train.py` on this data\n"
        "after this looks right."
    )


if __name__ == "__main__":
    main()
