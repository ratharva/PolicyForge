"""The "mcap" ingestion strategy: decode MCAP files with protobuf-encoded
messages into per-tick aligned arrays, driven entirely by a schema file's
`mcap:` block (training/data_prep/schemas/*.yaml) instead of a hand-written
RobotSchema subclass. Any future MCAP-based dataset (a different robot,
different topic names) is just a new schema file naming this strategy --
no new Python here.

Camera topics are returned by _extract() as raw bytes + codec only; actual
video decode happens in training/data_prep/video_decode.py (reused
unchanged -- it never depended on RobotSchema).
"""
from __future__ import annotations

import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from mcap.reader import McapReader, make_reader
from mcap_protobuf.decoder import DecoderFactory

from training.common.robots import RobotSchema
from training.data_prep.align import build_tick_grid, floor_align
from training.data_prep.config import ConvertConfig
from training.data_prep.video_decode import decode_camera_stream


@dataclass
class McapIngestionConfig:
    # Field names within a decoded protobuf message.
    state_value_field: str
    camera_data_field: str
    camera_format_field: str
    # Only one "top" feed is kept even if a station publishes more than one
    # top-camera topic -- locked to whichever candidate is seen first in an
    # episode (see _extract()).
    top_camera_candidates: tuple[str, ...]
    wrist_camera_topics: dict[str, str]
    # Episode-relative path -> (split, task): "<path_prefix>/{split}/{task}/.../<episode_filename>"
    path_prefix: str
    episode_filename: str


def build_ingestion_config(spec: dict) -> McapIngestionConfig:
    m = spec["mcap"]
    return McapIngestionConfig(
        state_value_field=m["state_value_field"],
        camera_data_field=m["camera_data_field"],
        camera_format_field=m["camera_format_field"],
        top_camera_candidates=tuple(m["top_camera_candidates"]),
        wrist_camera_topics=dict(m["wrist_camera_topics"]),
        path_prefix=m["path_prefix"],
        episode_filename=m["episode_filename"],
    )


def path_to_split_and_task(path: str, mcap_cfg: McapIngestionConfig) -> tuple[str, str]:
    """"<path_prefix>/{split}/{task}/episode_{uuid}/<episode_filename>" -> (split, task)."""
    parts = path.split("/")
    if len(parts) >= 4 and parts[0] == mcap_cfg.path_prefix:
        return parts[1], parts[2]
    return "unknown", "unknown"


def episode_filter(path: str, mcap_cfg: McapIngestionConfig) -> bool:
    return path.endswith(mcap_cfg.episode_filename)


@dataclass
class RawEpisode:
    # topic -> [(log_time_ns, value_array), ...]
    state: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    action: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    # camera_key -> [(log_time_ns, raw_compressed_bytes, format_str), ...]
    cameras: dict[str, list[tuple[int, bytes, str]]] = field(default_factory=dict)


def _extract(reader: McapReader, robot: RobotSchema, mcap_cfg: McapIngestionConfig) -> RawEpisode:
    state_topics = {t for t, _ in robot.state_components}
    action_topics = {t for t, _ in robot.action_components}
    wrist_topic_to_key = {v: k for k, v in mcap_cfg.wrist_camera_topics.items()}
    top_topics = set(mcap_cfg.top_camera_candidates)

    state: dict[str, list] = defaultdict(list)
    action: dict[str, list] = defaultdict(list)
    cameras: dict[str, list] = defaultdict(list)

    wanted = state_topics | action_topics | top_topics | set(wrist_topic_to_key)
    chosen_top_topic: str | None = None  # locked to whichever top topic is seen first

    for schema, channel, message, decoded in reader.iter_decoded_messages(topics=wanted):
        topic = channel.topic
        if topic in state_topics:
            value = getattr(decoded, mcap_cfg.state_value_field)
            state[topic].append((message.log_time, np.asarray(value, dtype=np.float64)))
        elif topic in action_topics:
            value = getattr(decoded, mcap_cfg.state_value_field)
            action[topic].append((message.log_time, np.asarray(value, dtype=np.float64)))
        elif topic in top_topics:
            # Lock to the first "top" topic seen so two cameras' frames
            # never mix into one "top" stream.
            if chosen_top_topic is None:
                chosen_top_topic = topic
            elif topic != chosen_top_topic:
                continue
            cam_data = getattr(decoded, mcap_cfg.camera_data_field)
            cam_format = getattr(decoded, mcap_cfg.camera_format_field)
            cameras["top"].append((message.log_time, cam_data, cam_format))
        elif topic in wrist_topic_to_key:
            key = wrist_topic_to_key[topic]
            cam_data = getattr(decoded, mcap_cfg.camera_data_field)
            cam_format = getattr(decoded, mcap_cfg.camera_format_field)
            cameras[key].append((message.log_time, cam_data, cam_format))

    return RawEpisode(state=dict(state), action=dict(action), cameras=dict(cameras))


def read_raw_episode(mcap_path: str, robot: RobotSchema, mcap_cfg: McapIngestionConfig) -> RawEpisode:
    """Local file; decode_and_align() uses the streaming version below by default."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        return _extract(reader, robot, mcap_cfg)


def read_raw_episode_streaming(
    source_uri: str, rel_path: str, robot: RobotSchema, mcap_cfg: McapIngestionConfig,
    token: str | None = None,
) -> RawEpisode:
    """Streams an episode.mcap directly from source_uri's backend -- no
    local file is ever written. token is only meaningful for hf:// sources;
    s3://gs:// rely on ambient credentials (see source.py's open_fs()).

    Forces NonSeekingReader explicitly rather than make_reader()'s
    auto-detection: some fsspec backends' seekable() reports True even
    though seek() actually raises, which would make auto-detection pick the
    seeking reader and crash. Every topic is read start to end regardless,
    which is exactly NonSeekingReader's access pattern anyway.
    """
    from mcap.reader import NonSeekingReader

    from training.data_prep.source import open_fs

    fs, fs_root = open_fs(source_uri, hf_token=token)
    full_path = f"{fs_root}/{rel_path}"
    with fs.open(full_path, mode="rb", block_size=0, cache_type="none") as f:
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
        return _extract(reader, robot, mcap_cfg)


def align_episode(raw: RawEpisode, robot: RobotSchema) -> dict[str, np.ndarray]:
    all_times: list[int] = []
    for topic_msgs in list(raw.state.values()) + list(raw.action.values()):
        all_times.extend(t for t, _ in topic_msgs)
    for cam_frames in raw.cameras.values():
        all_times.extend(t for t, _ in cam_frames)
    if not all_times:
        raise ValueError("episode has no messages on any expected topic")

    start_ns, end_ns = min(all_times), max(all_times)
    ticks = build_tick_grid(start_ns, end_ns, robot.tick_fps)
    if len(ticks) == 0:
        raise ValueError("episode too short to produce even one tick")

    def aligned_concat(components: tuple, source: dict) -> np.ndarray:
        parts = []
        for topic, _dim in components:
            msgs = source.get(topic)
            if not msgs:
                raise ValueError(f"expected topic {topic!r} missing from this episode")
            msgs = sorted(msgs, key=lambda m: m[0])
            times = np.array([t for t, _ in msgs], dtype=np.int64)
            values = np.stack([v for _, v in msgs])
            parts.append(floor_align(times, values, ticks))
        return np.concatenate(parts, axis=1)

    state = aligned_concat(robot.state_components, raw.state)
    action = aligned_concat(robot.action_components, raw.action)

    images: dict[str, np.ndarray] = {}
    for cam_key in robot.camera_keys:
        frames = raw.cameras.get(cam_key)
        if not frames:
            raise ValueError(f"episode is missing camera {cam_key!r} (station variant mismatch?)")
        times = np.array([t for t, _ in frames], dtype=np.int64)
        stack = np.stack([f for _, f in frames])  # (n, H, W, 3) uint8
        idx = np.searchsorted(times, ticks, side="right") - 1
        idx = np.clip(idx, 0, len(times) - 1)
        images[cam_key] = stack[idx]

    result: dict[str, np.ndarray] = {"state": state, "action": action}
    for k, v in images.items():
        result[f"image.{k}"] = v
    return result


def decode_and_align(
    source_uri: str, rel_path: str, robot: RobotSchema, mcap_cfg: McapIngestionConfig,
    token: str | None, convert_cfg: ConvertConfig, image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """The decode_and_align_fn convert.py's _convert_one() calls -- reads
    the raw episode, decodes each camera's compressed frames, and floor-
    aligns everything onto one tick grid. Exactly what _convert_one used to
    do inline before this was pulled out into an injectable per-strategy
    function (see training/data_prep/convert.py)."""
    if convert_cfg.mode == "download":
        from training.data_prep.source import download_one

        local_path = download_one(source_uri, rel_path, token)
        raw = read_raw_episode(local_path, robot, mcap_cfg)
    else:
        raw = read_raw_episode_streaming(source_uri, rel_path, robot, mcap_cfg, token)

    if convert_cfg.parallel_camera_decode:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(raw.cameras) or 1) as pool:
            futures = {
                cam_key: pool.submit(decode_camera_stream, raw.cameras[cam_key], image_size)
                for cam_key in list(raw.cameras)
            }
            for cam_key, fut in futures.items():
                raw.cameras[cam_key] = fut.result()
    else:
        for cam_key in list(raw.cameras):
            raw.cameras[cam_key] = decode_camera_stream(raw.cameras[cam_key], image_size)

    return align_episode(raw, robot)
