"""Decode one episode.mcap file into raw per-topic message streams.

read_raw_episode() reads a local file; read_raw_episode_streaming() streams
directly from DataConfig.source_uri (HF Hub, S3, or GCS) with no local file
ever written -- what convert.py uses by default. Camera topics are returned
here as raw bytes + codec only; actual video decode happens in
video_decode.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from mcap.reader import McapReader, make_reader
from mcap_protobuf.decoder import DecoderFactory

from training.config import DataConfig
from training.robots import RobotSchema


@dataclass
class RawEpisode:
    # topic -> [(log_time_ns, value_array), ...]
    state: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    action: dict[str, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    # camera_key -> [(log_time_ns, raw_compressed_bytes, format_str), ...]
    cameras: dict[str, list[tuple[int, bytes, str]]] = field(default_factory=dict)


def _extract(reader: McapReader, robot: RobotSchema) -> RawEpisode:
    state_topics = {t for t, _ in robot.state_topics}
    action_topics = {t for t, _ in robot.action_topics}
    wrist_topic_to_key = {v: k for k, v in robot.wrist_camera_topics.items()}
    top_topics = set(robot.top_camera_candidates)

    state: dict[str, list] = defaultdict(list)
    action: dict[str, list] = defaultdict(list)
    cameras: dict[str, list] = defaultdict(list)

    wanted = state_topics | action_topics | top_topics | set(wrist_topic_to_key)
    chosen_top_topic: str | None = None  # locked to whichever top topic is seen first

    for schema, channel, message, decoded in reader.iter_decoded_messages(topics=wanted):
        topic = channel.topic
        if topic in state_topics:
            value = getattr(decoded, robot.state_value_field)
            state[topic].append((message.log_time, np.asarray(value, dtype=np.float64)))
        elif topic in action_topics:
            value = getattr(decoded, robot.state_value_field)
            action[topic].append((message.log_time, np.asarray(value, dtype=np.float64)))
        elif topic in top_topics:
            # Lock to the first "top" topic seen so two cameras' frames
            # never mix into one "top" stream.
            if chosen_top_topic is None:
                chosen_top_topic = topic
            elif topic != chosen_top_topic:
                continue
            cam_data = getattr(decoded, robot.camera_data_field)
            cam_format = getattr(decoded, robot.camera_format_field)
            cameras["top"].append((message.log_time, cam_data, cam_format))
        elif topic in wrist_topic_to_key:
            key = wrist_topic_to_key[topic]
            cam_data = getattr(decoded, robot.camera_data_field)
            cam_format = getattr(decoded, robot.camera_format_field)
            cameras[key].append((message.log_time, cam_data, cam_format))

    return RawEpisode(state=dict(state), action=dict(action), cameras=dict(cameras))


def read_raw_episode(mcap_path: str, cfg: DataConfig) -> RawEpisode:
    """Local file; convert.py uses the streaming version below instead."""
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        return _extract(reader, cfg.robot)


def read_raw_episode_streaming(
    source_uri: str, rel_path: str, cfg: DataConfig, token: str | None = None,
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

    from training.data.source import open_fs

    fs, fs_root = open_fs(source_uri, hf_token=token)
    full_path = f"{fs_root}/{rel_path}"
    with fs.open(full_path, mode="rb", block_size=0, cache_type="none") as f:
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()])
        return _extract(reader, cfg.robot)
