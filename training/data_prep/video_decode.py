"""Decode a raw H.264/H.265 elementary bytestream into RGB frames.

THIS IS THE LEAST-VERIFIED PART OF THE PIPELINE. ABC-130k's camera topics
deliver compressed video as a raw elementary stream chunked across many mcap
messages -- NOT a container file like the mp4s
training/vendor/lerobot_datasource.py reads via av.open(path). There is no
container header here, so decode goes through PyAV's CodecContext.parse/decode
on the concatenated raw bytes instead.

Known simplification: frame timestamps are assigned by pairing decode order
with message publish order (FIFO). This holds only if the stream has no
B-frames (decode order == presentation order) -- common for low-latency
robotics streaming but NOT confirmed for this dataset. Verify with
`python -m training.verify_decode` on a real episode before trusting this
for a training run.
"""
from __future__ import annotations

import av
import numpy as np

_CODEC_NAMES = {"h264": "h264", "avc": "h264", "h265": "hevc", "hevc": "hevc"}


def decode_camera_stream(
    messages: list[tuple[int, bytes, str]], image_size: tuple[int, int]
) -> list[tuple[int, np.ndarray]]:
    """messages: [(log_time_ns, raw_bytes, format), ...], any order.
    Returns [(log_time_ns, HWC uint8 RGB frame), ...] in decode order."""
    if not messages:
        return []
    messages = sorted(messages, key=lambda m: m[0])
    fmt = messages[0][2].lower()
    codec_name = _CODEC_NAMES.get(fmt)
    if codec_name is None:
        raise ValueError(f"unrecognized video format {fmt!r} -- add it to _CODEC_NAMES")

    codec = av.CodecContext.create(codec_name, "r")
    h, w = image_size
    out: list[tuple[int, np.ndarray]] = []
    pending_times: list[int] = []

    def _drain(packets) -> None:
        for packet in packets:
            for frame in codec.decode(packet):
                arr = frame.to_ndarray(format="rgb24")
                if arr.shape[:2] != (h, w):
                    arr = _resize(arr, h, w)
                ts = pending_times.pop(0) if pending_times else messages[-1][0]
                out.append((ts, arr))

    for log_time, raw_bytes, _fmt in messages:
        pending_times.append(log_time)
        _drain(codec.parse(raw_bytes))
    _drain(codec.parse(b""))  # flush the parser's internal buffer
    for frame in codec.decode(None):  # flush the decoder
        arr = frame.to_ndarray(format="rgb24")
        if arr.shape[:2] != (h, w):
            arr = _resize(arr, h, w)
        ts = pending_times.pop(0) if pending_times else messages[-1][0]
        out.append((ts, arr))

    return out


def _resize(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
    frame = frame.reformat(width=w, height=h)
    return frame.to_ndarray(format="rgb24")
