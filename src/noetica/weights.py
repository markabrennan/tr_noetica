"""Minimal safetensors reader: JSON header + raw bytes -> numpy fp32 arrays."""

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """bf16 is the top 16 bits of an fp32: widen and shift left."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """Load every tensor in a .safetensors file as a contiguous float32 array."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    (header_len,) = struct.unpack("<Q", bytes(data[:8]))
    header = json.loads(bytes(data[8 : 8 + header_len]))
    base = 8 + header_len
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[base + start : base + end].view(_DTYPES[info["dtype"]])
        if info["dtype"] == "BF16":
            arr = bf16_to_f32(raw)
        else:
            arr = raw.astype(np.float32)
        out[name] = np.ascontiguousarray(arr.reshape(info["shape"]))
    return out
