"""Small non-algorithmic file helpers shared by calibration workflows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_raw_float32(path: str | Path, shape: tuple[int, int, int],
                     dtype: str = "<f4") -> np.memmap:
    path = Path(path)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected:
        raise ValueError(f"RAW size mismatch: expected {expected}, got {path.stat().st_size}")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def save_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_array(path: str | Path, value: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))
