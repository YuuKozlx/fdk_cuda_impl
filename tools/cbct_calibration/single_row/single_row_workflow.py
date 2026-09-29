"""Single-row bead calibration as a readable Python API.

The workflow is deliberately split into two layers:

1. preprocessing: RAW -> circular target detection -> permanent-ID tracking;
2. calibration: indexed tracks -> ellipse fits -> fixed CBCT geometry.

See ``SINGLE_ROW_GUIDE.zh-CN.md`` and ``single_row_example.py``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

# 允许从本目录直接点击运行示例，同时复用上一级公共算法模块。
COMMON_DIRECTORY = Path(__file__).resolve().parent.parent
if str(COMMON_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(COMMON_DIRECTORY))

from calibrate import calibrate
from detect import detect_and_track


@dataclass(frozen=True)
class SingleRowConfig:
    """All values needed by the single-row workflow."""
    raw_path: Path
    output_directory: Path
    views: int
    bead_count: int
    spacing_mm: float
    rows: int = 1024
    cols: int = 1024
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    threshold: float = 100.0
    radius_range_px: tuple[float, float] = (2.0, 20.0)
    min_circularity: float = 0.35
    border_margin_px: int = 0
    max_jump_px: float = 60.0
    min_row_span_px: float = 30.0
    min_axis_ratio: float = 0.02
    min_pair_index_gap: int = 1
    max_rotation_iterations: int = 1
    rotation_tolerance_px: float = 1e-4
    refine: bool = False


def load_raw(config: SingleRowConfig) -> np.ndarray:
    dtype = np.dtype("<f4")
    expected = config.views * config.rows * config.cols * dtype.itemsize
    actual = config.raw_path.stat().st_size
    if actual != expected:
        raise ValueError(f"RAW size mismatch: expected {expected} bytes, got {actual}")
    return np.memmap(config.raw_path, dtype=dtype, mode="r",
                     shape=(config.views, config.rows, config.cols))


def make_distance_matrix(bead_count: int, spacing_mm: float) -> np.ndarray:
    """Known distances for equally spaced collinear beads."""
    ids = np.arange(bead_count, dtype=float)
    return spacing_mm * np.abs(ids[:, None] - ids[None, :])


def _jsonable(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer)): return value.item()
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    return value


def preprocess_single_row(config: SingleRowConfig):
    """Detect and track beads; lost IDs remain permanently disabled."""
    stack = load_raw(config)
    tracks, visible, candidates = detect_and_track(
        stack, expected_count=config.bead_count, threshold=config.threshold,
        min_radius_px=config.radius_range_px[0], max_radius_px=config.radius_range_px[1],
        min_circularity=config.min_circularity, border_margin_px=config.border_margin_px,
        reject_border=True, max_jump_px=config.max_jump_px)
    kept_ids = np.flatnonzero(np.asarray(visible, dtype=bool))
    tracks = np.asarray(tracks, dtype=float)
    if tracks.shape[1] != len(kept_ids):
        raise RuntimeError("tracker output does not match its visibility map")
    if len(kept_ids) < 3:
        raise RuntimeError(f"only {len(kept_ids)} complete bead tracks remain")
    return tracks, kept_ids, candidates


def calibrate_single_row(config: SingleRowConfig, angles_rad=None) -> dict:
    """Run preprocessing and ellipse-based calibration, then save artifacts."""
    tracks, kept_ids, candidates = preprocess_single_row(config)
    angles = (np.arange(config.views, dtype=float) * 2.0 * np.pi / config.views
              if angles_rad is None else np.asarray(angles_rad, dtype=float))
    distances = make_distance_matrix(config.bead_count, config.spacing_mm)
    result, predicted = calibrate(
        tracks, angles, np.asarray(config.pixel_size_mm, dtype=float),
        distances[np.ix_(kept_ids, kept_ids)], refine=config.refine,
        image_size=(config.rows, config.cols), min_row_span_px=config.min_row_span_px,
        min_axis_ratio=config.min_axis_ratio, min_pair_index_gap=config.min_pair_index_gap,
        max_rotation_iterations=config.max_rotation_iterations,
        rotation_tolerance_px=config.rotation_tolerance_px)
    used_local = np.asarray(result["beads_used_index"], dtype=int)
    report = _jsonable({
        "workflow": "single_row", "input": asdict(config),
        "preprocessing": {
            "detected_tracks": int(tracks.shape[1]),
            "kept_physical_bead_ids": kept_ids,
            "calibration_physical_bead_ids": kept_ids[used_local],
            "candidate_count_min": min(map(len, candidates)),
            "candidate_count_max": max(map(len, candidates))},
        "calibration": result})
    config.output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.output_directory / "tracks.npz", tracks_px=tracks,
                        angles_rad=angles, kept_bead_ids=kept_ids)
    np.savez_compressed(config.output_directory / "reprojection.npz",
                        predicted_px=predicted, residual_px=predicted - tracks[:, used_local])
    (config.output_directory / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
