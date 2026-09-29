# -*- coding: utf-8 -*-
"""Compute calibration for the two bead-phantom RAW stacks.

Reads RAW, detects+tracks beads, runs calibrate with bead selection,
prints geometry and diagnostics, and compares against nominal values.

Distances are assumed collinear with equal adjacent spacing, so every
bead pair distance is known: d(i,j) = spacing * |i - j|.
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np

import detect
from detect import detect_and_track
from calibrate import calibrate


# ---------------------------------------------------------------------
# Configuration -- EDIT THESE to match your setup
# ---------------------------------------------------------------------
RAW_A = Path(
    r"G:\Code\fanproj\fdk-test\out\test-artifacts\bead-phantoms-v3"
    r"\beads-3mm-10mm-flat-offsetu5-offsetv10-tiltn1deg-1024x1024x359.raw"
)
RAW_B = Path(
    r"G:\Code\fanproj\fdk-test\out\test-artifacts\bead-phantoms-v3"
    r"\beads-2mm-5mm-flat-offsetu5-offsetv10-tiltn1deg-1024x1024x359.raw"
)

VIEWS, ROWS, COLS = 359, 1024, 1024
PIXEL_SIZE_MM = np.array([0.417, 0.417])
ANGLES = np.arange(VIEWS) * (2 * np.pi / VIEWS)

CENTER_U = (COLS - 1) / 2.0
CENTER_V = (ROWS - 1) / 2.0
OFFSET_U_MM = 5.0
OFFSET_V_MM = 10.0

NOMINAL_A = {
    "sdd_mm": 770.0, "sod_mm": 440.0,
    "u0_px": CENTER_U - OFFSET_U_MM / PIXEL_SIZE_MM[0],
    "v0_px": CENTER_V + OFFSET_V_MM / PIXEL_SIZE_MM[1],
    "eta_deg": 1.0,
}
NOMINAL_B = {
    "sdd_mm": 770.0, "sod_mm": 440.0,
    "u0_px": CENTER_U - OFFSET_U_MM / PIXEL_SIZE_MM[0],
    "v0_px": CENTER_V + OFFSET_V_MM / PIXEL_SIZE_MM[1],
    "eta_deg": 1.0,
}

MIN_ROW_SPAN_PX = 20.0
MIN_AXIS_RATIO = 0.02
MIN_PAIR_INDEX_GAP = 2
MAX_ROTATION_ITERATIONS = 5
ROTATION_TOLERANCE_PX = 1e-4


# ---------------------------------------------------------------------
# RAW loading
# ---------------------------------------------------------------------
def load_raw(path: Path) -> np.ndarray:
    """float32, column-fastest, row next, view slowest -> C-order (views, rows, cols)."""
    return np.memmap(path, dtype="<f4", mode="r", shape=(VIEWS, ROWS, COLS))


# ---------------------------------------------------------------------
# Compat: normalize detect_and_track's return (3 or 5 values)
# ---------------------------------------------------------------------
def call_detect_and_track(stack, **kwargs):
    result = detect_and_track(stack, **kwargs)
    if len(result) == 5:
        return result
    if len(result) != 3:
        raise ValueError(f"detect_and_track returned {len(result)} values; expected 3 or 5")
    tracks, visible, candidates = result
    visible = np.asarray(visible, dtype=bool)
    n_cols = tracks.shape[1]
    if n_cols == visible.sum():
        bead_ids = np.flatnonzero(visible)
        lost = {int(o): {"frame": None, "reason": "legacy_filtered"}
                for o in np.flatnonzero(~visible)}
    elif n_cols == visible.size:
        bead_ids = np.arange(n_cols)
        lost = {}
        for b in range(n_cols):
            bad = ~np.isfinite(tracks[:, b]).all(axis=1)
            if bad.any():
                lost[int(bead_ids[b])] = {"frame": int(np.argmax(bad)),
                                          "reason": "legacy_nan"}
    else:
        raise ValueError(f"tracks cols={n_cols} vs visible.size={visible.size}")
    return tracks, bead_ids, lost, visible, candidates


# ---------------------------------------------------------------------
# Distance matrix: all pairs known for collinear equal-spacing beads
# ---------------------------------------------------------------------
def full_distance_matrix(n_beads: int, spacing_mm: float) -> np.ndarray:
    """Beads collinear with equal adjacent spacing: d(i,j) = spacing * |i-j|."""
    idx = np.arange(n_beads)
    return spacing_mm * np.abs(idx[:, None] - idx[None, :])


# ---------------------------------------------------------------------
# Align tracks + distances by bead_id before calling position-based calibrate
# ---------------------------------------------------------------------
def prepare_for_calibration(tracks, bead_ids, distances_full):
    visible = np.isfinite(tracks).all(axis=(0, 2))
    kept_ids = bead_ids[visible]
    if len(kept_ids) < 3:
        raise ValueError(f"Only {len(kept_ids)} fully visible beads.")
    tracks_cal = tracks[:, visible]
    distances_cal = distances_full[np.ix_(kept_ids, kept_ids)]
    return tracks_cal, kept_ids, distances_cal


# ---------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------
def run_case(name, raw_path, n_beads, spacing_mm, nominal, *,
             max_jump_px=60.0, refine=False,
             min_row_span_px=MIN_ROW_SPAN_PX,
             min_axis_ratio=MIN_AXIS_RATIO,
             image_size=(ROWS, COLS),
             min_pair_index_gap=MIN_PAIR_INDEX_GAP,
             max_rotation_iterations=MAX_ROTATION_ITERATIONS,
             rotation_tolerance_px=ROTATION_TOLERANCE_PX,
             detect_kwargs=None):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    print(f"detect module: {detect.__file__}")
    print(f"pixel_size_mm: {PIXEL_SIZE_MM.tolist()}")
    print(f"image_size:    {image_size}")
    print(f"min_pair_index_gap: {min_pair_index_gap}")
    print(f"max_rotation_iterations: {max_rotation_iterations}")
    print(f"nominal:       {json.dumps(nominal)}")

    stack = load_raw(raw_path)
    print(f"stack: {stack.shape} {stack.dtype} "
          f"range [{stack.min():.3f}, {stack.max():.3f}]")

    dk = dict(expected_count=n_beads, threshold=100,
              min_radius_px=2, max_radius_px=20,
              min_circularity=0.35, max_jump_px=max_jump_px)
    if detect_kwargs:
        dk.update(detect_kwargs)

    tracks, bead_ids, lost, visible, candidates = call_detect_and_track(stack, **dk)
    print(f"bead_ids: {bead_ids.tolist()}")
    print(f"visible: {np.asarray(visible).tolist()}")
    print(f"lost: {lost}")
    print(f"candidates/view min/max: "
          f"{min(map(len, candidates))}/{max(map(len, candidates))}")

    distances_full = full_distance_matrix(n_beads, spacing_mm)
    tracks_cal, kept_ids, distances_cal = prepare_for_calibration(
        tracks, bead_ids, distances_full
    )
    print(f"beads fed to calibrate: {kept_ids.tolist()} (column order)")

    result, predicted = calibrate(
        tracks_cal, ANGLES, PIXEL_SIZE_MM, distances_cal,
        refine=refine,
        min_row_span_px=min_row_span_px,
        min_axis_ratio=min_axis_ratio,
        image_size=image_size,
        min_pair_index_gap=min_pair_index_gap,
        max_rotation_iterations=max_rotation_iterations,
        rotation_tolerance_px=rotation_tolerance_px,
    )

    diag = result["diagnostics"]
    print("\n-- bead selection --")
    print(f"  used:     {diag['beads_used_index']}")
    print(f"  rejected: {diag['beads_rejected_index']}")
    print(f"  reasons:  {diag['beads_rejected_reason']}")
    print(f"  axis_ratios: {[round(r,4) if np.isfinite(r) else None for r in diag['axis_ratios']]}")
    print(f"  row_spans:   {[round(s,1) if np.isfinite(s) else None for s in diag['row_spans']]}")
    print(f"  n_known_pairs: {diag.get('n_known_pairs')}")
    print(f"  min_pair_index_gap: {diag.get('min_pair_index_gap')}")
    print(f"  rotation_iterations: {diag.get('rotation_iterations')}")
    print(f"  rotation_center_px: {diag.get('rotation_center_px')}")

    print("\n-- recovered geometry --")
    print(json.dumps(result["geometry"], indent=2))

    print("\n-- key diagnostics --")
    for key in ("rmse_px", "max_point_error_px", "rmse_per_bead_px",
                "distance_rmse_mm"):
        print(f"  {key}: {diag.get(key)}")

    print("\n-- vs nominal --")
    for key, target in nominal.items():
        got = result["geometry"][key]
        diff = got - target
        print(f"  {key:8s}: got {got:10.4f}  nominal {target:10.4f}  diff {diff:+10.4f}")

    print("\n-- warnings --")
    for w in diag.get("warnings", []):
        print(f"  ! {w}")

    return result, predicted, kept_ids


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_case(
            "A: 3mm bead, 10mm spacing, offsetu5mm offsetv10mm tiltn1deg",
            RAW_A, n_beads=20, spacing_mm=10.0, nominal=NOMINAL_A,
            refine=False, min_row_span_px=30.0,
            image_size=(ROWS, COLS),
            min_pair_index_gap=1,
            max_rotation_iterations=1,
        )
    except Exception as exc:
        print(f"CASE A FAILED: {type(exc).__name__}: {exc}")

    try:
        run_case(
            "B: 2mm bead, 5mm spacing, offsetu5mm offsetv10mm tiltn1deg",
            RAW_B, n_beads=30, spacing_mm=5.0, nominal=NOMINAL_B,
            refine=False, min_row_span_px=30.0,
            image_size=(ROWS, COLS),
            min_pair_index_gap=1,
            max_rotation_iterations=1,
        )
    except Exception as exc:
        print(f"CASE B FAILED: {type(exc).__name__}: {exc}")