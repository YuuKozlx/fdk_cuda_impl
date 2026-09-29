# -*- coding: utf-8 -*-
"""Visualize bead tracks AFTER calibration's bead-selection preprocessing.

Read-only diagnostic: does not modify detect.py or calibrate.py.
Two passes:
  1. Compute detection + selection for every case (no plotting).
  2. Plot all cases in one figure (or separate figures) at the end.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import detect
from detect import detect_and_track
from calibrate import select_valid_beads, fit_ellipse


RAW_A = Path(
    r"G:\Code\fanproj\fdk-test\out\test-artifacts\bead-phantoms-v3"
    r"\beads-3mm-10mm-flat-offsetu5-offsetv10-tiltn1deg-1024x1024x359.raw"
)
RAW_B = Path(
    r"G:\Code\fanproj\fdk-test\out\test-artifacts\bead-phantoms-v3"
    r"\beads-2mm-5mm-flat-offsetu5-offsetv10-tiltn1deg-1024x1024x359.raw"
)

VIEWS, ROWS, COLS = 359, 1024, 1024
PIXEL_SIZE_MM = 0.417


# ---------------------------------------------------------------------
# RAW loading
# ---------------------------------------------------------------------
def load_raw(path: Path) -> np.ndarray:
    return np.memmap(path, dtype="<f4", mode="r", shape=(VIEWS, ROWS, COLS))


# ---------------------------------------------------------------------
# Compat: normalize detect_and_track's return (3 or 5 values)
# ---------------------------------------------------------------------
def normalize_detect_result(result):
    if len(result) == 5:
        tracks, bead_ids, lost, visible, candidates = result
        return tracks, bead_ids, np.asarray(visible, dtype=bool), candidates, lost
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
    return tracks, bead_ids, visible, candidates, lost


# ---------------------------------------------------------------------
# Pass 1: compute only (no plotting)
# ---------------------------------------------------------------------
def compute_case(name, path, *, pitch=PIXEL_SIZE_MM, min_axis_ratio=0.02,
                 min_row_span_px=20.0, **detect_kwargs):
    print(f"\n{'='*70}\n[compute] {name}\n{'='*70}")
    print(f"detect module: {detect.__file__}")

    stack = load_raw(path)
    result = detect_and_track(stack, **detect_kwargs)
    tracks, bead_ids, visible, candidates, lost = normalize_detect_result(result)

    print(f"tracks: {tracks.shape}")
    print(f"bead_ids: {bead_ids.tolist()}")
    print(f"visible: {visible.tolist()}")
    print(f"lost: {lost}")
    print(f"candidates/view (min/max): "
          f"{min(map(len, candidates))}/{max(map(len, candidates))}")

    pitch_arr = np.array([pitch, pitch])
    selection = select_valid_beads(
        tracks, pitch_arr, min_axis_ratio=min_axis_ratio,
        min_row_span_px=min_row_span_px,
    )
    used_ids = selection.used_ids
    rejected_ids = selection.rejected_ids

    print(f"\n-- bead selection (min_axis_ratio={min_axis_ratio}, "
          f"min_row_span_px={min_row_span_px}) --")
    for col, bid in enumerate(bead_ids):
        span = selection.row_spans[col]
        ratio = selection.axis_ratios[col]
        reason = selection.rejected_reason.get(col, "kept")
        span_str = f"{span:6.1f}" if np.isfinite(span) else "   NaN"
        ratio_str = f"{ratio:.4f}" if np.isfinite(ratio) else "  NaN"
        print(f"  col {col} (bead_id {bid}): "
              f"row_span={span_str}  axis_ratio={ratio_str}  -> {reason}")
    print(f"used bead_ids:     {bead_ids[used_ids].tolist()}")
    print(f"rejected bead_ids: {bead_ids[rejected_ids].tolist()}")
    print(f"kept {len(used_ids)} / rejected {len(rejected_ids)} / "
          f"total {len(bead_ids)}")

    return {
        "name": name,
        "tracks": tracks,
        "bead_ids": bead_ids,
        "selection": selection,
        "used_ids": used_ids,
        "rejected_ids": rejected_ids,
        "pitch": pitch_arr,
    }


# ---------------------------------------------------------------------
# Pass 2: plot all computed cases
# ---------------------------------------------------------------------
def plot_cases(cases, *, save_dir=None):
    for case in cases:
        name = case["name"]
        tracks = case["tracks"]
        bead_ids = case["bead_ids"]
        selection = case["selection"]
        used_ids = case["used_ids"]
        rejected_ids = case["rejected_ids"]
        pitch_arr = case["pitch"]

        fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
        kept_cols = set(used_ids.tolist())

        for col in range(tracks.shape[1]):
            bid = int(bead_ids[col])
            uv = tracks[:, col]
            finite = np.isfinite(uv).all(axis=1)
            u, v = uv[finite, 0], uv[finite, 1]
            if not len(u):
                continue

            if col in kept_cols:
                line, = ax.plot(u, v, ".", ms=2.5, alpha=0.7,
                                label=f"kept bead {bid} "
                                      f"(span={selection.row_spans[col]:.0f}, "
                                      f"r={selection.axis_ratios[col]:.3f})")
                try:
                    pts = uv[finite] * pitch_arr
                    center, q, _ = fit_ellipse(pts)
                    ax.plot(center[0] / pitch_arr[0],
                            center[1] / pitch_arr[1], "x",
                            color=line.get_color(), ms=10, mew=2)
                except Exception:
                    pass
            else:
                ax.plot(u, v, ".", ms=1.2, alpha=0.25, color="0.7")

        ax.set(xlabel="u (px)", ylabel="v (px)",
               title=f"{name}: kept beads after selection "
                     f"({len(used_ids)} kept / {len(rejected_ids)} rejected)\n"
                     f"min_row_span_px={selection.row_spans[used_ids].min():.0f}"
                     f"..{selection.row_spans[used_ids].max():.0f}"
                     f"  kept | rejected row_span max="
                     f"{selection.row_spans[rejected_ids].max():.1f}"
                     if len(rejected_ids) else
                     f"{name}: kept beads after selection "
                     f"({len(used_ids)} kept / {len(rejected_ids)} rejected)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right", ncol=1)
        ax.grid(alpha=0.3)

        if save_dir is not None:
            out = Path(save_dir) / f"tracks_filtered_{name}.png"
            fig.savefig(out, dpi=150)
            print(f"saved {out}")

    # 所有 case 都算完、画完后，统一弹窗
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    case_configs = [
        dict(
            name="A_10mm", path=RAW_A, pitch=PIXEL_SIZE_MM,
            min_axis_ratio=0.02, min_row_span_px=20.0,
            expected_count=10, threshold=100,
            min_radius_px=2, max_radius_px=20,
            min_circularity=0.35, max_jump_px=60,
        ),
        dict(
            name="B_5mm", path=RAW_B, pitch=PIXEL_SIZE_MM,
            min_axis_ratio=0.02, min_row_span_px=20.0,
            expected_count=10, threshold=100,
            min_radius_px=2, max_radius_px=20,
            min_circularity=0.35, max_jump_px=60,
        ),
    ]

    # Pass 1: 先全部算完
    print("\n########## PASS 1: compute all cases ##########")
    computed = []
    for cfg in case_configs:
        name = cfg.pop("name")
        path = cfg.pop("path")
        computed.append(compute_case(name, path, **cfg))

    # Pass 2: 全部画出来
    print("\n########## PASS 2: plot all cases ##########")
    save_dir = Path(".")          # 改成你要保存的目录，None 表示不保存
    plot_cases(computed, save_dir=save_dir)