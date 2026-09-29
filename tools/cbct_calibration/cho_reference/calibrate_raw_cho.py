"""Command-line pipeline: preprocessing layer -> calibration layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cho_calibration import ChoConfig, calibrate_stack_joint
from cho_preprocess import track_raw_stack


def process(raw_path: Path, output: Path, threshold: float):
    output.parent.mkdir(parents=True, exist_ok=True)
    config = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                       beads_per_ring=12, pixel_size_mm=(0.417, 0.417),
                       robust_scale_px=0.15, max_nfev=500)
    # Preprocessing output is the stable layer boundary: (views, 24, 2).
    points_px, frame_ids, diagnostics, rejected = track_raw_stack(
        raw_path, config, threshold=threshold, views=360,
        image_shape=(1024, 1024))
    np.save(output.with_suffix(".points.npy"), points_px)
    # Calibration never reads RAW or performs image processing.
    result = calibrate_stack_joint(points_px, config)
    center_px = np.array([511.5, 511.5])
    offset_px = result.shared_principal_point_px - center_px
    report = {
        "raw": str(raw_path), "threshold": threshold, "views": 360,
        "valid_views": len(frame_ids), "valid_frame_ids": frame_ids,
        "rejected_frames": rejected, "pixel_size_mm": config.pixel_size_mm,
        "shared_sdd_mm": result.shared_sdd_mm,
        "shared_principal_point_px": result.shared_principal_point_px.tolist(),
        "shared_offset_px": offset_px.tolist(),
        "shared_offset_mm": result.shared_offset_mm.tolist(),
        "joint_reprojection_rmse_px": result.reprojection_rmse_px,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.optimizer_message,
        "per_view_reprojection_rmse_px": [p.reprojection_rmse_px for p in result.poses],
        "frame_diagnostics": diagnostics,
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cho_raw_result.json"))
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()
    process(args.raw, args.output, args.threshold)
