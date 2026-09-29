"""Validation pipeline for the asymmetric 25-bead double-ring phantom.

The marked phantom has upper diameter 100 mm, lower diameter 80 mm, a 15 deg
lower-ring phase, and one 5 mm axial marker at z=80 mm.  This script uses the
known phantom dimensions to generate conservative image seeds for segmentation
only; the per-view camera matrices are then estimated from image centroids.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from cho_calibration import calibrate_view
from fit_right_hand_joint import project


RAW = Path(r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\quality\double-ring-upper100-lower80-phase15-marker5-above30-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw")


def marked_points():
    pts = []
    for k in range(12):
        a = np.deg2rad(30.0 * k)
        pts.append([50.0 * np.cos(a), 50.0 * np.sin(a), 50.0])
        a = np.deg2rad(30.0 * k + 15.0)
        pts.append([40.0 * np.cos(a), 40.0 * np.sin(a), -50.0])
    pts.append([50.0, 0.0, 80.0])
    return np.asarray(pts, float)


def extract(image, seeds, threshold=5.0):
    ys, xs = np.nonzero(image > threshold)
    pixels = np.column_stack([xs, ys]).astype(float)
    values = np.maximum(image[ys, xs].astype(float) - threshold, 1e-6)
    distance = np.sum((pixels[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    owner = np.argmin(distance, axis=1)
    nearest = np.min(distance, axis=1)
    result = []
    for j in range(len(seeds)):
        selected = (owner == j) & (nearest < 24.0 ** 2)
        if selected.sum() < 4:
            raise ValueError(f"bead {j} has insufficient pixels")
        w = values[selected]
        result.append(np.sum(pixels[selected] * w[:, None], axis=0) / np.sum(w))
    return np.asarray(result)


def fit_joint_geometry(measured, points3d, pixel, center):
    """Fit the shared scanner geometry and the phantom's rigid pose."""
    angles = np.arange(len(measured), dtype=float) * 2.0 * np.pi / len(measured)
    initial = np.array([
        440.0, 770.0, 2.1, 4.2,
        np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(3.0),
        np.deg2rad(2.0), 0.0, 0.0, 10.0, 15.0, 20.0,
    ])
    scale = np.array([
        100.0, 100.0, 2.0, 2.0,
        0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 10.0, 10.0, 10.0,
    ])
    lower = np.array([
        350.0, 650.0, -30.0, -30.0,
        -0.15, -0.15, -0.15, -np.pi, -np.pi, -np.pi,
        -50.0, -50.0, -50.0,
    ])
    upper = np.array([
        550.0, 900.0, 30.0, 30.0,
        0.15, 0.15, 0.15, np.pi, np.pi, np.pi,
        50.0, 50.0, 50.0,
    ])

    def residual(x):
        return (project(x, points3d, angles, pixel, center) - measured).ravel()

    result = least_squares(
        residual, initial, bounds=(lower, upper), x_scale=scale,
        loss="soft_l1", f_scale=0.15, max_nfev=1200,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    predicted = project(result.x, points3d, angles, pixel, center)
    error = predicted - measured
    values = result.x.copy()
    values[4:10] = np.degrees(values[4:10])
    names = [
        "sid_mm", "sdd_mm", "offset_u_mm", "offset_v_mm",
        "tilt_u_deg", "tilt_v_deg", "tilt_n_deg", "phantom_rotvec_x_deg",
        "phantom_rotvec_z_deg", "phantom_offset_x_mm",
        "phantom_offset_y_mm", "phantom_offset_z_mm",
    ]
    return {
        **{name: float(value) for name, value in zip(names, values)},
        "joint_rmse_px": float(np.sqrt(np.mean(error * error))),
        "joint_max_point_error_px": float(np.max(np.linalg.norm(error, axis=2))),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nfev": int(result.nfev),
        "fixed_parameters": {
            "source_offset_x_mm": 0.0,
            "source_offset_y_mm": 0.0,
            "source_offset_z_mm": 0.0,
            "offset_n_mm": 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--report", type=Path, default=Path("marked_double_ring_dlt_report.json"))
    parser.add_argument("--points-output", type=Path, default=Path("marked_double_ring.points.npy"))
    args = parser.parse_args()
    points3d = marked_points()
    pixel = np.array([0.417, 0.417])
    center = np.array([511.5, 511.5])
    # Validation-only seed model; this is not used by DLT itself.
    seed_geometry = np.array([
        440., 770., 2.085, 4.17,
        np.deg2rad(1.), np.deg2rad(2.), np.deg2rad(3.),
        *np.deg2rad([2., 0., 0.]), 10., 15., 20.
    ])
    raw = np.memmap(args.raw, dtype=np.float32, mode="r", shape=(360, 1024, 1024))
    tracks = []
    failures = []
    poses = []
    for i in range(360):
        seeds = project(seed_geometry, points3d,
                        np.array([2.0 * np.pi * i / 360.0]), pixel, center)[0]
        try:
            measured = extract(np.asarray(raw[i]), seeds)
            pose = calibrate_view(measured, type("Cfg", (), {
                "pixel_size_mm": (0.417, 0.417),
                "robust_loss": "soft_l1", "robust_scale_px": 0.15,
                "max_nfev": 500,
            })(), points3d)
            tracks.append(measured)
            poses.append(pose)
        except Exception as exc:
            failures.append({"frame": i, "reason": str(exc)})
    measured_stack = np.asarray(tracks)
    joint = fit_joint_geometry(measured_stack, points3d, pixel, center)
    report = {
        "raw": str(RAW),
        "point_count": len(points3d),
        "valid_views": len(poses),
        "failed_views": failures,
        "mean_reprojection_rmse_px": float(np.mean([p.reprojection_rmse_px for p in poses])) if poses else None,
        "max_reprojection_rmse_px": float(np.max([p.reprojection_rmse_px for p in poses])) if poses else None,
        "marker_point_xyz_mm": points3d[-1].tolist(),
        "first_frame_marker_uv_px": tracks[0][-1].tolist() if tracks else None,
        "first_frame_marker_seed_uv_px": project(seed_geometry, points3d, np.array([0.0]), pixel, center)[0][-1].tolist(),
        "joint_geometry": joint,
    }
    report["raw"] = str(args.raw)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(args.points_output, measured_stack)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
