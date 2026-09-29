"""Estimate a source circle from per-view DLT, then lock it in joint fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from analyze_marked_double_ring import marked_points
from cho_calibration import calibrate_view
from cho_system_geometry import fit_circle_3d
from fit_right_hand_joint import project


class Config:
    pixel_size_mm = (0.417, 0.417)
    robust_loss = "soft_l1"
    robust_scale_px = 0.15
    max_nfev = 500


def scanner_alignment(circle):
    axis = circle.axis_world.copy()
    bx = circle.basis_x_world.copy()
    if axis[2] < 0:
        axis = -axis
    by = np.cross(axis, bx)
    by /= np.linalg.norm(by)
    bx = np.cross(by, axis)
    phantom_basis = np.column_stack([bx, by, axis])
    # Encoder frame at view zero: radial -Y, tangent +X, axial +Z.
    scanner_basis = np.column_stack([
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    return scanner_basis @ phantom_basis.T


def run(points):
    local = marked_points()
    cfg = Config()
    poses = [calibrate_view(frame, cfg, local) for frame in points]
    sources = np.asarray([p.source_world_mm for p in poses])
    circle = fit_circle_3d(sources)
    align = scanner_alignment(circle)
    sid_fixed = float(circle.radius_mm)
    pixel = np.array(cfg.pixel_size_mm)
    center = np.array([511.5, 511.5])
    sdd0 = float(np.median([p.source_detector_distance_mm for p in poses]))
    principal0 = np.median(
        np.asarray([p.principal_point_px for p in poses]), axis=0)
    offset0 = (center - principal0) * pixel
    phantom_rotation0 = Rotation.from_matrix(align).as_rotvec()
    phantom_translation0 = -align @ circle.center_world_mm
    x0 = np.array([
        sid_fixed, sdd0, offset0[0], offset0[1],
        0.0, 0.0, 0.0,
        *phantom_rotation0, *phantom_translation0,
    ])
    # SID is fixed by the fitted source-circle radius.  The remaining seven
    # machine values and six phantom-pose values are optimized.
    free = np.arange(1, 13)
    scale = np.array([100., 2., 2., .02, .02, .02, .03, .03, .03, 10., 10., 10.])

    candidates = []
    for rotation_direction in (1, -1):
        angles = (rotation_direction * np.arange(len(points))
                  * 2.0 * np.pi / len(points))

        def residual(y):
            x = x0.copy()
            x[free] = y
            return (project(x, local, angles, pixel, center) - points).ravel()

        result = least_squares(
            residual, x0[free], x_scale=scale,
            loss="soft_l1", f_scale=0.15, max_nfev=1200,
            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        x = x0.copy()
        x[free] = result.x
        error = residual(result.x).reshape(points.shape)
        rmse = float(np.sqrt(np.mean(error * error)))
        candidates.append((rmse, rotation_direction, result, x, error))
    _, rotation_direction, result, x, error = min(
        candidates, key=lambda item: item[0])
    return {
        "dlt_source_circle": {
            "center_phantom_mm": circle.center_world_mm.tolist(),
            "axis_phantom": circle.axis_world.tolist(),
            "radius_mm": circle.radius_mm,
            "radial_rms_mm": circle.radial_rms_mm,
            "axial_rms_mm": circle.axial_rms_mm,
            "fixed_sid_mm": sid_fixed,
            "fixed_source_offset_mm": [0.0, 0.0, 0.0],
            "fixed_offset_n_mm": 0.0,
            "joint_initialization": {
                "source": "per-view DLT source circle",
                "sdd_mm": sdd0,
                "principal_point_px": principal0.tolist(),
                "offset_uv_mm": offset0.tolist(),
                "detector_tilt_deg": [0.0, 0.0, 0.0],
                "phantom_rotvec": phantom_rotation0.tolist(),
                "phantom_translation_mm": phantom_translation0.tolist(),
                "uses_nominal_system_parameters": False,
            },
        },
        "joint_fit": {
            "sdd_mm": float(x[1]),
            "offset_u_mm": float(x[2]),
            "offset_u_px": float(x[2] / pixel[0]),
            "offset_v_mm": float(x[3]),
            "offset_v_px": float(x[3] / pixel[1]),
            "tilt_u_deg": float(np.degrees(x[4])),
            "tilt_v_deg": float(np.degrees(x[5])),
            "tilt_n_deg": float(np.degrees(x[6])),
            "phantom_rotvec_deg": np.degrees(x[7:10]).tolist(),
            "phantom_offset_mm": x[10:13].tolist(),
            "rmse_px": float(np.sqrt(np.mean(error * error))),
            "max_point_error_px": float(np.max(np.linalg.norm(error, axis=2))),
            "success": bool(result.success),
            "nfev": int(result.nfev),
            "inferred_rotation_direction": int(rotation_direction),
            "direction_candidate_rmse_px": {
                str(direction): float(rmse)
                for rmse, direction, _, _, _ in candidates
            },
            "free_parameter_names": [
                "sdd_mm", "offset_u_mm", "offset_v_mm", "tilt_u_rad",
                "tilt_v_rad", "tilt_n_rad", "phantom_rotvec_x_rad",
                "phantom_rotvec_y_rad", "phantom_rotvec_z_rad",
                "phantom_offset_x_mm", "phantom_offset_y_mm",
                "phantom_offset_z_mm",
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(np.load(args.points))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
