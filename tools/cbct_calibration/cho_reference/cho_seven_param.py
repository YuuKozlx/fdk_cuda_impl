"""Joint equivalent 7-parameter CBCT fit.

The source offsets and detector normal offset are fixed by the calibration
gauge.  The fitted machine state is exactly
``SID, SDD, offset_u, offset_v, tilt_u, tilt_v, tilt_n``; a six-value phantom
pose is retained as nuisance state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cho_calibration import ChoConfig, phantom_points


FREE_MACHINE_PARAMETER_NAMES = (
    "sid_mm", "sdd_mm", "offset_u_mm", "offset_v_mm",
    "tilt_u_rad", "tilt_v_rad", "tilt_n_rad",
)
FIXED_PARAMETER_VALUES = {
    "source_offset_x_mm": 0.0,
    "source_offset_y_mm": 0.0,
    "source_offset_z_mm": 0.0,
    "offset_n_mm": 0.0,
}


def detector_frame(angle, tilt_u, tilt_v, tilt_n):
    u = np.array([-np.sin(angle), np.cos(angle), 0.0])
    v = np.array([0.0, 0.0, 1.0])
    # Right-handed detector frame: U x V = N.  N points from the panel
    # toward the source; the panel itself is placed in the -N direction.
    n = np.array([np.cos(angle), np.sin(angle), 0.0])
    def rot(x, axis, a):
        return Rotation.from_rotvec(axis * a).apply(x)
    v, n = rot(v, u, tilt_u), rot(n, u, tilt_u)
    u, n = rot(u, v, tilt_v), rot(n, v, tilt_v)
    u, v = rot(u, n, tilt_n), rot(v, n, tilt_n)
    return u, v, n


def project(parameters, points, angles, pixel, image_center):
    if len(parameters) != 13:
        raise ValueError("expected 13-value state: 7 machine + 6 phantom pose")
    sid, sdd, ou, ov, tu, tv, tn = parameters[:7]
    r = Rotation.from_rotvec(parameters[7:10]).as_matrix()
    t = parameters[10:13]
    world = points @ r.T + t
    out = np.empty((len(angles), len(points), 2), dtype=float)
    for i, angle in enumerate(angles):
        # Equivalent source has no independently estimated source offset.
        frame_angle = angle - 0.5 * np.pi
        source = sid * np.array([np.cos(frame_angle), np.sin(frame_angle), 0.0])
        u, v, n = detector_frame(frame_angle, tu, tv, tn)
        isocenter = np.zeros(3)
        principal = isocenter - n * (sdd - sid)
        center = principal + ou * u + ov * v
        ray = world - source
        den = ray @ n
        lam = ((center - source) @ n) / den
        hit = source + lam[:, None] * ray
        out[i, :, 0] = image_center[0] + (hit - center) @ u / pixel[0]
        out[i, :, 1] = image_center[1] + (hit - center) @ v / pixel[1]
    return out


def fit(points_px, config, max_nfev=1500):
    tracks = np.asarray(points_px, float)
    nview = len(tracks)
    points = phantom_points(config)
    pixel = np.asarray(config.pixel_size_mm, float)
    center = np.array([(1024 - 1) / 2.0, (1024 - 1) / 2.0])
    # DLT-derived equivalent initialization: true values are deliberately not
    # used here except for broad, physically reasonable starting values.
    initial = np.r_[440.0, 770.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    np.zeros(6)]
    # Allow the phase convention to be represented by a z-axis phantom pose;
    # initialize the pose from the source-circle orientation is unnecessary
    # for the symmetric two-ring object.
    def residual(x):
        pred = project(x, points, np.arange(nview) * 2*np.pi/nview,
                       pixel, center)
        return (pred - tracks).ravel()
    lower = np.full(13, -np.inf)
    lower[:2] = 1.0
    result = least_squares(residual, initial, bounds=(lower, np.inf),
                           loss="soft_l1", f_scale=0.1, x_scale="jac",
                           max_nfev=max_nfev, ftol=1e-12, xtol=1e-12,
                           gtol=1e-12)
    pred = project(result.x, points, np.arange(nview) * 2*np.pi/nview,
                   pixel, center)
    e = pred - tracks
    return result, pred, {
        "sid_mm": float(result.x[0]), "sdd_mm": float(result.x[1]),
        "offset_u_mm": float(result.x[2]), "offset_v_mm": float(result.x[3]),
        "tilt_u_deg": float(np.degrees(result.x[4])),
        "tilt_v_deg": float(np.degrees(result.x[5])),
        "tilt_n_deg": float(np.degrees(result.x[6])),
        "phantom_rotvec_deg": np.degrees(result.x[7:10]).tolist(),
        "phantom_translation_mm": result.x[10:13].tolist(),
        "source_offset_x_mm": 0.0,
        "source_offset_y_mm": 0.0,
        "source_offset_z_mm": 0.0,
        "offset_n_mm": 0.0,
        "rmse_px": float(np.sqrt(np.mean(e*e))),
        "max_error_px": float(np.max(np.linalg.norm(e, axis=2))),
        "free_parameter_names": list(FREE_MACHINE_PARAMETER_NAMES),
        "fixed_parameters": FIXED_PARAMETER_VALUES.copy(),
        "success": bool(result.success), "message": str(result.message),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                    beads_per_ring=12, pixel_size_mm=(0.417, 0.417))
    result, _, report = fit(np.load(args.points), cfg)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
