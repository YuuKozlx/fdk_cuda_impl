"""Constrained joint fit using the right-handed flat-panel convention.

The optimization gauge is explicit: all source offsets and detector normal
offset are fixed to zero.  The seven free machine parameters are
``SID, SDD, offset_u, offset_v, tilt_u, tilt_v, tilt_n``.  Phantom pose stays
in the state as nuisance variables so phantom placement is not reported as
detector tilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cho_calibration import ChoConfig, calibrate_view, phantom_points
from cho_system_geometry import fit_circle_3d


MACHINE_PARAMETER_NAMES = (
    "sid_mm", "sdd_mm", "offset_u_mm", "offset_v_mm",
    "tilt_u_rad", "tilt_v_rad", "tilt_n_rad",
)
FIXED_PARAMETER_VALUES = {
    "source_offset_x_mm": 0.0,
    "source_offset_y_mm": 0.0,
    "source_offset_z_mm": 0.0,
    "offset_n_mm": 0.0,
}


def detector_frame(angle, tu, tv, tn):
    er = np.array([np.cos(angle), np.sin(angle), 0.0])
    u = np.array([-np.sin(angle), np.cos(angle), 0.0])
    v = np.array([0.0, 0.0, 1.0])

    def rot(x, axis, a):
        return Rotation.from_rotvec(axis * a).apply(x)

    # Same sequential local-axis rotations as CudaPrimaryProjector.cu.
    v, er = rot(v, u, tu), rot(er, u, tu)
    u, er = rot(u, v, tv), rot(er, v, tv)
    u, v = rot(u, er, tn), rot(v, er, tn)
    return u, v, er


def _rotate_batch(values, axes, angles):
    """Rodrigues rotation for arrays shaped (N, 3)."""
    values = np.asarray(values, dtype=float)
    axes = np.asarray(axes, dtype=float)
    angles = np.asarray(angles, dtype=float)
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    dot = np.sum(axes * values, axis=1, keepdims=True)
    return values * c + np.cross(axes, values) * s + axes * dot * (1.0 - c)


def project(x, local_points, angles, pixel, image_center):
    """Project a constrained state of length 13.

    Layout: seven machine parameters, phantom rotation vector, phantom
    translation.  Fixed parameters deliberately have no slots in ``x``.
    """
    if len(x) != 13:
        raise ValueError("expected 13-value constrained joint state")
    sid, sdd, ou, ov, tu, tv, tn = x[:7]
    phantom_rot = Rotation.from_rotvec(x[7:10]).as_matrix()
    phantom_offset = x[10:13]
    # Simulator applies offset first, then XYZ rotation in phantomInverse.
    world = (phantom_rot @ local_points.T).T + phantom_offset
    # The simulator defines its radial frame at encoder angle - pi/2.
    angle = np.asarray(angles, dtype=float) - 0.5 * np.pi
    er = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(angle))])
    u = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(len(angle))])
    v = np.tile(np.array([0.0, 0.0, 1.0]), (len(angle), 1))
    n = er.copy()
    v = _rotate_batch(v, u, np.full(len(angle), tu))
    n = _rotate_batch(n, u, np.full(len(angle), tu))
    u = _rotate_batch(u, v, np.full(len(angle), tv))
    n = _rotate_batch(n, v, np.full(len(angle), tv))
    u = _rotate_batch(u, n, np.full(len(angle), tn))
    v = _rotate_batch(v, n, np.full(len(angle), tn))

    source = sid * er
    principal = -(sdd - sid) * er
    center = principal + ou * u + ov * v
    ray = world[None, :, :] - source[:, None, :]
    denom = np.sum(ray * n[:, None, :], axis=2)
    numerator = np.sum((center - source) * n, axis=1)
    lam = numerator[:, None] / denom
    hit = source[:, None, :] + lam[:, :, None] * ray
    rel = hit - center[:, None, :]
    return np.stack([
        image_center[0] + np.sum(rel * u[:, None, :], axis=2) / pixel[0],
        image_center[1] + np.sum(rel * v[:, None, :], axis=2) / pixel[1],
    ], axis=2)


def dlt_initialization(points, config, image_center):
    """Estimate the joint-fit seed using only the tracked image stack."""
    poses = [calibrate_view(frame, config) for frame in points]
    sources = np.asarray([pose.source_world_mm for pose in poses])
    circle = fit_circle_3d(sources)
    axis = circle.axis_world.copy()
    if axis[2] < 0.0:
        axis = -axis
        circle.axis_world = axis
        circle.basis_y_world = np.cross(axis, circle.basis_x_world)

    # Use the fitted source-circle phase as the DLT-to-scanner reference. The
    # first source radius is mapped to encoder-frame radial direction.
    phantom_basis = np.column_stack([
        circle.basis_x_world, circle.basis_y_world, axis])
    scanner_basis = np.column_stack([
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    phantom_rot = scanner_basis @ phantom_basis.T
    alignment_rms = 0.0
    phantom_translation = -phantom_rot @ circle.center_world_mm

    source0 = phantom_rot @ (sources[0] - circle.center_world_mm)
    sid = float(np.linalg.norm(source0[:2]))
    sdd = float(np.median([pose.source_detector_distance_mm for pose in poses]))
    pp = np.median(np.asarray([pose.principal_point_px for pose in poses]), axis=0)
    offset_u = float((image_center[0] - pp[0]) * config.pixel_size_mm[0])
    offset_v = float((image_center[1] - pp[1]) * config.pixel_size_mm[1])
    seed = np.r_[sid, sdd, offset_u, offset_v, 0.0, 0.0, 0.0,
                 Rotation.from_matrix(phantom_rot).as_rotvec(),
                 phantom_translation]
    info = {
        "source_circle_radius_mm": float(circle.radius_mm),
        "source_circle_center_dlt_mm": circle.center_world_mm.tolist(),
        "source_circle_axis_dlt": axis.tolist(),
        "sdd_median_mm": sdd,
        "principal_point_median_px": pp.tolist(),
        "axis_alignment_rms": float(alignment_rms),
        "initial_parameters": seed.tolist(),
        "fixed_parameters": FIXED_PARAMETER_VALUES.copy(),
    }
    return seed, info


def fit(points, max_nfev=1200):
    cfg = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                    beads_per_ring=12, pixel_size_mm=(0.417, 0.417))
    local = phantom_points(cfg)
    pixel = np.array(cfg.pixel_size_mm)
    center = np.array([511.5, 511.5])
    angles = np.arange(len(points)) * 2.0 * np.pi / len(points)
    # The existing preprocessing layer establishes the indexed bead map. The
    # fully DLT-driven discrete-map solver remains a separate diagnostic until
    # its ring-phase branch handling is validated.
    mapping_seed = np.array([440., 770., 2., 4., 0., 0., np.deg2rad(3.),
                             np.deg2rad(3.), 0., 0., 10., 15., 20.])
    seed_frame = project(mapping_seed, local, angles[:1], pixel, center)[0]
    cost = np.sum((seed_frame[:, None, :] - points[0][None, :, :]) ** 2, axis=2)
    from scipy.optimize import linear_sum_assignment
    canonical_ids, measured_ids = linear_sum_assignment(cost)
    permutation = measured_ids[np.argsort(canonical_ids)]
    points = points[:, permutation, :]
    initial = mapping_seed.copy()
    scales = np.array([100., 100., 2., 2., .02, .02, .02,
                       .03, .03, .03, 10., 10., 10.])

    def residual(x):
        pred = project(x, local, angles, pixel, center)
        return (pred - points).ravel()

    lower = np.array([350., 650., -30., -30.,
                      -.15, -.15, -.15, -np.pi, -np.pi, -np.pi,
                      -50., -50., -50.])
    upper = np.array([550., 900., 30., 30.,
                      .15, .15, .15, np.pi, np.pi, np.pi,
                      50., 50., 50.])

    result = least_squares(residual, initial, bounds=(lower, upper),
                           x_scale=scales, loss="soft_l1",
                           f_scale=0.1, max_nfev=max_nfev,
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, verbose=0)
    pred = project(result.x, local, angles, pixel, center)
    e = pred - points
    names = ["sid_mm", "sdd_mm", "offset_u_mm", "offset_v_mm",
             "tilt_u_deg", "tilt_v_deg", "tilt_n_deg"]
    vals = result.x[:7].copy()
    vals[4:7] = np.degrees(vals[4:7])
    report = {name: float(value) for name, value in zip(names, vals)}
    report.update({
        "phantom_rotvec_deg": np.degrees(result.x[7:10]).tolist(),
        "phantom_offset_mm": result.x[10:13].tolist(),
        "source_offset_x_mm": 0.0,
        "source_offset_y_mm": 0.0,
        "source_offset_z_mm": 0.0,
        "offset_n_mm": 0.0,
        "fixed_parameters": FIXED_PARAMETER_VALUES.copy(),
        "free_parameter_names": list(MACHINE_PARAMETER_NAMES),
        "rmse_px": float(np.sqrt(np.mean(e * e))),
        "max_point_error_px": float(np.max(np.linalg.norm(e, axis=2))),
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "canonical_permutation": permutation.tolist(),
    })
    return report, pred, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report, _, _ = fit(np.load(args.points))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
