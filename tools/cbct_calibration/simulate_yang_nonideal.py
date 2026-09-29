"""End-to-end non-ideal CBCT simulation for the Yang PIC calibrator.

The forward projector in this file is deliberately independent of
``yang_pic._transform_i_to_detector``.  Each view has a different source,
phantom centre, detector attitude, image offset, and non-uniform gantry angle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from yang_pic import YangConfig, calibrate_view, geometry_from_pose, project_with_geometry


@dataclass
class Truth:
    source_i_mm: np.ndarray
    center_i_mm: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    gantry_rad: float
    o_px: np.ndarray
    principal_px: np.ndarray
    sdd_mm: float


def wrap_pi(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def independent_forward(truth: Truth, radius_mm: float, half_spacing_mm: float,
                        pixel_size_mm: tuple[float, float]) -> np.ndarray:
    """Project 12 beads by explicit ray/plane intersection in 3-D."""
    theta, phi, eta = truth.roll_rad, truth.pitch_rad, truth.yaw_rad
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)

    # Real detector u/v axes expressed in the virtual detector frame.
    u0 = np.array([cp, st * sp, -ct * sp])
    v0 = np.array([0.0, ct, st])
    u_axis = np.cos(eta) * u0 + np.sin(eta) * v0
    v_axis = -np.sin(eta) * u0 + np.cos(eta) * v0
    normal = np.cross(u_axis, v_axis)

    # Same labelled-bead convention as Eq. (24): alpha=-t-(j-1)pi/3.
    alpha = -truth.gantry_rad - np.arange(6) * np.pi / 3.0
    beads = np.array([
        [radius_mm * np.cos(a), truth.center_i_mm[1] + side * half_spacing_mm,
         truth.center_i_mm[2] + radius_mm * np.sin(a)]
        for side in (-1.0, 1.0) for a in alpha
    ])

    rays = beads - truth.source_i_mm
    denom = rays @ normal
    if np.any(np.abs(denom) < 1.0e-10):
        raise ValueError("simulated ray parallel to detector")
    scale = -(truth.source_i_mm @ normal) / denom
    hits = truth.source_i_mm + scale[:, None] * rays
    detector_mm = np.column_stack([hits @ u_axis, hits @ v_axis])
    return truth.o_px + detector_mm / np.asarray(pixel_size_mm)


def make_truth(view: int, views: int) -> Truth:
    """Generate a smooth non-ideal trajectory with fixed detector errors.

    Detector errors are deliberately fixed over the stack: principal-point
    offset is (5, 10) mm and detector in-plane yaw is +1 degree. Only the
    source/phantom trajectory and gantry sampling are non-ideal.
    """
    q = 2.0 * np.pi * view / views
    # Non-uniform angular sampling, still monotonic for the chosen amplitudes.
    gantry = q + np.deg2rad(0.38) * np.sin(3.0 * q) + np.deg2rad(0.12) * np.sin(11.0 * q)
    # Fixed detector pose: only the in-plane rotation is nonzero here.
    roll = 0.0
    pitch = 0.0
    yaw = np.deg2rad(1.0)

    # Both source and phantom centre wobble.  x=0 is required by the paper's
    # virtual detector definition; y/z are otherwise independently variable.
    source = np.array([
        0.0,
        24.0 + 2.2 * np.sin(q) + 0.55 * np.cos(6.0 * q),
        1180.0 + 7.0 * np.cos(2.0 * q) + 2.0 * np.sin(7.0 * q),
    ])
    center_z = 405.0 + 3.2 * np.sin(2.0 * q) + 0.8 * np.cos(9.0 * q)
    # Enforce S, W and O collinearity, as required by the PIC coordinate frame.
    center_y = source[1] * center_z / source[2]
    center = np.array([0.0, center_y, center_z])
    ct, st, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    u0 = np.array([cp, st * sp, -ct * sp])
    v0 = np.array([0.0, ct, st])
    u_axis = np.cos(yaw) * u0 + np.sin(yaw) * v0
    v_axis = -np.sin(yaw) * u0 + np.cos(yaw) * v0
    normal = np.cross(u_axis, v_axis)
    pixel = np.array([0.254, 0.317])
    # Fixed detector principal-point offset from the image centre, in mm.
    image_center_px = np.array([511.5, 511.5])
    fixed_offset_mm = np.array([5.0, 10.0])
    principal = image_center_px + fixed_offset_mm / pixel
    # O is the perspective projection of W. Choose its pixel coordinate so
    # that the source's normal projection is exactly the fixed principal point.
    o_px = principal - np.array([source @ u_axis, source @ v_axis]) / pixel
    return Truth(source, center, roll, pitch, yaw, gantry, o_px, principal,
                 abs(float(source @ normal)))


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def run_simulation(views: int, noise_px: float, seed: int) -> dict:
    radius, half_spacing = 30.0, 25.0
    pixel = (0.254, 0.317)
    config = YangConfig(radius, half_spacing, pixel)
    rng = np.random.default_rng(seed)
    errors = {name: [] for name in (
        "roll_deg", "pitch_deg", "yaw_deg", "gantry_deg", "source_mm",
        "center_mm", "sdd_mm", "o_px", "principal_px", "reprojection_px",
        "matrix_bead_px")}
    failures = []

    for i in range(views):
        truth = make_truth(i, views)
        clean = independent_forward(truth, radius, half_spacing, pixel)
        measured = clean + rng.normal(0.0, noise_px, clean.shape)
        try:
            pose = calibrate_view(measured, config)
            geom = geometry_from_pose(pose)
            beta = np.arange(6) * np.pi / 3.0
            phantom_beads = np.array([
                [radius * np.cos(a), side * half_spacing, radius * np.sin(a)]
                for side in (-1.0, 1.0) for a in beta
            ])
            matrix_projection = project_with_geometry(phantom_beads, geom)
        except Exception as exc:  # report all algorithmic failures, do not hide them
            failures.append((i, type(exc).__name__, str(exc)))
            continue

        errors["roll_deg"].append(abs(np.rad2deg(wrap_pi(pose.roll_rad - truth.roll_rad))))
        errors["pitch_deg"].append(abs(np.rad2deg(wrap_pi(pose.pitch_rad - truth.pitch_rad))))
        errors["yaw_deg"].append(abs(np.rad2deg(wrap_pi(pose.yaw_rad - truth.yaw_rad))))
        # The fitted implementation tests both ring directions.  Compare the
        # phase modulo 60 degrees because an unmarked six-bead ring is periodic.
        phase_error = wrap_pi(pose.gantry_angle_rad - truth.gantry_rad)
        phase_error = (phase_error + np.pi / 6.0) % (np.pi / 3.0) - np.pi / 6.0
        errors["gantry_deg"].append(abs(np.rad2deg(phase_error)))
        errors["source_mm"].append(np.linalg.norm(pose.source_i_mm - truth.source_i_mm))
        errors["center_mm"].append(np.linalg.norm(pose.phantom_center_i_mm - truth.center_i_mm))
        errors["sdd_mm"].append(abs(pose.source_detector_distance_mm - truth.sdd_mm))
        errors["o_px"].append(np.linalg.norm(pose.o_px - truth.o_px))
        errors["principal_px"].append(np.linalg.norm(pose.principal_point_px - truth.principal_px))
        errors["reprojection_px"].append(float(pose.reprojection_rmse_px))
        errors["matrix_bead_px"].append(float(np.sqrt(np.mean((matrix_projection - clean) ** 2))))

    return {
        "views": views,
        "noise_px": noise_px,
        "successes": views - len(failures),
        "failures": failures,
        "metrics": {name: summarize(np.asarray(vals)) for name, vals in errors.items() if vals},
    }


def print_report(result: dict) -> None:
    print(f"views={result['views']} noise={result['noise_px']:.4f}px "
          f"success={result['successes']} failure={len(result['failures'])}")
    print(f"{'metric':<20} {'mean':>12} {'rms':>12} {'p95':>12} {'max':>12}")
    for name, stat in result["metrics"].items():
        print(f"{name:<20} {stat['mean']:12.6g} {stat['rms']:12.6g} "
              f"{stat['p95']:12.6g} {stat['max']:12.6g}")
    if result["failures"]:
        print("first failures:")
        for item in result["failures"][:5]:
            print("  ", item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=72)
    parser.add_argument("--noise-px", type=float, nargs="*", default=[0.0, 0.01])
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    for noise in args.noise_px:
        print_report(run_simulation(args.views, noise, args.seed))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
