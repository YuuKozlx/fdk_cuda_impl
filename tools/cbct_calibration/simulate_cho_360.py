"""360-view Cho 24-bead simulation with fixed detector offset and yaw."""

from __future__ import annotations

import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from cho_calibration import (ChoConfig, calibrate_stack_joint, calibrate_view,
                              phantom_points, project_points)


def truth_view(i, views, config):
    q = 2.0 * np.pi * i / views
    pixel = np.asarray(config.pixel_size_mm)
    sdd = 1600.0
    principal = np.array([511.5, 511.5]) + np.array([5.0, 10.0]) / pixel
    k = np.array([[sdd / pixel[0], 0.0, principal[0]],
                  [0.0, sdd / pixel[1], principal[1]],
                  [0.0, 0.0, 1.0]])
    # Non-ideal source trajectory and detector pose, while K is fixed.
    source = np.array([
        500.0 * np.cos(q) + 3.0 * np.sin(3.0 * q),
        500.0 * np.sin(q) + 2.0 * np.cos(5.0 * q),
        -2.0 + 1.0 * np.sin(2.0 * q),
    ])
    rotvec = np.array([
        0.035 * np.sin(2.0 * q),
        0.028 * np.cos(3.0 * q),
        q + np.deg2rad(1.0) + 0.01 * np.sin(7.0 * q),
    ])
    r = Rotation.from_rotvec(rotvec).as_matrix()
    # Place the source at -1500 mm in detector-frame z; this makes the object
    # points lie in front of the source for the projection convention here.
    source_camera = np.array([source[0] * 0.05, source[1] * 0.05, -1600.0])
    t = -r @ source_camera
    p = k @ np.column_stack([r, t])
    return p, source_camera, sdd, principal


def run(views, noise, seed):
    config = ChoConfig(50.0, 80.0, 12, (0.254, 0.317))
    points = phantom_points(config)
    rng = np.random.default_rng(seed)
    stack = []
    truths = []
    for i in range(views):
        p, source, sdd, principal = truth_view(i, views, config)
        clean = project_points(points, p)
        measured = clean + rng.normal(0.0, noise, clean.shape)
        stack.append(measured)
        truths.append((source, sdd, principal))
    try:
        result = calibrate_stack_joint(np.asarray(stack), config)
    except Exception as exc:
        print(f"views={views} noise={noise:.4f}px FAILED {type(exc).__name__}: {exc}")
        return
    sdd_errors = [abs(result.shared_sdd_mm - truth[1]) for truth in truths]
    offset_errors = [np.linalg.norm((result.shared_principal_point_px - truth[2]) * np.asarray(config.pixel_size_mm)) for truth in truths]
    source_errors = [np.linalg.norm(pose.source_world_mm - truth[0]) for pose, truth in zip(result.poses, truths)]
    print(f"views={views} noise={noise:.4f}px success={len(result.poses)} failure=0")
    print(f"shared_sdd_mm={result.shared_sdd_mm:.9f} error_mm={result.shared_sdd_mm-truths[0][1]:.9f}")
    print(f"shared_principal_px={result.shared_principal_point_px} "
          f"shared_offset_mm={result.shared_offset_mm} "
          f"offset_error_mm={offset_errors[0]:.6g}")
    for key, values in (("sdd", sdd_errors), ("offset_mm", offset_errors),
                        ("source_mm", source_errors),
                        ("view_rmse_px", [p.reprojection_rmse_px for p in result.poses])):
        a = np.asarray(values)
        print(f"{key:<12} mean={a.mean():.6g} rms={np.sqrt(np.mean(a*a)):.6g} "
              f"p95={np.percentile(a,95):.6g} max={a.max():.6g}")
    print(f"joint_rmse_px={result.reprojection_rmse_px:.6g} success={result.optimizer_success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=int, default=360)
    parser.add_argument("--noise-px", type=float, nargs="*", default=[0.0, 0.01])
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    for value in args.noise_px:
        run(args.views, value, args.seed)
