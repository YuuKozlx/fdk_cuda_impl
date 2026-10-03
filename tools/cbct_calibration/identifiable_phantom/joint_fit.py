"""Fixed-source-gauge seven-parameter joint fit for the indexed phantom."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .dlt import DltConfig, calibrate_view
from .phantom import marked_points
from .source_circle import fit_source_circle, scanner_alignment


FIXED_PARAMETERS = {"source_offset_x_mm": 0.0, "source_offset_y_mm": 0.0,
                    "source_offset_z_mm": 0.0, "offset_n_mm": 0.0}


def _rotate(values, axes, angles):
    values, axes, angles = np.asarray(values), np.asarray(axes), np.asarray(angles)
    c, s = np.cos(angles)[:, None], np.sin(angles)[:, None]
    return values * c + np.cross(axes, values) * s + axes * np.sum(axes * values, axis=1, keepdims=True) * (1 - c)


def project(state, local_points, angles, pixel, image_center):
    """Project the 13-value state; source/normal offsets are intentionally absent."""
    if len(state) != 13:
        raise ValueError("expected 13-value constrained joint state")
    sid, sdd, ou, ov, tu, tv, tn = state[:7]
    world = np.asarray(local_points) @ Rotation.from_rotvec(state[7:10]).as_matrix().T + state[10:13]
    angle = np.asarray(angles) - np.pi / 2
    radial = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(angle))])
    u = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(len(angle))])
    v = np.tile([0.0, 0.0, 1.0], (len(angle), 1)); n = radial.copy()
    v, n = _rotate(v, u, np.full(len(angle), tu)), _rotate(n, u, np.full(len(angle), tu))
    u, n = _rotate(u, v, np.full(len(angle), tv)), _rotate(n, v, np.full(len(angle), tv))
    u, v = _rotate(u, n, np.full(len(angle), tn)), _rotate(v, n, np.full(len(angle), tn))
    source = sid * radial
    center = -(sdd - sid) * radial + ou * u + ov * v
    rays = world[None, :, :] - source[:, None, :]
    scale = np.sum((center - source) * n, axis=1)[:, None] / np.sum(rays * n[:, None, :], axis=2)
    relative = source[:, None, :] + scale[:, :, None] * rays - center[:, None, :]
    return np.stack([image_center[0] + np.sum(relative * u[:, None, :], axis=2) / pixel[0],
                     image_center[1] + np.sum(relative * v[:, None, :], axis=2) / pixel[1]], axis=2)


def fit(points, pixel_size_mm=(0.417, 0.417), image_shape=(1024, 1024), max_nfev=1200):
    """Estimate the seven machine values after DLT source-circle initialization."""
    points = np.asarray(points, dtype=float)
    local = marked_points(); pixel = np.asarray(pixel_size_mm, dtype=float)
    cfg = DltConfig(pixel_size_mm=tuple(pixel_size_mm))
    poses = [calibrate_view(frame, cfg, local) for frame in points]
    sources = np.asarray([pose.source_phantom_mm for pose in poses])
    circle = fit_source_circle(sources)
    rotation0, translation0 = scanner_alignment(circle)
    center = np.array([(image_shape[1] - 1) / 2, (image_shape[0] - 1) / 2])
    principal = np.median([pose.principal_point_px for pose in poses], axis=0)
    state0 = np.r_[circle.radius_mm, np.median([p.sdd_mm for p in poses]),
                   (center - principal) * pixel, np.zeros(3),
                   Rotation.from_matrix(rotation0).as_rotvec(), translation0]
    free = np.arange(1, 13)
    candidates = []
    for direction in (1, -1):
        angles = direction * np.arange(len(points)) * 2 * np.pi / len(points)
        def residual(values):
            state = state0.copy(); state[free] = values
            return (project(state, local, angles, pixel, center) - points).ravel()
        result = least_squares(residual, state0[free], x_scale=np.array([100, 2, 2, .02, .02, .02, .03, .03, .03, 10, 10, 10]),
                               loss="soft_l1", f_scale=.15, max_nfev=max_nfev,
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        state = state0.copy(); state[free] = result.x
        error = residual(result.x).reshape(points.shape)
        candidates.append((np.sqrt(np.mean(error * error)), direction, state, result, error))
    rmse, direction, state, result, error = min(candidates, key=lambda item: item[0])
    return {"machine": {"sid_mm": float(state[0]), "sdd_mm": float(state[1]),
                         "offset_u_mm": float(state[2]), "offset_v_mm": float(state[3]),
                         "offset_u_px": float(state[2] / pixel[0]), "offset_v_px": float(state[3] / pixel[1]),
                         "tilt_u_deg": float(np.degrees(state[4])), "tilt_v_deg": float(np.degrees(state[5])),
                         "tilt_n_deg": float(np.degrees(state[6])), **FIXED_PARAMETERS},
            "phantom_pose": {"rotation_vector_deg": np.degrees(state[7:10]).tolist(),
                             "translation_mm": state[10:13].tolist()},
            "source_circle": {"center_phantom_mm": circle.center_phantom_mm.tolist(),
                              "axis_phantom": circle.axis_phantom.tolist(), "radius_mm": float(circle.radius_mm),
                              "radial_rms_mm": float(circle.radial_rms_mm), "axial_rms_mm": float(circle.axial_rms_mm)},
            "diagnostics": {"rmse_px": float(rmse), "max_point_error_px": float(np.max(np.linalg.norm(error, axis=2))),
                            "rotation_direction": int(direction), "success": bool(result.success), "nfev": int(result.nfev)}}
