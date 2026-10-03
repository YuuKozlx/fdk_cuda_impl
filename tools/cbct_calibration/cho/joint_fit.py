"""Fixed-source gauge joint fit for an already indexed Cho point stack.

This is the engineering stage after the paper-style per-view calculation.
The source offsets and detector normal offset are fixed to zero.  SID is
fixed to the DLT source-circle radius; SDD, detector U/V offsets, three
detector tilts, and the six nuisance phantom-pose values are optimized.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .dlt import ChoConfig, calibrate_view, phantom_points
from .source_circle import fit_circle_3d


FIXED_PARAMETERS = {
    "source_offset_x_mm": 0.0,
    "source_offset_y_mm": 0.0,
    "source_offset_z_mm": 0.0,
    "offset_n_mm": 0.0,
}


@dataclass(frozen=True)
class JointFitResult:
    machine: dict
    phantom_pose: dict
    source_circle: dict
    diagnostics: dict


def reverse_ring_handedness(points_px: np.ndarray,
                            beads_per_ring: int) -> np.ndarray:
    """Reverse both ring orders while keeping each marked bead zero fixed.

    A circular ring and its mirror have the same single-view DLT residual.
    The external marker fixes bead zero and the upper/lower ring, but it does
    not fix the direction in which the remaining beads are numbered. The two
    choices only become distinguishable after enforcing one rigid phantom
    pose and one scanner geometry over the complete acquisition.
    """
    points = np.asarray(points_px, dtype=float)
    n = int(beads_per_ring)
    if points.ndim != 3 or points.shape[1:] != (2 * n, 2):
        raise ValueError(f"points_px must have shape (views,{2 * n},2)")
    reverse = np.mod(-np.arange(n), n)
    return np.concatenate([points[:, :n][:, reverse],
                           points[:, n:][:, reverse]], axis=1)


def fit_with_handedness_search(
        points_px: np.ndarray, config: ChoConfig,
        image_shape: tuple[int, int] = (1024, 1024),
        frame_ids: np.ndarray | None = None,
        total_views: int | None = None,
        max_nfev: int = 1200,
) -> tuple[np.ndarray, JointFitResult, dict]:
    """Select the ring numbering hand by global mechanical-model closure.

    Single-view reprojection cannot distinguish the two mirror-related ring
    maps. Fit both maps to the same fixed-source circular scanner model and
    retain the one with the lower all-view reprojection RMSE.
    """
    original = np.asarray(points_px, dtype=float)
    candidates = {
        "as_tracked": original,
        "ring_order_reversed": reverse_ring_handedness(
            original, config.beads_per_ring),
    }
    fitted = {
        name: fit_fixed_source(values, config, image_shape, frame_ids,
                               total_views, max_nfev)
        for name, values in candidates.items()
    }
    selected = min(fitted, key=lambda name: fitted[name].diagnostics["rmse_px"])
    diagnostics = {
        "selected": selected,
        "criterion": "minimum all-view fixed-source joint reprojection RMSE",
        "candidate_rmse_px": {
            name: float(result.diagnostics["rmse_px"])
            for name, result in fitted.items()
        },
        "single_view_dlt_is_insufficient": True,
    }
    return candidates[selected].copy(), fitted[selected], diagnostics


def phantom_to_scanner_from_circle(circle) -> tuple[np.ndarray, np.ndarray]:
    """Build the rigid phantom-to-scanner seed from the DLT source circle."""
    axis = np.asarray(circle.axis_world, dtype=float)
    if axis[2] < 0.0:
        axis = -axis
    radial0 = np.asarray(circle.basis_x_world, dtype=float)
    tangent = np.cross(axis, radial0)
    tangent /= np.linalg.norm(tangent)
    radial0 = np.cross(tangent, axis)
    phantom_basis = np.column_stack([radial0, tangent, axis])
    scanner_basis = np.column_stack([
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    rotation = scanner_basis @ phantom_basis.T
    translation = -rotation @ np.asarray(circle.center_world_mm)
    return rotation, translation


def _rotate(values: np.ndarray, axes: np.ndarray, angle: float) -> np.ndarray:
    angles = np.full(len(values), angle)
    return Rotation.from_rotvec(axes * angles[:, None]).apply(values)


def project(state: np.ndarray, points_mm: np.ndarray, angles_rad: np.ndarray,
            pixel_size_mm: tuple[float, float], image_center_px: np.ndarray) -> np.ndarray:
    """Project a 13-value fixed-source-gauge state."""
    sid, sdd, offset_u, offset_v, tilt_u, tilt_v, tilt_n = state[:7]
    phantom_rotation = Rotation.from_rotvec(state[7:10]).as_matrix()
    world = np.asarray(points_mm) @ phantom_rotation.T + state[10:13]
    angle = np.asarray(angles_rad) - 0.5 * np.pi
    radial = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(angle))])
    u_axis = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(len(angle))])
    v_axis = np.tile([0.0, 0.0, 1.0], (len(angle), 1))
    normal = radial.copy()
    v_axis, normal = _rotate(v_axis, u_axis, tilt_u), _rotate(normal, u_axis, tilt_u)
    u_axis, normal = _rotate(u_axis, v_axis, tilt_v), _rotate(normal, v_axis, tilt_v)
    u_axis, v_axis = _rotate(u_axis, normal, tilt_n), _rotate(v_axis, normal, tilt_n)
    source = sid * radial
    detector_center = (-(sdd - sid) * radial
                       + offset_u * u_axis + offset_v * v_axis)
    ray = world[None, :, :] - source[:, None, :]
    scale = (np.sum((detector_center - source) * normal, axis=1)[:, None]
             / np.sum(ray * normal[:, None, :], axis=2))
    hit = source[:, None, :] + scale[:, :, None] * ray
    relative = hit - detector_center[:, None, :]
    pitch = np.asarray(pixel_size_mm)
    return np.stack([
        image_center_px[0] + np.sum(relative * u_axis[:, None, :], axis=2) / pitch[0],
        image_center_px[1] + np.sum(relative * v_axis[:, None, :], axis=2) / pitch[1],
    ], axis=2)


def fit_fixed_source(points_px: np.ndarray, config: ChoConfig,
                     image_shape: tuple[int, int] = (1024, 1024),
                     frame_ids: np.ndarray | None = None,
                     total_views: int | None = None,
                     max_nfev: int = 1200) -> JointFitResult:
    """Run DLT source-circle initialization followed by the constrained fit."""
    measured = np.asarray(points_px, dtype=float)
    local = phantom_points(config)
    if measured.ndim != 3 or measured.shape[1:] != (len(local), 2):
        raise ValueError(f"points_px must have shape (views,{len(local)},2)")
    ids = np.arange(len(measured)) if frame_ids is None else np.asarray(frame_ids)
    period = len(measured) if total_views is None else int(total_views)
    if ids.shape != (len(measured),) or period <= int(np.max(ids)):
        raise ValueError("frame_ids/total_views do not describe the point stack")
    poses = [calibrate_view(frame, config, local) for frame in measured]
    circle = fit_circle_3d(np.asarray([pose.source_world_mm for pose in poses]))
    rotation0, translation0 = phantom_to_scanner_from_circle(circle)
    # Resolve the remaining in-plane phase from the first indexed ring.  The
    # source-circle basis alone is anchored at an arbitrary image frame; the
    # marked phantom makes the first bead correspondence observable.
    local = phantom_points(config)
    camera0 = poses[0]
    measured0 = measured[0]
    phase_candidates = np.linspace(-np.pi, np.pi, 73)[:-1]
    best_phase = 0.0
    best_phase_score = np.inf
    for phase in phase_candidates:
        phase_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, phase])).as_matrix()
        trial_state = np.r_[circle.radius_mm, camera0.source_detector_distance_mm,
                            np.zeros(5), Rotation.from_matrix(phase_rotation @ rotation0).as_rotvec(),
                            translation0]
        trial_px = project(trial_state, local, np.array([0.0]), config.pixel_size_mm,
                           np.array([(image_shape[1] - 1) / 2.0,
                                     (image_shape[0] - 1) / 2.0]))[0]
        score = float(np.sqrt(np.mean((trial_px - measured0) ** 2)))
        if score < best_phase_score:
            best_phase_score, best_phase = score, phase
    phase_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, best_phase])).as_matrix()
    rotation0 = phase_rotation @ rotation0
    translation0 = -rotation0 @ np.asarray(circle.center_world_mm)
    center = np.array([(image_shape[1] - 1) / 2.0,
                       (image_shape[0] - 1) / 2.0])
    pixel = np.asarray(config.pixel_size_mm)
    principal = np.median([pose.principal_point_px for pose in poses], axis=0)
    initial = np.r_[circle.radius_mm,
                    np.median([pose.source_detector_distance_mm for pose in poses]),
                    (center - principal) * pixel,
                    np.zeros(3), Rotation.from_matrix(rotation0).as_rotvec(),
                    translation0]
    free = np.arange(1, 13)  # SID remains the DLT source-circle radius.
    scales = np.array([100.0, 2.0, 2.0, 0.02, 0.02, 0.02,
                       0.03, 0.03, 0.03, 10.0, 10.0, 10.0])
    candidates = []
    for direction in (1, -1):
        angles = direction * ids * 2.0 * np.pi / period

        def residual(values):
            state = initial.copy()
            state[free] = values
            return (project(state, local, angles, config.pixel_size_mm, center)
                    - measured).ravel()

        coarse = least_squares(
            residual, initial[free], x_scale=scales, loss="linear",
            max_nfev=max(200, max_nfev // 2),
            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        solved = least_squares(
            residual, coarse.x, x_scale=scales, loss="soft_l1",
            f_scale=config.robust_scale_px, max_nfev=max_nfev,
            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        state = initial.copy()
        state[free] = solved.x
        error = residual(solved.x).reshape(measured.shape)
        candidates.append((float(np.sqrt(np.mean(error * error))), direction,
                           solved, state, error))
    rmse, direction, solved, state, error = min(candidates, key=lambda item: item[0])
    return JointFitResult(
        machine={
            "sid_mm": float(state[0]), "sdd_mm": float(state[1]),
            "offset_u_mm": float(state[2]), "offset_v_mm": float(state[3]),
            "offset_u_px": float(state[2] / pixel[0]),
            "offset_v_px": float(state[3] / pixel[1]),
            "tilt_u_deg": float(np.degrees(state[4])),
            "tilt_v_deg": float(np.degrees(state[5])),
            "tilt_n_deg": float(np.degrees(state[6])),
            **FIXED_PARAMETERS,
        },
        phantom_pose={
            "rotation_vector_deg": np.degrees(state[7:10]).tolist(),
            "translation_mm": state[10:13].tolist(),
        },
        source_circle={
            "center_phantom_mm": circle.center_world_mm.tolist(),
            "axis_phantom": circle.axis_world.tolist(),
            "radius_mm": float(circle.radius_mm),
            "radial_rms_mm": float(circle.radial_rms_mm),
            "axial_rms_mm": float(circle.axial_rms_mm),
        },
        diagnostics={
            "rmse_px": rmse,
            "max_point_error_px": float(np.max(np.linalg.norm(error, axis=2))),
            "rotation_direction": int(direction),
            "success": bool(solved.success), "nfev": int(solved.nfev),
            "gauge": "source offsets and detector normal offset fixed to zero",
        },
    )
