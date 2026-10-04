"""Cho-style complete per-view cone-beam geometry calibration.

This independent implementation uses the 24-bead, two-ring phantom described
by Cho et al. (2005).  All bead correspondences are used simultaneously:

1. normalized DLT supplies a complete projective-camera initial estimate;
2. RQ decomposition converts it to physical cone-beam parameters;
3. bounded nonlinear least squares minimizes the 24-bead reprojection error.

The final geometric model is ``P = K [R | t]`` with known rectangular detector
pixel pitches and unknown SDD/principal point.  It avoids the Yang solver's
ellipse-axis intersections and finite/infinite vanishing-point branch switch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from ..math.dlt import (EPS, decompose_projection, normalized_dlt,
                        project_points)


@dataclass
class ChoConfig:
    ring_radius_mm: float = 50.0
    ring_half_spacing_mm: float = 80.0
    beads_per_ring: int = 12
    pixel_size_mm: tuple[float, float] = (0.254, 0.254)
    robust_loss: str = "soft_l1"
    robust_scale_px: float = 0.1
    max_nfev: int = 2000
    # External markers are used for correspondence only.  PIC continues to
    # consume the 2*N ring points; markers may optionally be added to DLT.
    marker_points_mm: tuple[tuple[float, float, float], ...] = ((50.0, 0.0, 80.0),)


@dataclass
class ChoPose:
    projection_matrix: np.ndarray
    intrinsic_matrix: np.ndarray
    rotation_world_to_detector: np.ndarray
    translation_world_to_detector_mm: np.ndarray
    source_world_mm: np.ndarray
    source_detector_distance_mm: float
    principal_point_px: np.ndarray
    detector_u_axis_world: np.ndarray
    detector_v_axis_world: np.ndarray
    detector_normal_world: np.ndarray
    detector_origin_world_mm: np.ndarray
    reprojection_rmse_px: float
    max_reprojection_error_px: float
    dlt_condition: float
    dlt_nullspace_gap: float
    optimizer_success: bool
    optimizer_message: str


@dataclass
class ChoJointResult:
    """Stack calibration with detector intrinsics shared by all views."""

    poses: list[ChoPose]
    shared_sdd_mm: float
    shared_principal_point_px: np.ndarray
    shared_offset_mm: np.ndarray
    reprojection_rmse_px: float
    optimizer_success: bool
    optimizer_message: str


def phantom_points(config: ChoConfig) -> np.ndarray:
    """Return Cho's two-ring 3-D bead coordinates in the phantom frame."""
    n = int(config.beads_per_ring)
    if n < 6 or n % 2:
        raise ValueError("an even number of at least six beads per ring is required")
    angle = np.arange(n, dtype=float) * 2.0 * np.pi / n
    return np.array([
        [config.ring_radius_mm * np.cos(a),
         config.ring_radius_mm * np.sin(a), side * config.ring_half_spacing_mm]
        for side in (-1.0, 1.0) for a in angle
    ])


def marker_points(config: ChoConfig) -> np.ndarray:
    return np.asarray(config.marker_points_mm, dtype=float).reshape((-1, 3))


def full_phantom_points(config: ChoConfig) -> np.ndarray:
    ring = phantom_points(config)
    markers = marker_points(config)
    return np.vstack([ring, markers]) if len(markers) else ring.copy()


def _pack(k: np.ndarray, r: np.ndarray, t: np.ndarray,
          pixel_size_mm: np.ndarray) -> np.ndarray:
    sdd = 0.5 * (k[0, 0] * pixel_size_mm[0] + k[1, 1] * pixel_size_mm[1])
    return np.r_[np.log(max(sdd, EPS)), k[0, 2], k[1, 2],
                 Rotation.from_matrix(r).as_rotvec(), t]


def _unpack(parameters: np.ndarray, pixel_size_mm: np.ndarray):
    sdd = float(np.exp(parameters[0]))
    u0, v0 = parameters[1:3]
    r = Rotation.from_rotvec(parameters[3:6]).as_matrix()
    t = parameters[6:9]
    k = np.array([[sdd / pixel_size_mm[0], 0.0, u0],
                  [0.0, sdd / pixel_size_mm[1], v0],
                  [0.0, 0.0, 1.0]])
    return k, r, t


def calibrate_view(points_px: np.ndarray, config: ChoConfig,
                   points_3d_mm: np.ndarray | None = None) -> ChoPose:
    """Estimate complete geometry from one ordered 24-bead projection."""
    world = phantom_points(config) if points_3d_mm is None else np.asarray(points_3d_mm, dtype=float)
    measured = np.asarray(points_px, dtype=float)
    if measured.shape != (len(world), 2):
        raise ValueError("points_px does not match the phantom point count")
    pixel = np.asarray(config.pixel_size_mm, dtype=float)
    if pixel.shape != (2,) or np.any(pixel <= 0):
        raise ValueError("pixel_size_mm must contain two positive values")

    p0, diagnostics = normalized_dlt(world, measured)
    k0, r0, t0 = decompose_projection(p0, pixel)
    initial = _pack(k0, r0, t0, pixel)

    def residual(parameters):
        k, r, t = _unpack(parameters, pixel)
        camera = world @ r.T + t
        z = camera[:, 2]
        if np.any(z <= EPS):
            return np.full(measured.size, 1.0e6)
        predicted = np.column_stack([
            k[0, 0] * camera[:, 0] / z + k[0, 2],
            k[1, 1] * camera[:, 1] / z + k[1, 2],
        ])
        return (predicted - measured).ravel()

    result = least_squares(
        residual, initial, loss=config.robust_loss,
        f_scale=config.robust_scale_px, max_nfev=config.max_nfev,
        xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13,
    )
    k, r, t = _unpack(result.x, pixel)
    projection = k @ np.column_stack([r, t])
    predicted = project_points(world, projection)
    point_error = np.linalg.norm(predicted - measured, axis=1)
    source = -r.T @ t
    sdd = float(k[0, 0] * pixel[0])
    u_axis, v_axis, normal = r[0], r[1], r[2]
    detector_origin = source + sdd * normal
    return ChoPose(
        projection_matrix=projection,
        intrinsic_matrix=k,
        rotation_world_to_detector=r,
        translation_world_to_detector_mm=t,
        source_world_mm=source,
        source_detector_distance_mm=sdd,
        principal_point_px=np.array([k[0, 2], k[1, 2]]),
        detector_u_axis_world=u_axis,
        detector_v_axis_world=v_axis,
        detector_normal_world=normal,
        detector_origin_world_mm=detector_origin,
        reprojection_rmse_px=float(np.sqrt(np.mean((predicted - measured) ** 2))),
        max_reprojection_error_px=float(np.max(point_error)),
        dlt_condition=diagnostics["condition"],
        dlt_nullspace_gap=diagnostics["nullspace_gap"],
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )


def calibrate_stack(points_stack_px: np.ndarray, config: ChoConfig) -> list[ChoPose]:
    stack = np.asarray(points_stack_px, dtype=float)
    expected = 2 * config.beads_per_ring
    if stack.ndim != 3 or stack.shape[1:] != (expected, 2):
        raise ValueError(f"points_stack_px must have shape (views,{expected},2)")
    return [calibrate_view(frame, config) for frame in stack]


def calibrate_stack_joint(points_stack_px: np.ndarray, config: ChoConfig) -> ChoJointResult:
    """Jointly estimate fixed detector intrinsics over a projection stack.

    Each view has its own six extrinsic parameters (world-to-detector rotation
    and translation), while SDD and the principal point are shared. This is the
    appropriate model for a fixed flat panel with non-ideal source motion.
    """
    stack = np.asarray(points_stack_px, dtype=float)
    expected = 2 * config.beads_per_ring
    if stack.ndim != 3 or stack.shape[1:] != (expected, 2):
        raise ValueError(f"points_stack_px must have shape (views,{expected},2)")
    world = phantom_points(config)
    pixel = np.asarray(config.pixel_size_mm, dtype=float)
    if pixel.shape != (2,) or np.any(pixel <= 0):
        raise ValueError("pixel_size_mm must contain two positive values")

    independent = [calibrate_view(frame, config, world) for frame in stack]
    shared_sdd = float(np.median([p.source_detector_distance_mm for p in independent]))
    shared_uv = np.median(np.asarray([p.principal_point_px for p in independent]), axis=0)
    extrinsics = []
    for pose in independent:
        extrinsics.append(np.r_[Rotation.from_matrix(pose.rotation_world_to_detector).as_rotvec(),
                                pose.translation_world_to_detector_mm])
    extrinsics = np.asarray(extrinsics)
    initial = np.r_[np.log(shared_sdd), shared_uv, extrinsics.ravel()]

    # Every residual depends on the three shared intrinsics and only the six
    # extrinsics of its own view. Supplying this block structure changes the
    # finite-difference cost from O(views^2) to O(views).
    residuals_per_view = expected * 2
    jacobian_pattern = lil_matrix(
        (len(stack) * residuals_per_view, 3 + len(stack) * 6), dtype=np.int8)
    for view in range(len(stack)):
        rows = slice(view * residuals_per_view, (view + 1) * residuals_per_view)
        jacobian_pattern[rows, :3] = 1
        jacobian_pattern[rows, 3 + view * 6:3 + (view + 1) * 6] = 1
    jacobian_pattern = jacobian_pattern.tocsr()

    def unpack(parameters):
        sdd = float(np.exp(parameters[0]))
        uv = parameters[1:3]
        ext = parameters[3:].reshape(len(stack), 6)
        return sdd, uv, ext

    def residual(parameters):
        sdd, uv, ext = unpack(parameters)
        k = np.array([[sdd / pixel[0], 0.0, uv[0]],
                      [0.0, sdd / pixel[1], uv[1]],
                      [0.0, 0.0, 1.0]])
        output = []
        for frame, values in zip(stack, ext):
            r = Rotation.from_rotvec(values[:3]).as_matrix()
            t = values[3:]
            camera = world @ r.T + t
            z = camera[:, 2]
            if np.any(z <= EPS):
                output.extend(np.full(frame.size, 1.0e6))
                continue
            predicted = np.column_stack([
                k[0, 0] * camera[:, 0] / z + uv[0],
                k[1, 1] * camera[:, 1] / z + uv[1],
            ])
            output.extend((predicted - frame).ravel())
        return np.asarray(output)

    result = least_squares(
        residual, initial, loss=config.robust_loss,
        f_scale=config.robust_scale_px, max_nfev=config.max_nfev,
        jac_sparsity=jacobian_pattern,
        xtol=1.0e-12, ftol=1.0e-12, gtol=1.0e-12,
        verbose=0,
    )
    sdd, uv, ext = unpack(result.x)
    poses = []
    all_residuals = []
    k = np.array([[sdd / pixel[0], 0.0, uv[0]],
                  [0.0, sdd / pixel[1], uv[1]],
                  [0.0, 0.0, 1.0]])
    for frame, values, seed_pose in zip(stack, ext, independent):
        r = Rotation.from_rotvec(values[:3]).as_matrix()
        t = values[3:]
        p = k @ np.column_stack([r, t])
        predicted = project_points(world, p)
        point_error = np.linalg.norm(predicted - frame, axis=1)
        source = -r.T @ t
        normal = r[2]
        detector_origin = source + sdd * normal
        residual_values = (predicted - frame).ravel()
        all_residuals.extend(residual_values)
        poses.append(ChoPose(
            projection_matrix=p,
            intrinsic_matrix=k,
            rotation_world_to_detector=r,
            translation_world_to_detector_mm=t,
            source_world_mm=source,
            source_detector_distance_mm=sdd,
            principal_point_px=np.asarray(uv).copy(),
            detector_u_axis_world=r[0],
            detector_v_axis_world=r[1],
            detector_normal_world=normal,
            detector_origin_world_mm=detector_origin,
            reprojection_rmse_px=float(np.sqrt(np.mean(residual_values ** 2))),
            max_reprojection_error_px=float(np.max(point_error)),
            dlt_condition=seed_pose.dlt_condition,
            dlt_nullspace_gap=seed_pose.dlt_nullspace_gap,
            optimizer_success=bool(result.success),
            optimizer_message=str(result.message),
        ))
    return ChoJointResult(
        poses=poses,
        shared_sdd_mm=sdd,
        shared_principal_point_px=np.asarray(uv),
        shared_offset_mm=(np.asarray(uv) - np.array([511.5, 511.5])) * pixel,
        reprojection_rmse_px=float(np.sqrt(np.mean(np.asarray(all_residuals) ** 2))),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
    )
