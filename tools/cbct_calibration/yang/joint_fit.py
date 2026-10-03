"""Fixed-source-gauge global fit following Yang's per-view PIC stage.

This module is intentionally local to the Yang workflow.  PIC supplies one
projection matrix per view; their camera centres supply the source-circle
initialization.  The engineering fit then uses the same explicit gauge as the
other workflows: all source offsets and detector normal offset are zero, SID
is fixed to the fitted circle radius, and seven equivalent machine parameters
are reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..coordinates import (PhantomFrameTransform, PointCorrespondence,
                           FDK_TEST_CONVENTION,
                           convert_projection_world_and_detector,
                           transform_indexed_model_points)
from .coordinate_adapter import (YANG_PAPER_CONVENTION,
                                 YANG_PHANTOM_TO_FDK_FRAME)
from ..geometry.dlt_to_conevec import dlt_to_conevec
from .pic import YangConfig, YangPose, geometry_from_pose


FIXED_PARAMETERS = {
    "source_offset_x_mm": 0.0,
    "source_offset_y_mm": 0.0,
    "source_offset_z_mm": 0.0,
    "offset_n_mm": 0.0,
}


@dataclass(frozen=True)
class SourceCircle:
    center_phantom_mm: np.ndarray
    axis_phantom: np.ndarray
    basis_x_phantom: np.ndarray
    radius_mm: float
    radial_rms_mm: float
    axial_rms_mm: float


def phantom_points_paper(config: YangConfig) -> np.ndarray:
    """Return Yang point order in the paper's ``x,y_axis,z_ring`` frame."""
    n = int(config.beads_per_ring)
    beta = np.arange(n, dtype=float) * 2.0 * np.pi / n
    return np.asarray([
        [config.ring_radius_mm * np.cos(a), side * config.ring_half_spacing_mm,
         config.ring_radius_mm * np.sin(a)]
        for side in (-1.0, 1.0) for a in beta
    ])


def phantom_points(config: YangConfig,
                   correspondence: PointCorrespondence | None = None,
                   frame_transform: PhantomFrameTransform =
                   YANG_PHANTOM_TO_FDK_FRAME,
                   ) -> np.ndarray:
    """Return Yang targets in the canonical fdk-test 3-D phantom frame.

    All joint fitting thereafter uses this array.  The paper-frame points are
    kept only for the PIC equations, so the 3-D-to-2-D map has one explicit
    world-coordinate definition.

    The optional ``correspondence`` is applied after the physical frame
    transform.  If omitted, the points stay in canonical model order.  A
    tracker-specific correspondence must therefore be supplied explicitly by
    the workflow that owns that tracker.
    """
    return transform_indexed_model_points(
        phantom_points_paper(config), frame_transform, correspondence)


def camera_center(projection_matrix: np.ndarray) -> np.ndarray:
    p = np.asarray(projection_matrix, dtype=float)
    return -np.linalg.solve(p[:, :3], p[:, 3])


def fit_source_circle_from_dlt(projection_matrices: list[np.ndarray]) -> SourceCircle:
    """Fit the source trajectory from matrices in the canonical FDK frame.

    The PIC pose contains useful virtual-detector quantities, but its paper
    phase parameter is not a complete camera matrix.  The DLT matrix built
    from the same indexed Yang targets is therefore the authoritative ray
    geometry for the source-circle and joint-fit stages.
    """
    source = np.asarray([camera_center(p) for p in projection_matrices])
    centroid = source.mean(axis=0)
    _, _, vt = np.linalg.svd(source - centroid)
    axis = vt[-1]
    if np.dot(np.cross(source[0] - centroid, source[1] - centroid), axis) < 0.0:
        axis = -axis
    x0 = source[0] - centroid
    x0 -= axis * np.dot(x0, axis)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(axis, x0)
    xy = np.column_stack([(source - centroid) @ x0, (source - centroid) @ y0])
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    rhs = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    a, b, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = centroid - 0.5 * a * x0 - 0.5 * b * y0
    delta = source - center
    axial = delta @ axis
    planar = delta - axial[:, None] * axis
    radii = np.linalg.norm(planar, axis=1)
    return SourceCircle(
        center, axis, planar[0] / np.linalg.norm(planar[0]),
        float(radii.mean()),
        float(np.sqrt(np.mean((radii - radii.mean()) ** 2))),
        float(np.sqrt(np.mean(axial ** 2))),
    )


def fit_source_circle(
        poses: list[YangPose],
        frame_transform: PhantomFrameTransform = YANG_PHANTOM_TO_FDK_FRAME,
        ) -> SourceCircle:
    """Fit a 3-D circle to PIC camera centres in an explicit target frame.

    ``geometry_from_pose`` returns a camera matrix whose world coordinates
    are Yang's paper phantom coordinates.  The rigid transform is therefore
    applied here exactly once.  This is deliberately a parameter rather than
    a hidden global conversion: a different right-handed reconstruction frame
    must produce the same physical circle expressed in that new frame.
    """
    source = np.asarray([
        frame_transform.apply(
            camera_center(geometry_from_pose(pose).projection_matrix)[None, :]
        )[0]
        for pose in poses
    ])
    centroid = source.mean(axis=0)
    _, _, vt = np.linalg.svd(source - centroid)
    axis = vt[-1]
    if np.dot(np.cross(source[0] - centroid, source[1] - centroid), axis) < 0.0:
        axis = -axis
    x0 = source[0] - centroid
    x0 -= axis * np.dot(x0, axis)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(axis, x0)
    xy = np.column_stack([(source - centroid) @ x0, (source - centroid) @ y0])
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    rhs = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    a, b, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = centroid - 0.5 * a * x0 - 0.5 * b * y0
    delta = source - center
    axial = delta @ axis
    planar = delta - axial[:, None] * axis
    radii = np.linalg.norm(planar, axis=1)
    basis_x = planar[0] / np.linalg.norm(planar[0])
    return SourceCircle(
        center, axis, basis_x, float(radii.mean()),
        float(np.sqrt(np.mean((radii - radii.mean()) ** 2))),
        float(np.sqrt(np.mean(axial ** 2))),
    )


def _phantom_to_scanner(circle: SourceCircle) -> tuple[np.ndarray, np.ndarray]:
    axis = circle.axis_phantom.copy()
    radial = circle.basis_x_phantom.copy()
    tangent = np.cross(axis, radial)
    phantom_basis = np.column_stack([radial, tangent, axis])
    # The first nominal source position is angle -pi/2: scanner radial -Y.
    scanner_basis = np.column_stack([
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    rotation = scanner_basis @ phantom_basis.T
    return rotation, -rotation @ circle.center_phantom_mm


def _rotate(values: np.ndarray, axes: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(axes * angle).apply(values)


def _machine_axes(angles_rad: np.ndarray, tilts: np.ndarray):
    angle = np.asarray(angles_rad) - np.pi / 2.0
    radial = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(angle))])
    # fdk-test at zero degrees: source=(0,-SID,0), U=(+1,0,0),
    # V=(0,0,+1), N=(0,-1,0), hence U x V = N.  The opposite tangent used
    # here previously made the frame left-handed and forced the optimizer to
    # hide that reflection in an approximately 180-degree tilt_v.
    u_axis = np.column_stack([-np.sin(angle), np.cos(angle), np.zeros(len(angle))])
    v_axis = np.tile([0.0, 0.0, 1.0], (len(angle), 1))
    normal = radial.copy()
    v_axis, normal = _rotate(v_axis, u_axis, tilts[0]), _rotate(normal, u_axis, tilts[0])
    u_axis, normal = _rotate(u_axis, v_axis, tilts[1]), _rotate(normal, v_axis, tilts[1])
    u_axis, v_axis = _rotate(u_axis, normal, tilts[2]), _rotate(v_axis, normal, tilts[2])
    return radial, u_axis, v_axis, normal


def project(state: np.ndarray, points_mm: np.ndarray, angles_rad: np.ndarray,
            pixel_size_mm: tuple[float, float], image_center_px: np.ndarray) -> np.ndarray:
    """Project a 13-value state; fixed source/normal offsets have no slots."""
    sid, sdd, offset_u, offset_v, tilt_u, tilt_v, tilt_n = state[:7]
    world = (np.asarray(points_mm) @
             Rotation.from_rotvec(state[7:10]).as_matrix().T + state[10:13])
    radial, u_axis, v_axis, normal = _machine_axes(
        angles_rad, np.array([tilt_u, tilt_v, tilt_n]))
    source = sid * radial
    detector = -(sdd - sid) * radial + offset_u * u_axis + offset_v * v_axis
    ray = world[None, :, :] - source[:, None, :]
    scale = (np.sum((detector - source) * normal, axis=1)[:, None] /
             np.sum(ray * normal[:, None, :], axis=2))
    hit = source[:, None, :] + scale[:, :, None] * ray
    relative = hit - detector[:, None, :]
    pitch = np.asarray(pixel_size_mm, dtype=float)
    return np.stack([
        image_center_px[0] + np.sum(relative * u_axis[:, None, :], axis=2) / pitch[0],
        image_center_px[1] + np.sum(relative * v_axis[:, None, :], axis=2) / pitch[1],
    ], axis=2)


def _machine_seeds_from_pic(poses: list[YangPose], rotation: np.ndarray,
                            translation: np.ndarray, circle: SourceCircle,
                            image_center_px: np.ndarray, frame_ids: np.ndarray,
                            total_views: int,
                            image_shape: tuple[int, int],
                            projection_matrices: list[np.ndarray] | None = None,
                            frame_transform: PhantomFrameTransform =
                            YANG_PHANTOM_TO_FDK_FRAME) -> dict[int, np.ndarray]:
    """Fit machine-only seeds to detector planes extracted from the PIC P_i."""
    scanner_to_phantom = np.eye(4)
    scanner_to_phantom[:3, :3] = rotation.T
    scanner_to_phantom[:3, 3] = -rotation.T @ translation
    detector_centers, u_measured, v_measured = [], [], []
    for index, pose in enumerate(poses):
        # One canonical conversion: the matrix now consumes canonical
        # fdk-test phantom coordinates and produces raw fdk-test pixels.  The
        # scanner-gauge rigid transform is a separate world-frame operation,
        # not another detector/image-axis conversion.
        if projection_matrices is None:
            projection_fdk = convert_projection_world_and_detector(
                geometry_from_pose(pose).projection_matrix, image_shape,
                phantom_source_to_target=frame_transform.rotation,
                phantom_translation=frame_transform.translation,
                source_detector=YANG_PAPER_CONVENTION,
                target_detector=FDK_TEST_CONVENTION)
        else:
            projection_fdk = np.asarray(projection_matrices[index], dtype=float)
        projection_scanner = projection_fdk @ scanner_to_phantom
        cone = dlt_to_conevec(
            projection_scanner, sdd_mm=pose.source_detector_distance_mm,
            point_toward=np.zeros(3))
        ru, rv = cone.r_u, cone.r_v
        detector_centers.append(cone.det_s + image_center_px[0] * ru
                                + image_center_px[1] * rv)
        u_measured.append(ru / np.linalg.norm(ru))
        v_measured.append(rv / np.linalg.norm(rv))
    detector_centers = np.asarray(detector_centers)
    u_measured, v_measured = np.asarray(u_measured), np.asarray(v_measured)

    seeds = {}
    sdd0 = float(np.median([pose.source_detector_distance_mm for pose in poses]))
    for direction in (1, -1):
        angles = direction * frame_ids * 2.0 * np.pi / total_views
        def residual(values):
            sdd, offset_u, offset_v = values[:3]
            radial, u_axis, v_axis, _ = _machine_axes(angles, values[3:6])
            detector = (-(sdd - circle.radius_mm) * radial +
                        offset_u * u_axis + offset_v * v_axis)
            # Axes are dimensionless; multiplying them by 100 mm balances
            # their angular information against detector-centre positions.
            return np.r_[(detector - detector_centers).ravel(),
                         100.0 * (u_axis - u_measured).ravel(),
                         100.0 * (v_axis - v_measured).ravel()]

        solved = least_squares(
            residual, np.array([sdd0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            x_scale=np.array([100.0, 10.0, 10.0, 0.1, 0.1, 0.1]),
            max_nfev=500)
        seeds[direction] = solved.x
    return seeds


def fit_fixed_source(points_px: np.ndarray, poses: list[YangPose], config: YangConfig,
                     image_shape: tuple[int, int] = (1024, 1024),
                     frame_ids: np.ndarray | None = None, total_views: int | None = None,
                     max_nfev: int = 1200,
                     projection_matrices: list[np.ndarray] | None = None,
                     correspondence: PointCorrespondence | None = None,
                     frame_transform: PhantomFrameTransform =
                     YANG_PHANTOM_TO_FDK_FRAME) -> dict:
    """Estimate seven machine values in fdk-test image coordinates.

    ``points_px`` use u-right/v-down. ``poses`` remain native Yang PIC poses;
    their matrices are converted at the mechanical-model boundary.
    """
    measured = np.asarray(points_px, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if measured.ndim != 3 or measured.shape[1:] != (expected, 2):
        raise ValueError(
            f"points_px must have shape (views,{expected},2)")
    if len(poses) != len(measured):
        raise ValueError("poses and points_px must contain the same views")
    ids = np.arange(len(measured)) if frame_ids is None else np.asarray(frame_ids)
    period = len(measured) if total_views is None else int(total_views)
    if ids.shape != (len(measured),) or period <= int(np.max(ids)):
        raise ValueError("frame_ids/total_views do not describe the point stack")

    if projection_matrices is None:
        circle = fit_source_circle(poses, frame_transform)
    else:
        if len(projection_matrices) != len(measured):
            raise ValueError("projection_matrices length does not match points_px")
        circle = fit_source_circle_from_dlt(projection_matrices)
    rotation0, translation0 = _phantom_to_scanner(circle)
    center = np.array([(image_shape[1] - 1) / 2.0, (image_shape[0] - 1) / 2.0])
    pixel = np.asarray(config.pixel_size_mm, dtype=float)
    machine_seeds = _machine_seeds_from_pic(
        poses, rotation0, translation0, circle, center, ids, period,
        image_shape, projection_matrices, frame_transform)
    initial = np.r_[circle.radius_mm, machine_seeds[1],
                    Rotation.from_matrix(rotation0).as_rotvec(), translation0]
    free = np.arange(1, 13)  # SID is fixed to the source-circle radius.
    scales = np.array([100.0, 2.0, 2.0, 0.02, 0.02, 0.02,
                       0.03, 0.03, 0.03, 10.0, 10.0, 10.0])
    points = phantom_points(config, correspondence, frame_transform)
    candidates = []
    for direction in (1, -1):
        angles = direction * ids * 2.0 * np.pi / period
        initial[1:7] = machine_seeds[direction]

        def residual(values):
            state = initial.copy()
            state[free] = values
            return (project(state, points, angles, config.pixel_size_mm, center)
                    - measured).ravel()

        lower = np.full(len(free), -np.inf)
        upper = np.full(len(free), np.inf)
        # The mechanical detector is calibrated around its nominal pose.  A
        # 180-degree Euler branch is projectively equivalent but is not the
        # requested physical parameterization.
        for state_index in (4, 5, 6):
            local_index = int(np.flatnonzero(free == state_index)[0])
            lower[local_index] = -np.deg2rad(45.0)
            upper[local_index] = np.deg2rad(45.0)
        lower[0] = 100.0
        upper[0] = 2000.0
        x0 = np.clip(initial[free], lower + 1.0e-10, upper - 1.0e-10)
        coarse = least_squares(
            residual, x0, x_scale=scales, bounds=(lower, upper),
            loss="linear", max_nfev=max(200, max_nfev // 2),
            ftol=1e-9, xtol=1e-9, gtol=1e-9)
        solved = least_squares(
            residual, coarse.x, x_scale=scales, bounds=(lower, upper),
            loss="soft_l1", f_scale=0.15, max_nfev=max_nfev,
            ftol=1e-12, xtol=1e-12, gtol=1e-12)
        state = initial.copy()
        state[free] = solved.x
        error = residual(solved.x).reshape(measured.shape)
        candidates.append((float(np.sqrt(np.mean(error * error))), direction,
                           state, solved, error))
    rmse, direction, state, solved, error = min(candidates, key=lambda item: item[0])
    return {
        "machine": {
            "sid_mm": float(state[0]), "sdd_mm": float(state[1]),
            "offset_u_mm": float(state[2]), "offset_v_mm": float(state[3]),
            "offset_u_px": float(state[2] / pixel[0]),
            "offset_v_px": float(state[3] / pixel[1]),
            "tilt_u_deg": float(np.degrees(state[4])),
            "tilt_v_deg": float(np.degrees(state[5])),
            "tilt_n_deg": float(np.degrees(state[6])), **FIXED_PARAMETERS,
        },
        "phantom_pose": {
            "rotation_vector_deg": np.degrees(state[7:10]).tolist(),
            "translation_mm": state[10:13].tolist(),
        },
        "source_circle": {
            "center_phantom_mm": circle.center_phantom_mm.tolist(),
            "axis_phantom": circle.axis_phantom.tolist(),
            "radius_mm": circle.radius_mm,
            "radial_rms_mm": circle.radial_rms_mm,
            "axial_rms_mm": circle.axial_rms_mm,
        },
        "diagnostics": {
            "rmse_px": rmse,
            "max_point_error_px": float(np.max(np.linalg.norm(error, axis=2))),
            "rotation_direction": int(direction), "success": bool(solved.success),
            "nfev": int(solved.nfev),
            "gauge": "source offsets and detector normal offset fixed to zero",
            "pic_virtual_detector_note": (
                "Yang PIC determines rays in its virtual detector frame. The reported "
                "machine values are an equivalent fixed-source gauge; mechanical offsets "
                "require an explicit virtual-to-real detector convention."
            ),
        },
    }
