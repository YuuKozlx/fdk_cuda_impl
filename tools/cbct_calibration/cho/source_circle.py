"""Convert indexed Cho-phantom observations into system-level CBCT geometry.

The first-stage correspondence map is assumed known.  Per-view projective
poses are estimated, the recovered source trajectory defines the physical
rotation axis/circle, and detector pose is then expressed in the rotating
radial/tangential/axial frame.  This separates phantom placement from fixed
scanner errors such as detector tilt and detector-center offset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .dlt import ChoPose


EPS = 1e-12


@dataclass
class Circle3D:
    center_world_mm: np.ndarray
    axis_world: np.ndarray
    basis_x_world: np.ndarray
    basis_y_world: np.ndarray
    radius_mm: float
    radial_rms_mm: float
    axial_rms_mm: float


@dataclass
class SystemGeometry:
    source_circle: Circle3D
    sdd_mm: float
    sid_mm: float
    source_radial_offset_from_nominal_mm: float | None
    source_radial_rms_mm: float
    source_axial_rms_mm: float
    source_tangential_rms_mm: float
    detector_tilt_rotvec_local_deg: np.ndarray
    detector_tilt_u_deg: float
    detector_tilt_v_deg: float
    detector_in_plane_deg: float
    detector_rotation_rms_deg: float
    detector_center_local_mean_mm: np.ndarray
    detector_center_local_std_mm: np.ndarray
    detector_radial_position_ideal_mm: float
    detector_radial_offset_mm: float
    detector_tangential_offset_mm: float
    detector_axial_offset_mm: float
    principal_point_mean_px: np.ndarray
    principal_point_std_px: np.ndarray
    piercing_displacement_from_image_center_px: np.ndarray
    mechanical_center_offset_local_uv_mm: np.ndarray
    reprojection_rmse_mean_px: float
    reprojection_rmse_max_px: float
    per_view: list[dict]


def fit_circle_3d(points: np.ndarray) -> Circle3D:
    """Least-squares circle in a best-fit 3-D plane."""
    p = np.asarray(points, dtype=float)
    centroid = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - centroid)
    axis = vt[-1]
    # Orient axis so increasing frame index has positive angular direction.
    if len(p) > 2 and np.dot(np.cross(p[0] - centroid, p[1] - centroid), axis) < 0:
        axis = -axis
    x0 = p[0] - centroid
    x0 -= axis * np.dot(x0, axis)
    x0 /= np.linalg.norm(x0)
    y0 = np.cross(axis, x0)
    xy = np.column_stack([(p - centroid) @ x0, (p - centroid) @ y0])
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    rhs = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    a, b, c = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center2 = np.array([-a / 2.0, -b / 2.0])
    center = centroid + center2[0] * x0 + center2[1] * y0
    delta = p - center
    axial = delta @ axis
    planar = delta - axial[:, None] * axis
    radii = np.linalg.norm(planar, axis=1)
    radius = float(np.mean(radii))
    # Re-anchor phase basis at the first view around the fitted center.
    bx = planar[0] / max(np.linalg.norm(planar[0]), EPS)
    by = np.cross(axis, bx)
    return Circle3D(
        center_world_mm=center,
        axis_world=axis,
        basis_x_world=bx,
        basis_y_world=by,
        radius_mm=radius,
        radial_rms_mm=float(np.sqrt(np.mean((radii - radius) ** 2))),
        axial_rms_mm=float(np.sqrt(np.mean(axial ** 2))),
    )


def _wrap_line_angle_deg(angle: float) -> float:
    return float((angle + 90.0) % 180.0 - 90.0)


def analyze_poses(poses: list[ChoPose], image_shape: tuple[int, int],
                  pixel_size_mm: tuple[float, float],
                  nominal_sid_mm: float | None = None) -> SystemGeometry:
    if len(poses) < 6:
        raise ValueError("at least six poses are required")
    sources = np.asarray([p.source_world_mm for p in poses])
    circle = fit_circle_3d(sources)
    axis, center = circle.axis_world, circle.center_world_mm
    # The simulator's positive axial direction is +Z.  The fitted circle
    # normal is otherwise sign-ambiguous; fixing it here makes all reported
    # detector angles reproducible.
    if axis[2] < 0.0:
        axis = -axis
        circle.axis_world = axis
        circle.basis_y_world = np.cross(axis, circle.basis_x_world)
    sdd_values = np.asarray([p.source_detector_distance_mm for p in poses])
    sdd = float(np.median(sdd_values))
    pu, pv = pixel_size_mm
    image_center = np.array([(image_shape[1] - 1) / 2.0,
                             (image_shape[0] - 1) / 2.0])

    local_rotations = []
    detector_local = []
    principal = []
    per_view = []
    src_radial_residual = []
    src_axial_residual = []
    src_tangent_residual = []
    detector_centers_world = []

    # The source can have a fixed tangential offset.  Therefore its fitted
    # circle center is not necessarily the scanner isocenter.  Recover the
    # detector-center trajectory first; its circle center is the correct
    # radial reference for detector attitude.
    for pose in poses:
        pp = pose.principal_point_px
        # Cho's camera decomposition stores the normal from source toward the
        # detector.  The simulator's N axis points from detector toward source.
        piercing_world = (pose.source_world_mm +
                          pose.source_detector_distance_mm * pose.detector_normal_world)
        detector_centers_world.append(
            piercing_world +
            (image_center[0] - pp[0]) * pu * pose.detector_u_axis_world +
            (image_center[1] - pp[1]) * pv * pose.detector_v_axis_world)
    detector_centers_world = np.asarray(detector_centers_world)
    detector_circle = fit_circle_3d(detector_centers_world)
    orientation_center = detector_circle.center_world_mm
    previous_angle = None
    unwrapped_angle = 0.0

    for i, pose in enumerate(poses):
        delta = pose.source_world_mm - orientation_center
        axial_value = float(np.dot(delta, axis))
        planar = delta - axial_value * axis
        radial_distance = float(np.linalg.norm(planar))
        er = planar / max(radial_distance, EPS)
        et = np.cross(axis, er)
        # The simulator now uses a right-handed detector frame U x V = N.
        # Its N axis points from the panel toward the source.
        ideal = np.column_stack([et, axis, er])
        actual = np.column_stack([pose.detector_u_axis_world,
                                  pose.detector_v_axis_world,
                                  -pose.detector_normal_world])
        r_local = ideal.T @ actual
        # Project numerical drift to SO(3).
        uu, _, vv = np.linalg.svd(r_local)
        r_local = uu @ vv
        if np.linalg.det(r_local) < 0:
            uu[:, -1] *= -1
            r_local = uu @ vv
        local_rotations.append(r_local)

        pp = pose.principal_point_px
        principal.append(pp)
        detector_center_world = detector_centers_world[i]
        det_delta = detector_center_world - orientation_center
        det_local = np.array([np.dot(det_delta, er),
                              np.dot(det_delta, et),
                              np.dot(det_delta, axis)])
        detector_local.append(det_local)

        source_delta = pose.source_world_mm - center
        source_axial = float(np.dot(source_delta, axis))
        source_planar = source_delta - source_axial * axis
        source_er = source_planar / max(np.linalg.norm(source_planar), EPS)
        raw_angle = float(np.arctan2(np.dot(source_er, circle.basis_y_world),
                                     np.dot(source_er, circle.basis_x_world)))
        if previous_angle is None:
            unwrapped_angle = raw_angle
        else:
            step = (raw_angle - previous_angle + np.pi) % (2 * np.pi) - np.pi
            unwrapped_angle += step
        previous_angle = raw_angle

        src_radial_residual.append(float(np.linalg.norm(source_planar)) - circle.radius_mm)
        src_axial_residual.append(source_axial)
        # With angle chosen from the measured source, tangential residual is
        # identically the phase residual relative to a uniform angle sequence.
        # The calibrated source trajectory in this detector convention runs
        # clockwise as the frame index increases.
        expected = -2.0 * np.pi * i / len(poses)
        phase_error = (unwrapped_angle - expected + np.pi) % (2 * np.pi) - np.pi
        tangential = circle.radius_mm * phase_error
        src_tangent_residual.append(tangential)
        rv = Rotation.from_matrix(r_local).as_rotvec()
        per_view.append({
            "frame": i,
            "angle_deg": float(np.degrees(unwrapped_angle)),
            "source_world_mm": pose.source_world_mm.tolist(),
            "source_local_rta_mm": [radial_distance, tangential, axial_value],
            "detector_center_local_rta_mm": det_local.tolist(),
            "detector_rotation_local_rotvec_deg": np.degrees(rv).tolist(),
            "principal_point_px": pp.tolist(),
            "sdd_mm": float(pose.source_detector_distance_mm),
            "reprojection_rmse_px": float(pose.reprojection_rmse_px),
        })

    mean_rotation = Rotation.from_matrix(np.asarray(local_rotations)).mean()
    mean_rotvec_deg = np.degrees(mean_rotation.as_rotvec())
    rotation_residual = (mean_rotation.inv() *
                         Rotation.from_matrix(np.asarray(local_rotations))).magnitude()
    detector_local = np.asarray(detector_local)
    principal = np.asarray(principal)
    detector_mean = detector_local.mean(axis=0)
    ideal_radial = circle.radius_mm - sdd

    # Mechanical center relative to piercing point, expressed in the actual
    # detector U/V axes.  This is the sign-correct conversion from principal
    # point displacement to physical detector-center offset.
    pp_mean = principal.mean(axis=0)
    mechanical_uv = np.array([(image_center[0] - pp_mean[0]) * pu,
                              (image_center[1] - pp_mean[1]) * pv])
    return SystemGeometry(
        source_circle=circle,
        sdd_mm=sdd,
        sid_mm=circle.radius_mm,
        source_radial_offset_from_nominal_mm=(None if nominal_sid_mm is None
                                              else circle.radius_mm - nominal_sid_mm),
        source_radial_rms_mm=float(np.sqrt(np.mean(np.square(src_radial_residual)))),
        source_axial_rms_mm=float(np.sqrt(np.mean(np.square(src_axial_residual)))),
        source_tangential_rms_mm=float(np.sqrt(np.mean(np.square(src_tangent_residual)))),
        detector_tilt_rotvec_local_deg=mean_rotvec_deg,
        detector_tilt_u_deg=float(mean_rotvec_deg[0]),
        detector_tilt_v_deg=float(mean_rotvec_deg[1]),
        detector_in_plane_deg=_wrap_line_angle_deg(float(mean_rotvec_deg[2])),
        detector_rotation_rms_deg=float(np.degrees(np.sqrt(np.mean(rotation_residual ** 2)))),
        detector_center_local_mean_mm=detector_mean,
        detector_center_local_std_mm=detector_local.std(axis=0),
        detector_radial_position_ideal_mm=ideal_radial,
        detector_radial_offset_mm=float(detector_mean[0] - ideal_radial),
        detector_tangential_offset_mm=float(detector_mean[1]),
        detector_axial_offset_mm=float(detector_mean[2]),
        principal_point_mean_px=pp_mean,
        principal_point_std_px=principal.std(axis=0),
        piercing_displacement_from_image_center_px=pp_mean - image_center,
        mechanical_center_offset_local_uv_mm=mechanical_uv,
        reprojection_rmse_mean_px=float(np.mean([p.reprojection_rmse_px for p in poses])),
        reprojection_rmse_max_px=float(np.max([p.reprojection_rmse_px for p in poses])),
        per_view=per_view,
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


'''Legacy command-line wrapper; call ``analyze_poses`` from Python instead.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points", type=Path, help="indexed (N,24,2) NumPy array")
    parser.add_argument("--output", type=Path, default=Path("cho_system_geometry.json"))
    parser.add_argument("--nominal-sid", type=float, default=None)
    parser.add_argument("--max-nfev", type=int, default=800)
    args = parser.parse_args()
    points = np.load(args.points)
    cfg = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                    beads_per_ring=12, pixel_size_mm=(0.417, 0.417),
                    robust_scale_px=0.15, max_nfev=args.max_nfev)
    poses = [calibrate_view(frame, cfg) for frame in points]
    result = analyze_poses(poses, (1024, 1024), cfg.pixel_size_mm,
                           nominal_sid_mm=args.nominal_sid)
    args.output.write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    summary = _jsonable(result)
    summary.pop("per_view", None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
'''
