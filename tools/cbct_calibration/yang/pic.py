"""Pose-independent CBCT calibration following Yang et al. (2017).

This module is deliberately independent from the earlier single-rod/ellipse
calibrator in this directory.  It implements the six pose-by-pose PIC steps
described in Yang et al., *Med. Phys.* 2017, doi:10.1002/mp.12163:

1. locate O, the projection of the phantom centre W, from crossed bead lines;
2. estimate detector yaw from two same-ring bead lines;
3. fit the two ring ellipses;
4. solve roll and the source/phantom-centre locations in the virtual detector
   plane;
5. solve pitch by the ellipse-consistency objective;
6. optionally solve the marked-bead gantry angle.

The required bead order is ``[E1..E6, F1..F6]``. E and F are the two
parallel circular patterns, each containing six equally spaced beads. The
input coordinates are detector-bin coordinates ``[u, v]`` in Yang's paper
detector convention. The workflow owns conversion to and from the public
fdk-test convention; direct callers must use
``tools.cbct_calibration.coordinates.convert_detector_points`` explicitly.

The implementation keeps both detector-bin and physical-mm quantities in the
result.  The point O is *not* the principal point: it is the projection of
the phantom centre W used by the PIC construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.linalg import eig
from scipy.optimize import least_squares, minimize_scalar
from scipy.sparse import lil_matrix


EPS = 1.0e-12


@dataclass
class Ellipse:
    """An ellipse represented by ``(p-c)^T Q (p-c) = 1`` in mm."""

    center_mm: np.ndarray
    q_mm_inv2: np.ndarray
    conic: np.ndarray
    condition: float

    def value(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=float)
        d = points - self.center_mm
        return np.einsum("...i,ij,...j->...", d, self.q_mm_inv2, d) - 1.0

    def x_axis_intersections(self, x_mm: float = 0.0) -> np.ndarray:
        """Return the two y values where the ellipse meets x = ``x_mm``."""
        q = self.q_mm_inv2
        dx = float(x_mm - self.center_mm[0])
        # q22*y^2 + 2*q12*dx*y + q11*dx^2 - 1 = 0 after using y-c_y.
        aa = float(q[1, 1])
        bb = float(2.0 * q[0, 1] * dx)
        cc = float(q[0, 0] * dx * dx - 1.0)
        disc = bb * bb - 4.0 * aa * cc
        if aa <= 0.0 or disc < -1.0e-9:
            raise ValueError("ellipse does not intersect the requested axis")
        roots = np.sqrt(max(0.0, disc))
        # y-c_y = (-b +/- sqrt(discriminant))/(2a)
        return np.sort(np.array([
            self.center_mm[1] + (-bb - roots) / (2.0 * aa),
            self.center_mm[1] + (-bb + roots) / (2.0 * aa),
        ]))


@dataclass
class YangConfig:
    """Known phantom and detector quantities."""

    ring_radius_mm: float
    ring_half_spacing_mm: float
    pixel_size_mm: tuple[float, float] = (1.0, 1.0)
    beads_per_ring: int = 6
    pitch_search_deg: tuple[float, float] = (-89.0, 89.0)
    pitch_samples: int = 361
    phase_samples: int = 72
    estimate_gantry_angle: bool = True
    # ``paper_linear`` implements Yang Eq. (26).  ``periodic_fit`` is kept as
    # a diagnostic fallback for degenerate/noisy views.
    gantry_solver: str = "paper_linear"


@dataclass
class YangPose:
    """All parameters recovered for one projection pose."""

    o_px: np.ndarray
    o_mm_origin: np.ndarray
    d_px: np.ndarray
    d_mm: np.ndarray
    yaw_rad: float
    roll_rad: float
    pitch_rad: float
    source_i_mm: np.ndarray
    phantom_center_i_mm: np.ndarray
    source_detector_distance_mm: float
    source_phantom_distance_mm: float
    source_axis_distance_mm: float
    principal_point_px: np.ndarray
    reprojection_rmse_px: float | None
    gantry_angle_rad: float | None
    ring_ellipses: tuple[Ellipse, Ellipse]
    points_yaw0_mm: np.ndarray
    predicted_points_yaw0_mm: np.ndarray | None
    reprojection_rmse_mm: float | None
    diagnostics: dict
    pixel_size_mm: np.ndarray


@dataclass
class ViewGeometry:
    """Per-view cone-beam geometry in a detector-centered Cartesian frame.

    The source and detector frame is centred at O (the projection of phantom
    centre). ``projection_matrix`` maps phantom-coordinate homogeneous points
    (whose origin is W and whose y axis is the phantom axis) to detector-bin
    homogeneous coordinates for this view.
    """

    source_mm: np.ndarray
    detector_origin_mm: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    normal: np.ndarray
    principal_point_px: np.ndarray
    projection_matrix: np.ndarray
    pixel_size_mm: np.ndarray


@dataclass
class YangExtendedResult:
    """Multi-view result for Yang's fixed-source/fixed-detector extension."""

    poses: list[YangPose]
    source_grid_mean_mm: np.ndarray
    source_grid_std_before_mm: np.ndarray
    source_grid_std_after_mm: np.ndarray
    reprojection_rmse_before_px: float
    reprojection_rmse_after_px: float
    optimizer_success: bool
    optimizer_message: str
    optimizer_nfev: int
    source_constraint_weight: float


def geometry_from_pose(pose: YangPose, pixel_size_mm=None) -> ViewGeometry:
    """Convert a PIC pose into the per-view geometry needed by reconstruction.

    The matrix includes phantom-to-virtual-detector rotation by the recovered
    gantry phase, virtual-to-real detector rotation (roll/pitch/yaw), and
    perspective projection onto the detector plane. It maps a common phantom
    coordinate point directly to image-bin coordinates for this view. The
    per-view phantom coordinates rotate with the scanned object; this is the
    natural object-fixed reconstruction frame.
    """
    pitch = _as_pitch(pixel_size_mm if pixel_size_mm is not None else pose.pixel_size_mm)
    theta, phi, yaw = pose.roll_rad, pose.pitch_rad, pose.yaw_rad
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    roll = np.array([[1., 0., 0.], [0., ct, st], [0., -st, ct]])
    pitch_m = np.array([[cp, 0., -sp], [0., 1., 0.], [sp, 0., cp]])
    to_yaw0 = pitch_m @ roll
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_m = np.array([[c, -s], [s, c]])
    # yaw0 = yaw_m @ real, hence real = yaw_m.T @ yaw0.
    to_real = np.eye(3)
    to_real[:2, :2] = yaw_m.T
    source = to_real @ (to_yaw0 @ pose.source_i_mm)
    if pose.gantry_angle_rad is None:
        raise ValueError("gantry angle is required for a reconstruction projection matrix")
    pu, pv = pitch
    u0, v0 = pose.o_px
    sz = float(source[2])
    if abs(sz) < EPS:
        raise ValueError("source is on or too close to the detector plane")
    # Pixel projection matrix in the detector-centred frame. Denominator is
    # sz-z, matching the ray intersection used by _transform_i_to_detector.
    P_detector = np.array([
        [sz / pu, 0., -source[0] / pu - u0, u0 * sz],
        [0., sz / pv, -source[1] / pv - v0, v0 * sz],
        [0., 0., -1., sz],
    ])
    # Eq. (24) parameterizes labelled ring beads with angle alpha=-t-beta.
    # Consequently the object-fixed x/z coordinates use a rotation followed
    # by the paper's clockwise ring-index convention (a reflection in z).
    t = pose.gantry_angle_rad
    ct, st = np.cos(t), np.sin(t)
    phantom_to_i = np.array([
        [ct, 0., -st, pose.phantom_center_i_mm[0]],
        [0., 1., 0., pose.phantom_center_i_mm[1]],
        [-st, 0., -ct, pose.phantom_center_i_mm[2]],
        [0., 0., 0., 1.],
    ])
    i_to_detector = np.eye(4)
    i_to_detector[:3, :3] = to_real @ to_yaw0
    P = P_detector @ i_to_detector @ phantom_to_i
    return ViewGeometry(
        source_mm=source,
        detector_origin_mm=np.zeros(3),
        u_axis=np.array([1., 0., 0.]),
        v_axis=np.array([0., 1., 0.]),
        normal=np.array([0., 0., 1.]),
        principal_point_px=np.asarray(pose.principal_point_px, dtype=float),
        projection_matrix=P,
        pixel_size_mm=pitch,
    )


def project_with_geometry(points_mm: np.ndarray, geometry: ViewGeometry) -> np.ndarray:
    """Project detector-frame 3-D points using a :class:`ViewGeometry`."""
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N, 3)")
    h = np.column_stack([points, np.ones(len(points))]) @ geometry.projection_matrix.T
    if np.any(np.abs(h[:, 2]) < EPS):
        raise ValueError("point projects at infinity")
    return h[:, :2] / h[:, 2, None]


def calibrate_stack_geometries(points_stack_px: np.ndarray,
                               config: YangConfig) -> tuple[list[YangPose], list[ViewGeometry]]:
    """Calibrate all views and return reconstruction-ready view geometries."""
    poses = calibrate_stack(points_stack_px, config)
    geometries = [geometry_from_pose(pose, config.pixel_size_mm) for pose in poses]
    return poses, geometries


def geometry_to_dict(geometry: ViewGeometry) -> dict:
    """Serialize a :class:`ViewGeometry` for a reconstruction configuration."""
    return _jsonable(asdict(geometry))


def _as_points(points_px: np.ndarray) -> np.ndarray:
    points = np.asarray(points_px, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2
            or points.shape[0] < 12 or points.shape[0] % 4 != 0
            or not np.isfinite(points).all()):
        raise ValueError(
            "points_px must contain two equal rings with an even number "
            "of at least six finite points per ring")
    return points


def _as_pitch(pixel_size_mm) -> np.ndarray:
    pitch = np.asarray(pixel_size_mm, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pixel_size_mm must contain two positive values")
    return pitch


def _homogeneous_line(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return np.cross(np.r_[p1, 1.0], np.r_[p2, 1.0])


def line_intersection(p1: np.ndarray, p2: np.ndarray,
                      p3: np.ndarray, p4: np.ndarray,
                      *, parallel_tol: float = 1.0e-12) -> np.ndarray:
    """Intersection of two 2-D lines, with an explicit parallel error."""
    l1 = _homogeneous_line(np.asarray(p1, float), np.asarray(p2, float))
    l2 = _homogeneous_line(np.asarray(p3, float), np.asarray(p4, float))
    p = np.cross(l1, l2)
    if abs(p[2]) <= parallel_tol * max(np.linalg.norm(l1) * np.linalg.norm(l2), 1.0):
        raise ValueError("two calibration lines are parallel or nearly parallel")
    return p[:2] / p[2]


def _line_intersection_or_lstsq(p1, p2, p3, p4) -> tuple[np.ndarray, float]:
    """Return intersection and a parallelism diagnostic for one bead pair."""
    l1 = _homogeneous_line(np.asarray(p1, float), np.asarray(p2, float))
    l2 = _homogeneous_line(np.asarray(p3, float), np.asarray(p4, float))
    p = np.cross(l1, l2)
    scale = max(np.linalg.norm(l1) * np.linalg.norm(l2), 1.0)
    if abs(p[2]) > 1.0e-12 * scale:
        return p[:2] / p[2], 0.0
    # Parallel/coincident lines do not identify a unique O.
    raise ValueError("crossed bead lines do not identify a finite phantom projection O")


def _intersect_line_family(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares intersection of normalized homogeneous image lines."""
    values = np.asarray(lines, dtype=float)
    norms = np.linalg.norm(values[:, :2], axis=1)
    if values.ndim != 2 or values.shape[1] != 3 or np.any(norms < EPS):
        raise ValueError("invalid homogeneous line family")
    normalized = values / norms[:, None]
    point, _, rank, singular = np.linalg.lstsq(
        normalized[:, :2], -normalized[:, 2], rcond=None)
    if rank < 2 or not np.isfinite(point).all():
        raise ValueError("line family is parallel or rank deficient")
    distances = normalized[:, :2] @ point + normalized[:, 2]
    condition = float(singular[0] / max(singular[-1], EPS))
    return point, distances, condition


def estimate_o(points_px: np.ndarray) -> tuple[np.ndarray, dict]:
    """Step 1: estimate O from every crossed opposing line pair."""
    p = _as_points(points_px)
    n = len(p) // 2
    opposite = n // 2
    # Generalizes (E1,F4) x (E4,F1) from six to any even ring size.
    pairs = [(j, n + j + opposite, j + opposite, n + j)
             for j in range(opposite)]
    intersections, residuals, lines = [], [], []
    for a, b, c, d in pairs:
        x, residual = _line_intersection_or_lstsq(p[a], p[b], p[c], p[d])
        intersections.append(x)
        residuals.append(residual)
        lines.extend([_homogeneous_line(p[a], p[b]),
                      _homogeneous_line(p[c], p[d])])
    intersections = np.asarray(intersections)
    # The paper averages the three minimal pair intersections. With more
    # targets, the same concurrency constraint is better solved at once.
    o, line_distances, line_condition = _intersect_line_family(lines)
    scatter = np.linalg.norm(intersections - o, axis=1)
    if not np.isfinite(o).all() or np.max(scatter) > 0.1 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0):
        raise ValueError("crossed line intersections do not agree on O")
    if max(residuals) > 1.0e-4 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0):
        raise ValueError("at least one opposing line pair is effectively parallel")
    return o, {
        "cross_intersections_px": intersections,
        "cross_intersection_scatter_px": scatter,
        "cross_parallel_residual": residuals,
        "all_line_residual_px": line_distances,
        "all_line_condition": line_condition,
    }


def estimate_yaw(points_px: np.ndarray, o_px: np.ndarray,
                 pixel_size_mm: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Step 2: estimate yaw and rotate measurements into the yaw-zero frame."""
    p = _as_points(points_px)
    o = np.asarray(o_px, float)
    n = len(p) // 2
    opposite = n // 2
    generatrices = np.asarray([
        _homogeneous_line(p[j], p[n + j]) for j in range(n)])
    try:
        d_px, d_residuals, d_condition = _intersect_line_family(generatrices)
        d_mm = (d_px - o) * pixel_size_mm
        # Eq. (4): angle between OD and the detector y axis.
        yaw = float(np.arctan2(d_mm[0], d_mm[1]))
        yaw_method = "line_intersection_D"
    except ValueError:
        # Yang et al. explicitly give this branch: if E1F1 and E4F4 are
        # parallel, their common direction determines yaw relative to yI.
        directions = (p[n:] - p[:n]) * pixel_size_mm
        covariance = directions.T @ directions
        line = np.linalg.eigh(covariance)[1][:, -1]
        if np.linalg.norm(line) < 1.0e-12:
            raise ValueError("cannot estimate yaw from a zero-length projected bead line")
        yaw = float(np.arctan2(line[0], line[1]))
        d_mm = np.full(2, np.nan)
        d_px = np.full(2, np.nan)
        yaw_method = "parallel_line_direction"
        d_residuals = np.full(n, np.nan)
        d_condition = np.inf
    yaw = (yaw + np.pi/2) % np.pi - np.pi/2
    c, s = np.cos(yaw), np.sin(yaw)
    # Passive yaw of the detector measurements: column vectors are R(+yaw).
    rotate = np.array([[c, -s], [s, c]])
    relative_mm = (p - o) * pixel_size_mm
    yaw0_mm = relative_mm @ rotate.T
    d_yaw0 = (d_px - o) * pixel_size_mm @ rotate.T
    return yaw, yaw0_mm, d_yaw0, {
        "d_px": d_px,
        "d_mm": d_mm,
        "method": yaw_method,
        "all_line_residual_px": d_residuals,
        "all_line_condition": d_condition,
        "rotation_matrix_column": rotate,
    }


def fit_ellipse(points_mm: np.ndarray) -> Ellipse:
    """Fit a general ellipse using a normalized constrained conic fit."""
    p = np.asarray(points_mm, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 5 or not np.isfinite(p).all():
        raise ValueError("ellipse fit needs at least five finite 2-D points")
    mean = p.mean(axis=0)
    scale = p.std(axis=0)
    if np.any(scale < 1.0e-12):
        raise ValueError("ellipse points are collinear")
    q = (p - mean) / scale
    x, y = q.T
    design = np.column_stack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    scatter = design.T @ design
    constraint = np.array([
        [0, 0, 2, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)
    values, vectors = eig(scatter, constraint)
    candidates = []
    for k in range(vectors.shape[1]):
        v = vectors[:, k]
        if np.max(np.abs(v.imag)) > 1.0e-7:
            continue
        v = v.real
        if 4.0 * v[0] * v[2] - v[1] * v[1] <= 0:
            continue
        residual = np.linalg.norm(design @ v)
        candidates.append((float(residual), v))
    if not candidates:
        raise ValueError("no positive-definite conic was found")
    _, coeff = min(candidates, key=lambda item: item[0])
    coeff /= max(np.linalg.norm(coeff[:3]), EPS)
    cq = np.array([
        [coeff[0], coeff[1] / 2.0, coeff[3] / 2.0],
        [coeff[1] / 2.0, coeff[2], coeff[4] / 2.0],
        [coeff[3] / 2.0, coeff[4] / 2.0, coeff[5]],
    ])
    # q = N p_h, so C_p = N^T C_q N.
    nmat = np.array([
        [1.0 / scale[0], 0.0, -mean[0] / scale[0]],
        [0.0, 1.0 / scale[1], -mean[1] / scale[1]],
        [0.0, 0.0, 1.0],
    ])
    cp = nmat.T @ cq @ nmat
    q2 = cp[:2, :2]
    linear = cp[:2, 2]
    if np.linalg.eigvalsh(q2).min() <= 0:
        cp = -cp
        q2 = cp[:2, :2]
        linear = cp[:2, 2]
    if np.linalg.eigvalsh(q2).min() <= 0:
        raise ValueError("fitted conic is not an ellipse")
    center = np.linalg.solve(q2, -linear)
    level = float(center @ q2 @ center - cp[2, 2])
    if level <= 0 or not np.isfinite(level):
        raise ValueError("ellipse has non-positive squared axes")
    qnorm = q2 / level
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / max(singular[-2], EPS))
    return Ellipse(center, qnorm, cp / level, condition)


def fit_ring_ellipses(points_yaw0_mm: np.ndarray) -> tuple[Ellipse, Ellipse]:
    p = np.asarray(points_yaw0_mm, dtype=float)
    n = len(p) // 2
    return fit_ellipse(p[:n]), fit_ellipse(p[n:])


def _roll_candidate(a1: float, b1: float, a2: float, b2: float,
                    y_d: float, r: float, l: float) -> dict | None:
    """Evaluate Eqs. (6)-(12) for one A/B root assignment."""
    g = b1 / a2 if abs(a2) > EPS else np.nan
    gp = b2 / a1 if abs(a1) > EPS else np.nan
    numerator = r * (-(1.0 + gp) * b1 + (1.0 + g) * b2)
    denominator = l * ((1.0 + g) * (1.0 + gp) * y_d
                        - (1.0 + gp) * b1 - (1.0 + g) * b2)
    if not np.isfinite(numerator + denominator) or abs(denominator) < EPS:
        return None
    # Eq. (12) is an atan of a ratio, but the physical quadrant is carried by
    # the signed numerator and denominator.  Plain arctan loses that quadrant
    # and causes discontinuous roll branch flips under centroid noise.
    theta = float(np.arctan2(numerator, denominator))
    # Keep the paper's principal roll convention.
    if theta > np.pi / 2.0:
        theta -= np.pi
    elif theta <= -np.pi / 2.0:
        theta += np.pi
    z_s = y_d * np.sin(theta)
    if abs(z_s) < 1.0e-10 or abs(g+1) < 1e-10:
        return None
    y_b1_i, z_b1_i = b1 * np.cos(theta), b1 * np.sin(theta)
    den = (g + 1.0) * z_s - 2.0 * z_b1_i
    if abs(den) < EPS:
        return None
    z_w = z_s + (g - 1.0) * z_s * r / den
    if abs(z_w) < 1.0e-10:
        return None
    y_w = z_w * (2.0 * (r * y_b1_i - l * z_b1_i)
                 / ((g + 1.0) * z_s * r) + l / r)
    y_s = y_w * z_s / z_w
    if not np.isfinite([theta, y_s, z_s, y_w, z_w]).all():
        return None
    return {
        "theta": theta,
        "source_i": np.array([0.0, y_s, z_s]),
        "phantom_center_i": np.array([0.0, y_w, z_w]),
        "g": float(g),
        "g_prime": float(gp),
        "roots_assignment": (a1, b1, a2, b2),
    }


def roll_candidates(ellipses: tuple[Ellipse, Ellipse], d_mm: np.ndarray,
                    config: YangConfig) -> list[dict]:
    """Enumerate the four A/B assignments required by the ellipse intersections."""
    roots = [e.x_axis_intersections(0.0) for e in ellipses]
    y_d = float(d_mm[1])
    if not np.isfinite(y_d):
        # E lies below O, F above O. A is the near-source side (+z).
        a1, b1 = roots[0]
        b2, a2 = roots[1]
        aa, bb = a2-a1, b2-b1
        r, l = config.ring_radius_mm, config.ring_half_spacing_mm
        if aa <= bb or bb <= 0:
            raise ValueError("invalid zero-roll ring intersections")
        zs = r*aa*bb/(l*(aa-bb))
        zw = zs - 2*l*zs/bb + r
        ys = l*zs*(2*b2-bb)/(r*bb)
        yw = ys*zw/zs
        return [dict(theta=0.0, source_i=np.array([0.,ys,zs]),
                     phantom_center_i=np.array([0.,yw,zw]), g=None,
                     g_prime=None, swap=(False,True), roots_mm=roots)]
    out = []
    for swap1 in (False, True):
        for swap2 in (False, True):
            r1 = roots[0][::-1] if swap1 else roots[0]
            r2 = roots[1][::-1] if swap2 else roots[1]
            candidate = _roll_candidate(float(r1[0]), float(r1[1]),
                                        float(r2[0]), float(r2[1]), y_d,
                                        config.ring_radius_mm,
                                        config.ring_half_spacing_mm)
            if candidate is not None and (candidate['source_i'][2] >
                    candidate['phantom_center_i'][2] + config.ring_radius_mm and
                    abs(candidate['theta']) < np.deg2rad(50)):
                candidate["roots_mm"] = roots
                candidate["swap"] = (swap1, swap2)
                out.append(candidate)
    if not out:
        raise ValueError("could not solve roll from the two ellipse intersections")
    return out


def _transform_i_to_detector(points_i: np.ndarray, source_i: np.ndarray,
                             theta: float, phi: float) -> np.ndarray:
    """Project virtual-detector coordinates to yaw-zero detector coordinates."""
    points = np.asarray(points_i, dtype=float)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    roll = np.array([[1.0, 0.0, 0.0],
                     [0.0, ct, st],
                     [0.0, -st, ct]])
    pitch = np.array([[cp, 0.0, -sp],
                      [0.0, 1.0, 0.0],
                      [sp, 0.0, cp]])
    transform = pitch @ roll
    src_b = transform @ np.asarray(source_i, dtype=float)
    p_b = (transform @ points.T).T
    denom = p_b[:, 2] - src_b[2]
    if np.any(np.abs(denom) < 1.0e-12):
        raise ValueError("a calibration ray is parallel to the detector plane")
    lam = -src_b[2] / denom
    hit = src_b[None, :] + lam[:, None] * (p_b - src_b[None, :])
    return hit[:, :2]


def _ring_points(config: YangConfig, center_i: np.ndarray,
                 angles: np.ndarray, ring_sign: float) -> np.ndarray:
    r = config.ring_radius_mm
    l = config.ring_half_spacing_mm
    a = np.asarray(angles, dtype=float)
    return np.column_stack([
        r * np.cos(a),
        np.full_like(a, center_i[1] + ring_sign * l),
        center_i[2] + r * np.sin(a),
    ])


def _ellipse_consistency(phi: float, roll: dict,
                         ellipses: tuple[Ellipse, Ellipse],
                         config: YangConfig) -> float:
    sample = np.linspace(0.0, 2.0 * np.pi, config.phase_samples, endpoint=False)
    total = 0.0
    try:
        for sign, ellipse in ((-1.0, ellipses[0]), (1.0, ellipses[1])):
            pts_i = _ring_points(config, roll["phantom_center_i"], sample, sign)
            pred = _transform_i_to_detector(pts_i, roll["source_i"],
                                            roll["theta"], phi)
            val = ellipse.value(pred)
            total += float(np.mean(val * val))
    except ValueError:
        return 1.0e30
    return total


def estimate_pitch(roll: dict, ellipses: tuple[Ellipse, Ellipse],
                   config: YangConfig) -> tuple[float, dict]:
    """Step 5: minimize Eq. (17), with a coarse scan plus bounded refinement."""
    lo, hi = np.deg2rad(config.pitch_search_deg)
    if not lo < hi or config.pitch_samples < 5:
        raise ValueError("invalid pitch search range")
    grid = np.linspace(lo, hi, int(config.pitch_samples))
    values = np.array([_ellipse_consistency(x, roll, ellipses, config) for x in grid])
    best = int(np.argmin(values))
    left = grid[max(0, best - 1)]
    right = grid[min(len(grid) - 1, best + 1)]
    if right - left < 1.0e-12:
        phi = float(grid[best])
        value = float(values[best])
    else:
        opt = minimize_scalar(
            lambda x: _ellipse_consistency(float(x), roll, ellipses, config),
            bounds=(left, right), method="bounded",
            options={"xatol": 1.0e-12},
        )
        phi, value = float(opt.x), float(opt.fun)
    return phi, {
        "pitch_grid_min_value": float(values[best]),
        "pitch_objective": value,
        "pitch_grid_deg": float(np.rad2deg(grid[best])),
    }


def _marked_ring_points(config: YangConfig, center_i: np.ndarray,
                        t: float, direction: float) -> np.ndarray:
    n = int(config.beads_per_ring)
    base = t + np.arange(n, dtype=float) * 2.0 * np.pi / n
    a = direction * base
    return np.vstack([
        _ring_points(config, center_i, a, -1.0),
        _ring_points(config, center_i, a, 1.0),
    ])


def estimate_gantry_angle(roll: dict, phi: float,
                          measured_yaw0_mm: np.ndarray,
                          config: YangConfig) -> tuple[float, dict, np.ndarray]:
    """Step 6: fit the marked E1/F1 angle using all twelve projections.

    The paper gives a linear least-squares equation for sin(t), cos(t).  The
    implementation below evaluates the same known phantom geometry through
    Eqs. (18)-(22) and performs a one-dimensional periodic least-squares fit.
    This is an implementation variation, not a reproduction of the paper's
    linear solver; it retains the same forward projection parameterization.
    """
    p = np.asarray(measured_yaw0_mm, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if p.shape != (expected, 2):
        raise ValueError(f"measured_yaw0_mm must have shape ({expected}, 2)")

    def residual(t: float, direction: float) -> tuple[float, np.ndarray]:
        virtual = _marked_ring_points(config, roll["phantom_center_i"], t, direction)
        pred = _transform_i_to_detector(virtual, roll["source_i"], roll["theta"], phi)
        return float(np.mean(np.sum((pred - p) ** 2, axis=1))), pred

    grid = np.linspace(-np.pi, np.pi, 721, endpoint=False)
    candidates = []
    for direction in (-1.0, 1.0):
        vals = np.array([residual(float(t), direction)[0] for t in grid])
        k = int(np.argmin(vals))
        step = 2.0 * np.pi / len(grid)
        centre = float(grid[k])
        opt = minimize_scalar(
            lambda x: residual(float(x), direction)[0],
            bounds=(centre - step, centre + step), method="bounded",
            options={"xatol": 1.0e-12},
        )
        pred = residual(float(opt.x), direction)[1]
        candidates.append((float(opt.fun), float(opt.x), direction, pred))
    err, t, direction, pred = min(candidates, key=lambda x: x[0])
    return t, {
        "gantry_direction": float(direction),
        "gantry_fit_mse_mm2": err,
        "gantry_fit_rmse_mm": float(np.sqrt(err)),
    }, pred


def _paper_gantry_linear_system(roll: dict, phi: float,
                                measured_yaw0_mm: np.ndarray,
                                config: YangConfig,
                                direction: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the Eq. (26) ``[sin(t), cos(t)]`` linear system.

    This is derived by cross-multiplying Eqs. (18)-(23), rather than copying
    the paper's expanded ``e1..e6`` coefficients.  The two forms are
    algebraically equivalent, while this form keeps detector-axis and ring
    signs explicit in the implementation's right-handed convention.
    """
    measured = np.asarray(measured_yaw0_mm, dtype=float)
    theta = float(roll["theta"])
    source_i = np.asarray(roll["source_i"], dtype=float)
    center_i = np.asarray(roll["phantom_center_i"], dtype=float)
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    transform = np.array([[cp, st * sp, -ct * sp],
                          [0.0, ct, st],
                          [sp, -st * cp, ct * cp]])
    source = transform @ source_i
    radius = float(config.ring_radius_mm)
    half_spacing = float(config.ring_half_spacing_mm)
    rows, rhs = [], []

    for ring, side in enumerate((-1.0, 1.0)):
        n = int(config.beads_per_ring)
        for j in range(n):
            beta = j * 2.0 * np.pi / n
            cb, sb = np.cos(beta), np.sin(beta)
            # X_i(t) = base + sin(t)*sin_part + cos(t)*cos_part,
            # matching alpha=direction*(t+beta).
            base_i = np.array([0.0, center_i[1] + side * half_spacing,
                               center_i[2]])
            sin_i = np.array([-radius * sb, 0.0,
                              direction * radius * cb])
            cos_i = np.array([radius * cb, 0.0,
                              direction * radius * sb])
            base = transform @ base_i
            sin_part = transform @ sin_i
            cos_part = transform @ cos_i
            for detector_axis in range(2):
                q = measured[ring * n + j, detector_axis]
                # q = (s_z*x - s_x*z)/(s_z-z).  Move every term to
                # the left and collect sin(t), cos(t), and the constant.
                s_coord = source[detector_axis]
                # Direct expansion of q(s_z-p_z)=s_z p_axis-p_z s_axis.
                sin_coeff = ((source[2] * sin_part[detector_axis]
                              - s_coord * sin_part[2]) + q * sin_part[2])
                cos_coeff = ((source[2] * cos_part[detector_axis]
                              - s_coord * cos_part[2]) + q * cos_part[2])
                constant = (source[2] * base[detector_axis]
                            - s_coord * base[2] - q * source[2]
                            + q * base[2])
                rows.append([sin_coeff, cos_coeff])
                rhs.append(-constant)
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def estimate_gantry_angle_paper(roll: dict, phi: float,
                                measured_yaw0_mm: np.ndarray,
                                config: YangConfig) -> tuple[float, dict, np.ndarray]:
    """Estimate marked-bead angle using Yang et al. Eq. (26).

    The linear solve estimates ``sin(t)`` and ``cos(t)`` jointly.  Both
    winding conventions are evaluated and the one with the smaller algebraic
    residual is selected.  A unit-circle projection prevents noise from
    changing the scale of the two solved coefficients.
    """
    p = np.asarray(measured_yaw0_mm, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if p.shape != (expected, 2):
        raise ValueError(f"measured_yaw0_mm must have shape ({expected}, 2)")
    candidates = []
    for direction in (-1.0, 1.0):
        matrix, rhs = _paper_gantry_linear_system(
            roll, phi, p, config, direction)
        solution, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
        norm = float(np.linalg.norm(solution))
        if norm < 1.0e-12 or rank < 2:
            continue
        unit = solution / norm
        t = float(np.arctan2(unit[0], unit[1]))
        algebraic = matrix @ unit - rhs
        candidates.append((float(np.mean(algebraic ** 2)), t, direction,
                           unit, rank, singular))
    if not candidates:
        raise ValueError("Yang Eq. (26) is rank deficient for this view")
    score, t, direction, unit, rank, singular = min(candidates,
                                                     key=lambda item: item[0])
    # Use the same geometric forward model as the PIC implementation only to
    # report a directly interpretable residual and predicted bead positions.
    virtual = _marked_ring_points(config, roll["phantom_center_i"], t, direction)
    predicted = _transform_i_to_detector(virtual, roll["source_i"],
                                         roll["theta"], phi)
    residual = predicted - p
    return t, {
        "method": "yang_eq26_linear_sin_cos",
        "gantry_direction": float(direction),
        "sin_t": float(unit[0]),
        "cos_t": float(unit[1]),
        "equation_mse_mm2": score,
        "equation_rank": int(rank),
        "equation_singular_values": singular.tolist(),
        "gantry_fit_rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
    }, predicted


def _choose_roll(rolls: list[dict], ellipses: tuple[Ellipse, Ellipse],
                 config: YangConfig,
                 reference_sdd_mm: float | None = None
                 ) -> tuple[dict, float, dict]:
    scored = []
    for candidate in rolls:
        try:
            phi, info = estimate_pitch(candidate, ellipses, config)
            source = np.asarray(candidate["source_i"], dtype=float)
            sdd = float(abs(np.cos(phi) *
                            (-np.sin(candidate["theta"]) * source[1]
                             + np.cos(candidate["theta"]) * source[2])))
            # Yang's per-view objective can select a mathematically valid but
            # physically wrong ellipse-root branch near a six-point
            # degeneracy.  In a flat-panel CBCT scan, SDD is common to all
            # views, so a robust stack estimate can disambiguate that branch.
            if reference_sdd_mm is None:
                score = (info["pitch_objective"],)
            else:
                score = (abs(sdd - reference_sdd_mm) / max(reference_sdd_mm, EPS),
                         info["pitch_objective"])
            scored.append((score, candidate, phi, info, sdd))
        except (ValueError, FloatingPointError):
            continue
    if not scored:
        raise ValueError("none of the A/B ellipse-root assignments gives a valid pose")
    score, roll, phi, info, sdd = min(scored, key=lambda x: x[0])
    return roll, phi, {"roll_candidates": len(rolls),
                       "selected_pitch_objective": float(info["pitch_objective"]),
                       "selected_sdd_mm": float(sdd),
                       "reference_sdd_mm": (None if reference_sdd_mm is None
                                             else float(reference_sdd_mm)),
                       "pitch": info,
                       "all_pitch_objectives": [float(x[3]["pitch_objective"])
                                                for x in scored]}


def calibrate_view(points_px: np.ndarray, config: YangConfig,
                   reference_sdd_mm: float | None = None) -> YangPose:
    """Calibrate one Yang PIC view from twelve ordered bead centroids."""
    p = _as_points(points_px)
    if len(p) != 2 * int(config.beads_per_ring):
        raise ValueError("points_px does not match config.beads_per_ring")
    pitch = _as_pitch(config.pixel_size_mm)
    if (not np.isfinite([config.ring_radius_mm, config.ring_half_spacing_mm]).all()
            or min(config.ring_radius_mm, config.ring_half_spacing_mm) <= 0
            or config.phase_samples < 12):
        raise ValueError("positive phantom dimensions and >=12 phase samples required")
    o_px, o_info = estimate_o(p)
    yaw, yaw0, d_yaw0, yaw_info = estimate_yaw(p, o_px, pitch)
    ellipses = fit_ring_ellipses(yaw0)
    rolls = roll_candidates(ellipses, d_yaw0, config)
    roll, phi, select_info = _choose_roll(
        rolls, ellipses, config, reference_sdd_mm=reference_sdd_mm)

    t = None
    pred = None
    gantry_info = {"enabled": bool(config.estimate_gantry_angle)}
    if config.estimate_gantry_angle:
        if config.gantry_solver == "paper_linear":
            try:
                t, gantry_info, pred = estimate_gantry_angle_paper(
                    roll, phi, yaw0, config)
            except ValueError as exc:
                t, gantry_info, pred = estimate_gantry_angle(
                    roll, phi, yaw0, config)
                gantry_info["method"] = "periodic_fit_fallback"
                gantry_info["paper_linear_failure"] = str(exc)
        elif config.gantry_solver == "periodic_fit":
            t, gantry_info, pred = estimate_gantry_angle(
                roll, phi, yaw0, config)
            gantry_info["method"] = "periodic_fit"
        else:
            raise ValueError("gantry_solver must be 'paper_linear' or 'periodic_fit'")
    else:
        gantry_info["reason"] = "disabled: E1/F1 marking is not available"

    source = roll["source_i"]
    phantom = roll["phantom_center_i"]
    # O is detector-plane origin; these are useful engineering distances.
    sdd = float(abs(np.cos(phi)*(-np.sin(roll['theta'])*source[1]
                                + np.cos(roll['theta'])*source[2])))
    sod = float(np.linalg.norm(source - phantom))
    rmse = None if pred is None else float(np.sqrt(np.mean((pred - yaw0) ** 2)))
    ct, st = np.cos(roll['theta']), np.sin(roll['theta'])
    cp, sp = np.cos(phi), np.sin(phi)
    source_yaw0 = np.array([-sp*(-st*source[1]+ct*source[2]),
                           ct*source[1]+st*source[2]])
    back = yaw_info['rotation_matrix_column']
    principal = o_px + (source_yaw0 @ back) / pitch
    residual_px = None if pred is None else ((pred-yaw0) @ back) / pitch
    diagnostics = {
        "o": o_info,
        "yaw": yaw_info,
        "ellipse_condition": [ellipses[0].condition, ellipses[1].condition],
        "ellipse_axis_intersections_mm": [
            ellipses[0].x_axis_intersections(0.0),
            ellipses[1].x_axis_intersections(0.0),
        ],
        "roll": {
            "g": roll["g"], "g_prime": roll["g_prime"],
            "root_swap": roll["swap"],
        },
        "selection": select_info,
        "gantry": gantry_info,
        "distance_definition": {
            "sdd": "abs((T_I_from_i @ source_i)[2]); perpendicular plane distance",
            "sod": "norm(source_i - phantom_center_i)",
        },
    }
    return YangPose(
        o_px=np.asarray(o_px),
        o_mm_origin=np.zeros(2),
        d_px=np.asarray(yaw_info["d_px"]),
        d_mm=np.asarray(yaw_info["d_mm"]),
        yaw_rad=float(yaw),
        roll_rad=float(roll["theta"]),
        pitch_rad=float(phi),
        source_i_mm=np.asarray(source),
        phantom_center_i_mm=np.asarray(phantom),
        source_detector_distance_mm=sdd,
        source_phantom_distance_mm=sod,
        source_axis_distance_mm=float(abs(source[2]-phantom[2])),
        principal_point_px=principal,
        reprojection_rmse_px=None if residual_px is None else float(np.sqrt(np.mean(residual_px**2))),
        gantry_angle_rad=None if t is None else float(t),
        ring_ellipses=ellipses,
        points_yaw0_mm=yaw0,
        predicted_points_yaw0_mm=pred,
        reprojection_rmse_mm=rmse,
        diagnostics=diagnostics,
        pixel_size_mm=np.asarray(pitch),
    )


def calibrate_stack(points_stack_px: np.ndarray, config: YangConfig) -> list[YangPose]:
    """Calibrate two indexed rings independently for every view."""
    stack = np.asarray(points_stack_px, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if stack.ndim != 3 or stack.shape[1:] != (expected, 2):
        raise ValueError(
            f"points_stack_px must have shape (views, {expected}, 2)")
    # First pass follows the paper's independent-view rule.  The robust SDD
    # median is then used only to disambiguate near-degenerate roll/pitch
    # branches; it does not inject any mechanical ground truth.
    initial = [calibrate_view(frame, config) for frame in stack]
    sdds = np.asarray([pose.source_detector_distance_mm for pose in initial],
                      dtype=float)
    median_sdd = float(np.median(sdds))
    finite = sdds[np.isfinite(sdds)
                  & (sdds > 0.25 * median_sdd)
                  & (sdds < 4.0 * median_sdd)]
    reference_sdd = float(np.median(finite)) if len(finite) else None
    if reference_sdd is None:
        return initial
    return [calibrate_view(frame, config, reference_sdd_mm=reference_sdd)
            for frame in stack]


def _pose_parameter_vector(pose: YangPose) -> np.ndarray:
    """Pack the seven Yang beta values and two O coordinates."""
    source = np.asarray(pose.source_i_mm, dtype=float)
    centre = np.asarray(pose.phantom_center_i_mm, dtype=float)
    return np.array([
        pose.roll_rad, pose.pitch_rad, pose.yaw_rad,
        source[2], centre[1], centre[2],
        0.0 if pose.gantry_angle_rad is None else pose.gantry_angle_rad,
        pose.o_px[0] * pose.pixel_size_mm[0],
        pose.o_px[1] * pose.pixel_size_mm[1],
    ], dtype=float)


def _model_pose_from_parameters(initial: YangPose, parameters: np.ndarray,
                                points_px: np.ndarray,
                                config: YangConfig) -> tuple[YangPose, np.ndarray]:
    """Create a YangPose and predicted points from the extended variables."""
    x = np.asarray(parameters, dtype=float)
    theta, phi, yaw, z_s, y_w, z_w, t, o_u_mm, o_v_mm = x
    if abs(z_w) < 1.0e-9:
        raise ValueError("extended Yang parameterization has z_W near zero")
    y_s = y_w * z_s / z_w
    source = np.array([0.0, y_s, z_s])
    centre = np.array([0.0, y_w, z_w])
    direction = float(initial.diagnostics.get("gantry", {}).get(
        "gantry_direction", -1.0))
    yaw0 = _marked_ring_points(config, centre, t, direction)
    predicted_yaw0 = _transform_i_to_detector(yaw0, source, theta, phi)
    yaw_matrix = np.asarray(initial.diagnostics["yaw"]["rotation_matrix_column"],
                            dtype=float)
    predicted_relative_mm = predicted_yaw0 @ yaw_matrix
    o_mm = np.array([o_u_mm, o_v_mm])
    predicted_px = (predicted_relative_mm + o_mm[None, :]) / config.pixel_size_mm

    # Recompute the derived physical quantities using the same definitions as
    # the independent PIC solver.  The ellipse objects and diagnostics remain
    # useful provenance from the measured frame.
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    source_yaw0 = np.array([-sp * (-st * y_s + ct * z_s),
                            ct * y_s + st * z_s])
    principal = np.array([o_u_mm, o_v_mm]) / config.pixel_size_mm
    principal = principal + (source_yaw0 @ yaw_matrix) / config.pixel_size_mm
    sdd = float(abs(cp * (-st * y_s + ct * z_s)))
    sod = float(np.linalg.norm(source - centre))
    residual_px = predicted_px - np.asarray(points_px, dtype=float)
    updated = replace(
        initial,
        o_px=np.array([o_u_mm, o_v_mm]) / config.pixel_size_mm,
        o_mm_origin=np.zeros(2),
        yaw_rad=float(yaw),
        roll_rad=float(theta),
        pitch_rad=float(phi),
        source_i_mm=source,
        phantom_center_i_mm=centre,
        source_detector_distance_mm=sdd,
        source_phantom_distance_mm=sod,
        source_axis_distance_mm=float(abs(z_s - z_w)),
        principal_point_px=principal,
        reprojection_rmse_px=float(np.sqrt(np.mean(residual_px ** 2))),
        gantry_angle_rad=float(t),
        points_yaw0_mm=predicted_yaw0,
        predicted_points_yaw0_mm=predicted_yaw0,
        reprojection_rmse_mm=float(np.sqrt(np.mean(
            (predicted_yaw0 - initial.points_yaw0_mm) ** 2))),
        diagnostics=dict(initial.diagnostics, extended_parameters=x.tolist()),
    )
    return updated, predicted_px


def _source_grid_coordinates(parameters: np.ndarray,
                             pixel_size_mm: np.ndarray) -> np.ndarray:
    """Equation (28): source position in the real detector grid."""
    theta, phi, yaw, z_s, y_w, z_w, _, o_u_mm, o_v_mm = parameters
    if abs(z_w) < 1.0e-9:
        raise ValueError("extended Yang parameterization has z_W near zero")
    y_s = y_w * z_s / z_w
    ct, st = np.cos(theta), np.sin(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    to_yaw0 = np.array([[cp, st * sp, -ct * sp],
                        [0.0, ct, st],
                        [sp, -st * cp, ct * cp]])
    c, s = np.cos(yaw), np.sin(yaw)
    to_real = np.eye(3)
    to_real[:2, :2] = np.array([[c, s], [-s, c]])
    source_real = to_real @ (to_yaw0 @ np.array([0.0, y_s, z_s]))
    return source_real + np.array([o_u_mm, o_v_mm, 0.0])


def extended_calibrate_stack(points_stack_px: np.ndarray,
                             config: YangConfig,
                             source_constraint_weight: float = 12.0,
                             max_nfev: int = 500,
                             initial_poses: list[YangPose] | None = None,
                             robust_loss: str = "soft_l1") -> YangExtendedResult:
    """Yang 2017 Section II.D extended calibration.

    This extension is intended for the paper's fixed-source/fixed-detector
    arrangement (the phantom or stage changes pose).  It starts from the
    independent PIC result and minimizes the measured bead reprojection error
    plus the Eq. (29) consistency penalty on the source position in the real
    detector grid.  Do not enable this constraint for a gantry whose source
    physically moves between views.
    """
    stack = np.asarray(points_stack_px, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if stack.ndim != 3 or stack.shape[1:] != (expected, 2):
        raise ValueError(
            f"points_stack_px must have shape (views, {expected}, 2)")
    if source_constraint_weight < 0 or not np.isfinite(source_constraint_weight):
        raise ValueError("source_constraint_weight must be finite and nonnegative")
    poses = calibrate_stack(stack, config) if initial_poses is None else list(initial_poses)
    if len(poses) != len(stack):
        raise ValueError("initial_poses length does not match points_stack_px")
    pixel = _as_pitch(config.pixel_size_mm)
    view_count = len(poses)
    packed = np.asarray([_pose_parameter_vector(pose) for pose in poses])
    source_before = np.asarray([_source_grid_coordinates(row, pixel)
                                for row in packed])
    mean_before = source_before.mean(axis=0)
    x0 = np.r_[packed.ravel(), mean_before]
    scale_mm = float(np.mean(pixel))

    def unpack(x):
        return np.asarray(x[:view_count * 9]).reshape(view_count, 9), np.asarray(x[-3:])

    def residual(x):
        variables, source_mean = unpack(x)
        values = []
        sources = []
        for view, (row, measured) in enumerate(zip(variables, stack)):
            try:
                _, predicted = _model_pose_from_parameters(
                    poses[view], row, measured, config)
            except (ValueError, FloatingPointError):
                return np.full(view_count * 24 + view_count * 3, 1.0e6)
            values.extend((predicted - measured).ravel())
            sources.append(_source_grid_coordinates(row, pixel))
        sources = np.asarray(sources)
        values.extend((np.sqrt(source_constraint_weight / 12.0) *
                       (sources - source_mean[None, :]) / scale_mm).ravel())
        return np.asarray(values, dtype=float)

    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    lower[:view_count * 9:9] = -np.deg2rad(80.0)
    upper[:view_count * 9:9] = np.deg2rad(80.0)
    lower[1:view_count * 9:9] = -np.deg2rad(89.0)
    upper[1:view_count * 9:9] = np.deg2rad(89.0)
    jac_sparsity = lil_matrix((view_count * (24 + 3), len(x0)), dtype=int)
    for view in range(view_count):
        col0 = view * 9
        row0 = view * 24
        jac_sparsity[row0:row0 + 24, col0:col0 + 9] = 1
        source_row = view_count * 24 + view * 3
        jac_sparsity[source_row:source_row + 3, col0:col0 + 9] = 1
        jac_sparsity[source_row:source_row + 3, view_count * 9:] = 1
    result = least_squares(
        residual, x0, bounds=(lower, upper), jac_sparsity=jac_sparsity,
        loss=robust_loss, f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
        ftol=1.0e-7, xtol=1.0e-7, gtol=1.0e-7, verbose=0,
    )
    variables, source_mean = unpack(result.x)
    final_poses = []
    final_residuals = []
    for view, (row, measured) in enumerate(zip(variables, stack)):
        pose, predicted = _model_pose_from_parameters(
            poses[view], row, measured, config)
        final_poses.append(pose)
        final_residuals.append(predicted - measured)
    source_after = np.asarray([_source_grid_coordinates(row, pixel)
                               for row in variables])
    before_error = np.asarray([p.reprojection_rmse_px for p in poses])
    after_error = np.asarray([p.reprojection_rmse_px for p in final_poses])
    return YangExtendedResult(
        poses=final_poses,
        source_grid_mean_mm=np.asarray(source_mean),
        source_grid_std_before_mm=source_before.std(axis=0),
        source_grid_std_after_mm=source_after.std(axis=0),
        reprojection_rmse_before_px=float(np.sqrt(np.mean(before_error ** 2))),
        reprojection_rmse_after_px=float(np.sqrt(np.mean(after_error ** 2))),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        optimizer_nfev=int(result.nfev),
        source_constraint_weight=float(source_constraint_weight),
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    return value


def pose_to_dict(pose: YangPose) -> dict:
    result = asdict(pose)
    result["ring_ellipses"] = [
        {"center_mm": e.center_mm.tolist(),
         "q_mm_inv2": e.q_mm_inv2.tolist(),
         "conic": e.conic.tolist(),
         "condition": e.condition}
        for e in pose.ring_ellipses
    ]
    return _jsonable(result)


def extended_result_to_dict(result: YangExtendedResult) -> dict:
    """Serialize PIC + extended calibration output for JSON reports."""
    return _jsonable({
        "poses": [pose_to_dict(pose) for pose in result.poses],
        "source_grid_mean_mm": result.source_grid_mean_mm,
        "source_grid_std_before_mm": result.source_grid_std_before_mm,
        "source_grid_std_after_mm": result.source_grid_std_after_mm,
        "reprojection_rmse_before_px": result.reprojection_rmse_before_px,
        "reprojection_rmse_after_px": result.reprojection_rmse_after_px,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.optimizer_message,
        "optimizer_nfev": result.optimizer_nfev,
        "source_constraint_weight": result.source_constraint_weight,
    })


'''Legacy command-line wrapper; call ``calibrate_view`` or workflow instead.
def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help=".npy/.npz array with shape [views,12,2] or [12,2]")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--radius-mm", type=float, required=True)
    parser.add_argument("--half-spacing-mm", type=float, required=True)
    parser.add_argument("--pixel-u-mm", type=float, required=True)
    parser.add_argument("--pixel-v-mm", type=float, required=True)
    parser.add_argument("--no-gantry", action="store_true")
    args = parser.parse_args()

    loaded = np.load(args.input)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = "points_px" if "points_px" in loaded else loaded.files[0]
        data = loaded[key]
    else:
        data = loaded
    if data.ndim == 2:
        data = data[None, ...]
    config = YangConfig(
        ring_radius_mm=args.radius_mm,
        ring_half_spacing_mm=args.half_spacing_mm,
        pixel_size_mm=(args.pixel_u_mm, args.pixel_v_mm),
        estimate_gantry_angle=not args.no_gantry,
    )
    poses = calibrate_stack(data, config)
    output = {"config": _jsonable(asdict(config)),
              "poses": [pose_to_dict(p) for p in poses],
              "reconstruction_geometries": [
                  geometry_to_dict(geometry_from_pose(p, config.pixel_size_mm))
                  for p in poses
              ]}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
'''
