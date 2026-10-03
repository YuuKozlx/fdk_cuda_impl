"""Per-view calibration following Cho et al. (2005).

This module intentionally follows the paper's construction rather than the
projective DLT route: fit the two bead ellipses, form projected lines from
opposite bead pairs, estimate the piercing point, and expose the geometric
intermediates used by Cho's detector-angle/source-position equations.

The analytic eta stage implements Eqs. (14)-(16), including the ellipse-model
extrema from Appendix Eqs. (A4)-(A5).  The same indexed points also determine
the complete per-view projective geometry, which is decomposed into the Cho
virtual/real detector frames for an independent angle check.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.linalg import svd
from scipy.optimize import least_squares


EPS = 1e-12


@dataclass
class EllipseFit:
    center_px: np.ndarray
    conic_coefficients: np.ndarray
    shape_matrix: np.ndarray
    normalized_level: float
    semi_axes_px: np.ndarray
    principal_axis_angle_rad: float
    residual_rms: float


@dataclass
class ChoAnalyticFrame:
    ellipse_lower: EllipseFit
    ellipse_upper: EllipseFit
    piercing_point_px: np.ndarray
    opposite_pair_lines: np.ndarray
    line_fit_rms_px: float
    detector_in_plane_angle_rad: float
    detector_in_plane_angle_deg: float
    ellipse_extrema_px: dict
    converging_point_pa_px: np.ndarray
    ellipse_lines: np.ndarray
    eta_diagnostics: dict
    complete_geometry: dict
    notes: list[str]


def _fit_ellipse(points_px: np.ndarray) -> EllipseFit:
    """Fit a general conic and convert it to center/axes form."""
    p = np.asarray(points_px, dtype=float)
    if p.shape[0] < 5 or p.shape[1] != 2:
        raise ValueError("ellipse fitting needs at least five (N,2) points")
    # Normalize before forming squared terms.  Directly fitting 1024-pixel
    # coordinates mixes O(1e6) quadratic columns with an O(1) constant and
    # measurably biases the analytic extrema used by Cho's eta equations.
    mean = p.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((p - mean) ** 2, axis=1))))
    if not np.isfinite(scale) or scale <= EPS:
        raise ValueError("ellipse points are degenerate")
    normalized = (p - mean) / scale
    x, y = normalized[:, 0], normalized[:, 1]
    design = np.column_stack([x * x, y * y, x * y, x, y, np.ones(len(p))])
    _, _, vt = svd(design)
    normalized_coeff = vt[-1]
    qn = np.array([[normalized_coeff[0], normalized_coeff[2] / 2.0],
                   [normalized_coeff[2] / 2.0, normalized_coeff[1]]])
    dn = np.array([normalized_coeff[3], normalized_coeff[4]])
    try:
        cn0 = -np.linalg.solve(2.0 * qn, dn)
        ln0 = float(cn0 @ qn @ cn0 - normalized_coeff[5])
        algebraic_is_ellipse = (abs(ln0) > EPS and
                                np.all(np.linalg.eigvalsh(qn / ln0) > 0.0))
    except np.linalg.LinAlgError:
        algebraic_is_ellipse = False
    if not algebraic_is_ellipse:
        # The unconstrained algebraic SVD may select a hyperbola for noisy,
        # strongly foreshortened rings.  Fit a guaranteed ellipse in the same
        # normalized coordinates without changing any downstream Cho formula.
        covariance = np.cov(normalized.T)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        initial = np.array([0.0, 0.0,
                            np.log(max(np.sqrt(2.0 * values[0]), 0.05)),
                            np.log(max(np.sqrt(2.0 * values[1]), 0.05)),
                            np.arctan2(vectors[1, 0], vectors[0, 0])])

        def ellipse_residual(x):
            c, s = np.cos(x[4]), np.sin(x[4])
            rotation = np.array([[c, -s], [s, c]])
            local = (normalized - x[:2]) @ rotation
            axes = np.exp(x[2:4])
            return np.sum((local / axes) ** 2, axis=1) - 1.0

        fitted = least_squares(ellipse_residual, initial, loss="soft_l1",
                               f_scale=0.02, max_nfev=500)
        c, s = np.cos(fitted.x[4]), np.sin(fitted.x[4])
        rotation = np.array([[c, -s], [s, c]])
        axes = np.exp(fitted.x[2:4])
        qn = rotation @ np.diag(1.0 / axes ** 2) @ rotation.T
        centre_n = fitted.x[:2]
        dn = -2.0 * qn @ centre_n
        fn = float(centre_n @ qn @ centre_n - 1.0)
        normalized_coeff = np.array([qn[0, 0], qn[1, 1], 2.0 * qn[0, 1],
                                     dn[0], dn[1], fn])
    cn = np.array([
        [normalized_coeff[0], normalized_coeff[2] / 2.0,
         normalized_coeff[3] / 2.0],
        [normalized_coeff[2] / 2.0, normalized_coeff[1],
         normalized_coeff[4] / 2.0],
        [normalized_coeff[3] / 2.0, normalized_coeff[4] / 2.0,
         normalized_coeff[5]],
    ])
    transform = np.array([
        [1.0 / scale, 0.0, -mean[0] / scale],
        [0.0, 1.0 / scale, -mean[1] / scale],
        [0.0, 0.0, 1.0],
    ])
    conic = transform.T @ cn @ transform
    coeff = np.array([
        conic[0, 0], conic[1, 1], 2.0 * conic[0, 1],
        2.0 * conic[0, 2], 2.0 * conic[1, 2], conic[2, 2],
    ])
    q = np.array([[coeff[0], coeff[2] / 2.0],
                  [coeff[2] / 2.0, coeff[1]]])
    d = np.array([coeff[3], coeff[4]])
    f = coeff[5]
    center = -np.linalg.solve(2.0 * q, d)
    level = float(center @ q @ center - f)
    if level <= 0:
        q = -q
        level = -level
    eigval, eigvec = np.linalg.eigh(q / level)
    if np.any(eigval <= 0):
        raise ValueError("fitted conic is not an ellipse")
    axes = 1.0 / np.sqrt(eigval)
    # First principal axis is the longer axis.
    order = np.argsort(axes)[::-1]
    axes = axes[order]
    eigvec = eigvec[:, order]
    angle = float(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    # Algebraic residual after normalizing to (x-c)^T Q (x-c)=1.
    delta = p - center
    residual = np.sum((delta @ (q / level)) * delta, axis=1) - 1.0
    rms = float(np.sqrt(np.mean(residual * residual)))
    return EllipseFit(center, coeff, q, level, axes, angle, rms)


def _ellipse_point(ellipse: EllipseFit, direction: np.ndarray) -> np.ndarray:
    """Return the ellipse boundary point maximizing a linear direction."""
    direction = np.asarray(direction, dtype=float)
    qn = ellipse.shape_matrix / ellipse.normalized_level
    return ellipse.center_px + direction / np.sqrt(direction @ np.linalg.inv(qn) @ direction)


def _homogeneous_line(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.cross(np.r_[p, 1.0], np.r_[q, 1.0])


def _intersect_lines(lines: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares intersection of 2-D homogeneous lines."""
    lines = np.asarray(lines, dtype=float)
    _, singular, vt = svd(lines)
    h = vt[-1]
    if abs(h[2]) < EPS:
        return np.array([np.inf, np.inf]), float(singular[-1])
    point = h[:2] / h[2]
    distances = np.abs(lines[:, :2] @ point + lines[:, 2]) / np.maximum(
        np.linalg.norm(lines[:, :2], axis=1), EPS)
    return point, float(np.sqrt(np.mean(distances * distances)))


def _ellipse_extrema(ellipse: EllipseFit) -> dict:
    """Analytic extrema on the fitted ellipse, not just measured beads."""
    qn = ellipse.shape_matrix / ellipse.normalized_level
    inv_q = np.linalg.inv(qn)
    out = {}
    for label, direction in (("u_min", np.array([-1.0, 0.0])),
                             ("u_max", np.array([1.0, 0.0])),
                             ("v_min", np.array([0.0, -1.0])),
                             ("v_max", np.array([0.0, 1.0]))):
        out[label] = (ellipse.center_px +
                      (inv_q @ direction) /
                      np.sqrt(direction @ inv_q @ direction))
    return out


def _wrap_line_angle(angle: float) -> float:
    """Return an undirected line angle in [-pi/2, pi/2)."""
    return float((angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)


def _cho_eta_from_ellipses(ellipses: tuple[EllipseFit, EllipseFit]):
    """Implement Cho 2005 Eqs. (14)-(16) for detector rotation eta.

    The two ellipse fits are expressed in the real detector image coordinates.
    ``P_a`` is the virtual-detector converging point obtained from the ratio
    of the short and long axes.  ``L1`` and ``L2`` are the chords joining the
    analytic extrema in the real-detector U direction.  Their signed line
    angles relative to X_I are combined with Cho's distance weighting.
    """
    p1, p2 = (np.asarray(e.center_px, dtype=float) for e in ellipses)
    # Cho Eq. (14) uses sqrt(a_k / b_k) from Eq. (8), not the principal-axis
    # eigenvalue ratio.  Keep the literal paper coefficients after normalizing
    # each conic to (p-c)^T Q (p-c)=1.
    axis_ratios = []
    for ellipse in ellipses:
        qn = ellipse.shape_matrix / ellipse.normalized_level
        if np.linalg.det(qn) <= 0.0 or qn[0, 0] <= 0.0 or qn[1, 1] <= 0.0:
            raise ValueError("ellipse shape matrix is not positive definite")
        axis_ratios.append(float(np.sqrt(qn[0, 0] / qn[1, 1])))
    r1, r2 = axis_ratios
    denominator = r1 + r2
    if denominator <= EPS:
        raise ValueError("Cho eta has degenerate ellipse axis ratios")
    # Eq. (14): P_a lies on the line joining the two ellipse centers.
    pa = (p1 * r1 + p2 * r2) / denominator

    extrema = [_ellipse_extrema(e) for e in ellipses]
    # Eq. (A4)-(A5): extrema in the real detector U direction.  The line
    # joining the two extrema of each ellipse is L_k.
    lines = []
    line_angles = []
    for item in extrema:
        p_min = np.asarray(item["u_min"], dtype=float)
        p_max = np.asarray(item["u_max"], dtype=float)
        direction = p_max - p_min
        length = float(np.linalg.norm(direction))
        if length <= EPS:
            raise ValueError("Cho eta ellipse extrema are coincident")
        lines.append(np.r_[p_min, p_max])
        line_angles.append(_wrap_line_angle(np.arctan2(direction[1], direction[0])))

    # A line has a pi ambiguity.  Cho's small-angle calibration convention
    # uses the branch closest to the detector X_I direction.  For larger
    # rotations, unwrap both lines consistently before applying Eq. (16).
    a1, a2 = line_angles
    while a2 - a1 >= 0.5 * np.pi:
        a2 -= np.pi
    while a2 - a1 < -0.5 * np.pi:
        a2 += np.pi
    d12 = float(np.linalg.norm(p2 - p1))
    if d12 <= EPS:
        raise ValueError("ellipse centers are coincident; Cho eta is undefined")
    d1a = float(np.linalg.norm(pa - p1))
    da2 = float(np.linalg.norm(p2 - pa))
    # Eq. (16).  A(p,q) is the signed angle between the real X_I axis and q.
    # Eq. (5) maps +X_i to a line with real-detector coordinate angle -eta.
    # Therefore the signed image-line angle is negated when reporting Cho's
    # detector rotation parameter.  The paper writes A(p,q) as an unsigned
    # angle; this explicit sign convention is needed in software.
    eta = -(d1a * a1 + da2 * a2) / d12
    eta = _wrap_line_angle(eta)
    return eta, pa, np.asarray(lines, dtype=float), {
        "sqrt_a_over_b": np.asarray(axis_ratios),
        "center_p1_px": p1,
        "center_p2_px": p2,
        "pa_px": pa,
        "line_angles_rad": np.asarray([a1, a2]),
        "line_angles_deg": np.degrees([a1, a2]),
        "center_distances_px": np.asarray([d1a, da2, d12]),
        "formula": "eta=(|P1Pa|*A(XI,L1)+|PaP2|*A(XI,L2))/|P1P2|",
        "method": "cho_2005_equations_14_16",
        "sign_convention": "Eq. (5): virtual +XI maps to real line angle -eta",
    }


def _complete_geometry_from_dlt(points_px: np.ndarray, config) -> dict:
    """Recover complete pose and express its rotation in Cho's frames."""
    from .dlt import calibrate_view, phantom_points

    world = phantom_points(config)
    pose = calibrate_view(points_px, config, world)
    source = np.asarray(pose.source_world_mm, dtype=float)
    gantry = float(np.arctan2(source[1], source[0]))
    st, ct = np.sin(gantry), np.cos(gantry)
    # Cho Eq. (2), world -> virtual detector coordinates.
    r_world_to_virtual = np.array([
        [st, ct, 0.0],
        [0.0, 0.0, -1.0],
        [-ct, st, 0.0],
    ])
    # RQ decomposition in cho_calibration uses the same image-axis signs as
    # Cho's real detector frame for U=-Sf*X and V=-Sf*Y.
    r_world_to_real = np.asarray(pose.rotation_world_to_detector, dtype=float)
    r_virtual_to_real = r_world_to_real @ r_world_to_virtual.T
    # Invert Cho Eq. (5).  This branch is nonsingular for |theta| < 90 deg.
    theta = float(np.arcsin(np.clip(r_virtual_to_real[2, 1], -1.0, 1.0)))
    phi = float(np.arctan2(r_virtual_to_real[2, 0],
                           r_virtual_to_real[2, 2]))
    eta = float(np.arctan2(r_virtual_to_real[0, 1],
                           r_virtual_to_real[1, 1]))
    detector_center = np.asarray(pose.detector_origin_world_mm, dtype=float)
    return {
        "projection_matrix": np.asarray(pose.projection_matrix),
        "source_world_mm": source,
        "detector_principal_point_world_mm": detector_center,
        "source_detector_distance_mm": float(pose.source_detector_distance_mm),
        "principal_point_px": np.asarray(pose.principal_point_px),
        "detector_u_axis_world": np.asarray(pose.detector_u_axis_world),
        "detector_v_axis_world": np.asarray(pose.detector_v_axis_world),
        "detector_normal_world": np.asarray(pose.detector_normal_world),
        "rotation_world_to_real_detector": r_world_to_real,
        "rotation_world_to_virtual_detector": r_world_to_virtual,
        "rotation_virtual_to_real_detector": r_virtual_to_real,
        "cho_phi_rad": phi,
        "cho_theta_rad": theta,
        "cho_eta_rad": eta,
        "cho_phi_deg": float(np.degrees(phi)),
        "cho_theta_deg": float(np.degrees(theta)),
        "cho_eta_deg": float(np.degrees(eta)),
        "gantry_angle_rad": gantry,
        "gantry_angle_deg": float(np.degrees(gantry)),
        "reprojection_rmse_px": float(pose.reprojection_rmse_px),
        "max_reprojection_error_px": float(pose.max_reprojection_error_px),
        "optimizer_success": bool(pose.optimizer_success),
    }


def calibrate_frame_cho(points_indexed_px: np.ndarray,
                        config=None,
                        ring_indices: tuple[np.ndarray, np.ndarray] | None = None,
                        opposite_shift: int | None = None) -> ChoAnalyticFrame:
    """Run the ellipse/line stages of Cho's analytic procedure.

    ``points_indexed_px`` must contain two equally sized, consistently indexed
    rings.  The ring size must be even so that the default ``k <-> k+n/2``
    physical opposite-pair construction is defined.  Cho used twelve targets
    per ring, while the Yang 6+6 phantom uses the same construction with six.
    """
    if config is None:
        from .dlt import ChoConfig
        config = ChoConfig()
    n = int(config.beads_per_ring)
    p = np.asarray(points_indexed_px, dtype=float)
    if n < 6 or n % 2:
        raise ValueError("Cho analytic calibration needs an even bead count >= 6 per ring")
    if p.shape != (2 * n, 2):
        raise ValueError(f"expected indexed points with shape ({2 * n},2)")
    if ring_indices is None:
        ring_indices = (np.arange(n), np.arange(n, 2 * n))
    if opposite_shift is None:
        opposite_shift = n // 2
    if len(ring_indices[0]) != n or len(ring_indices[1]) != n:
        raise ValueError("ring_indices must contain every target in two equal rings")
    fits = [_fit_ellipse(p[idx]) for idx in ring_indices]
    # The origin-defining pairs are cross-ring pairs with opposite transverse
    # coordinates: (x,y,-h) <-> (-x,-y,+h).  Pairing points within one ring
    # instead produces lines that do not pass through the projected origin.
    lower, upper = ring_indices
    lines = []
    for k in range(n):
        lines.append(_homogeneous_line(p[lower[k]],
                                       p[upper[(k + opposite_shift) % n]]))
    lines = np.asarray(lines)
    piercing, line_rms = _intersect_lines(lines)

    extrema = {"ring_lower": _ellipse_extrema(fits[0]),
               "ring_upper": _ellipse_extrema(fits[1])}
    eta, pa, ellipse_lines, eta_diagnostics = _cho_eta_from_ellipses(
        (fits[0], fits[1]))
    complete_geometry = _complete_geometry_from_dlt(p, config)
    eta_diagnostics["dlt_decomposition_eta_rad"] = complete_geometry["cho_eta_rad"]
    eta_diagnostics["dlt_decomposition_eta_deg"] = complete_geometry["cho_eta_deg"]
    eta_diagnostics["analytic_minus_dlt_deg"] = float(
        np.degrees(eta - complete_geometry["cho_eta_rad"]))
    notes = [
        "Piercing point is obtained from the least-squares intersection of opposite-bead lines.",
        "Eta follows Cho 2005 Eqs. (14)-(16): ellipse axis ratios, analytic extrema, and distance-weighted line angles.",
        "The ellipse-major-axis average is not used as eta.",
        "The complete DLT pose independently supplies source, detector frame, SDD, principal point, phi, theta, eta, and gantry phase.",
    ]
    return ChoAnalyticFrame(fits[0], fits[1], piercing, lines, line_rms,
                            eta, np.degrees(eta), extrema, pa, ellipse_lines,
                            eta_diagnostics, complete_geometry, notes)


def frame_to_jsonable(frame: ChoAnalyticFrame) -> dict:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, EllipseFit):
            return {k: convert(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    return convert(asdict(frame))


def calibrate_stack_pic(points_stack_px: np.ndarray, config) -> list[dict]:
    """Apply the paper-style ellipse/line PIC stages to every indexed view."""
    return [frame_to_jsonable(calibrate_frame_cho(frame, config))
            for frame in np.asarray(points_stack_px, dtype=float)]


'''Legacy command-line wrapper; call ``calibrate_frame_cho`` or workflow instead.
if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from .tracker import detect_frame, initialize_indices
    from .dlt import ChoConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output", type=Path, default=Path("cho_analytic_frame0.json"))
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args()
    raw = np.memmap(args.raw, dtype=np.float32, mode="r", shape=(360, 1024, 1024))
    cfg = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
                    beads_per_ring=12, pixel_size_mm=(0.417, 0.417))
    frame = detect_frame(np.asarray(raw[0]), args.threshold, split_merged=False)
    indexed = initialize_indices(frame, cfg)
    result = calibrate_frame_cho(indexed, cfg)
    args.output.write_text(json.dumps(frame_to_jsonable(result), indent=2), encoding="utf-8")
    print(json.dumps(frame_to_jsonable(result), indent=2))
'''
