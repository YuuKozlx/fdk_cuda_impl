"""Flat-panel circular CBCT calibration from tracked BB centers (Wu 2011).

No detector out-of-plane tilt is modeled. See README.md for coordinates,
identifiability, data requirements, and deviations from the paper.

Coordinates and units:
  Ellipse fitting, principal point, SDD regression, phase recovery and SOD
  scale estimation all run in DETECTOR PIXEL coordinates. u0_px/v0_px are
  therefore in pixels. SDD is converted to mm at the very end by
  multiplying the pixel-domain D with the pixel pitch.

  In-plane rotation (eta) is applied about a rotation center. The first pass
  uses the image center, then the principal point is refined iteratively by
  re-using the previously solved principal point as the rotation center.

Bead selection:
  Each bead trajectory is fit to an ellipse. A bead is rejected if
    - the ellipse fit fails, or
    - the short/long axis ratio is below `min_axis_ratio`, or
    - the vertical span (v_max - v_min) is below `min_row_span_px`.
  Rejected and kept bead indices are recorded so distances_mm can be
  sliced by original bead id rather than by column position.

Bead-pair filtering:
  Only known positive separations are used. Pairs with |i-j| below
  `min_pair_index_gap` (default 2, i.e. adjacent beads excluded) are dropped
  because short separations have weak leverage and larger relative error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


# =====================================================================
# 数据容器
# =====================================================================

@dataclass
class Geometry:
    sdd_mm: float
    sod_mm: float
    u0_px: float
    v0_px: float
    eta_deg: float


@dataclass
class BeadSelection:
    used_ids: np.ndarray
    rejected_ids: np.ndarray
    axis_ratios: np.ndarray
    row_spans: np.ndarray
    fits: dict
    points_used: np.ndarray
    rejected_reason: dict


# =====================================================================
# 基础几何与线性代数工具
# =====================================================================

def rotation(eta_deg):
    t = np.deg2rad(eta_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def project(geometry, beads_mm, angles_rad, pixel_size_mm):
    """Return [view, bead, (u,v)], with pixel centers indexed from zero."""
    beads = np.asarray(beads_mm, dtype=float)
    c, s = np.cos(angles_rad)[:, None], np.sin(angles_rad)[:, None]
    x = c * beads[:, 0] - s * beads[:, 1]
    y = s * beads[:, 0] + c * beads[:, 1]
    denominator = geometry.sod_mm + x
    if np.any(denominator <= 0):
        raise ValueError("A bead is at or behind the source plane.")
    aligned = geometry.sdd_mm * np.stack(
        [-y / denominator, np.broadcast_to(beads[:, 2], x.shape) / denominator],
        axis=-1,
    )
    return (aligned @ rotation(geometry.eta_deg)) / pixel_size_mm + [
        geometry.u0_px, geometry.v0_px
    ]


def scaled_lstsq(matrix, target, label, max_condition=1e10):
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale < 1e-14):
        raise ValueError(f"{label}: degenerate configuration.")
    normalized = matrix / scale
    answer, _, rank, singular = np.linalg.lstsq(normalized, target, rcond=None)
    condition = float(singular[0] / max(singular[-1], 1e-300))
    if rank < matrix.shape[1] or condition > max_condition:
        raise ValueError(f"{label}: rank deficient or ill-conditioned ({condition:.3g}).")
    return answer / scale, condition


def fit_ellipse(points):
    """Normalized direct ellipse fit with 4*A*C-B**2 > 0 (Halir-Flusser).

    Return center and Q in the same units as `points` (pixels here).
    """
    origin = points.mean(axis=0)
    scale = points.std(axis=0)
    if np.any(scale < 1e-8):
        raise ValueError("Point/line trajectory: bead is on the axis or mid-plane.")
    x, y = ((points - origin) / scale).T
    quadratic = np.column_stack([x * x, x * y, y * y])
    linear = np.column_stack([x, y, np.ones_like(x)])
    design = np.column_stack([quadratic, linear])
    singular = np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / max(singular[-2], 1e-300))
    if condition > 1e10:
        raise ValueError("Ellipse fit is degenerate or ill-conditioned.")
    transform = -np.linalg.lstsq(linear, quadratic, rcond=None)[0]
    reduced = quadratic.T @ (quadratic + linear @ transform)
    constraint = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])
    _, vectors = np.linalg.eig(np.linalg.solve(constraint, reduced))
    candidates = []
    for vector in vectors.T:
        if np.max(abs(vector.imag)) > 1e-8:
            continue
        vector = vector.real
        ellipse_constraint = 4 * vector[0] * vector[2] - vector[1] ** 2
        if ellipse_constraint <= 0:
            continue
        vector = vector / np.sqrt(ellipse_constraint)
        coefficients = np.r_[vector, transform @ vector]
        candidates.append((np.linalg.norm(design @ coefficients), coefficients))
    if not candidates:
        raise ValueError("No nondegenerate ellipse fit found.")
    _, coefficients = min(candidates, key=lambda item: item[0])
    a, b, c, d, e, f = coefficients
    q = np.array([[a, b / 2], [b / 2, c]])
    if np.trace(q) < 0:
        q, d, e, f = -q, -d, -e, -f
    center = np.linalg.solve(q, -np.array([d, e]) / 2)
    level = center @ q @ center - f
    if level <= 0:
        raise ValueError("Ellipse has nonpositive squared axes.")
    q = q / level / np.outer(scale, scale)
    if np.linalg.eigvalsh(q)[0] <= 0:
        raise ValueError("Ellipse quadratic form is not positive definite.")
    return origin + center * scale, q, condition


# =====================================================================
# 输入校验
# =====================================================================

def validate(tracks_px, angles_rad, pixel_size_mm, distances_mm):
    tracks = np.asarray(tracks_px, dtype=float)
    angles = np.asarray(angles_rad, dtype=float)
    pitch = np.asarray(pixel_size_mm, dtype=float)
    distances = np.asarray(distances_mm, dtype=float)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks_px must have shape (views, beads, 2).")
    n, k, _ = tracks.shape
    if n < 12 or k < 3:
        raise ValueError("Need >=12 views and >=3 beads.")
    if not np.isfinite(tracks).all():
        raise ValueError("Missing/nonfinite tracks are unsupported; all supplied views must track every bead.")
    if angles.shape != (n,) or not np.isfinite(angles).all():
        raise ValueError("angles_rad must contain one finite angle per view.")
    unwrapped = np.unwrap(angles)
    step = np.diff(unwrapped)
    if not (np.all(step > 1e-10) or np.all(step < -1e-10)):
        raise ValueError("Angles must be strictly monotonic after unwrapping, without duplicates.")
    span = abs(unwrapped[-1] - unwrapped[0])
    if span > 2 * np.pi + 1e-10:
        raise ValueError("Supply at most one revolution of observations.")
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pixel_size_mm must be two positive finite pitches (du,dv).")
    if distances.shape != (k, k) or not np.allclose(distances, distances.T, equal_nan=True):
        raise ValueError("distances_mm must be a symmetric bead-by-bead matrix.")
    if np.any(np.isinf(distances)) or not np.allclose(np.diag(distances), 0):
        raise ValueError("Distance diagonal must be zero; use NaN for unknown distances.")
    i, j = np.triu_indices(k, 1)
    known = np.isfinite(distances[i, j])
    i, j = i[known], j[known]
    if not len(i) or np.any(distances[i, j] <= 0):
        raise ValueError("Need at least one known positive bead separation.")
    return tracks, angles, pitch, distances


# =====================================================================
# 步骤 1：椭圆拟合（像素坐标）+ 有效球筛选
# =====================================================================

def select_valid_beads(tracks, pixel_size, min_axis_ratio,
                       min_row_span_px=30.0):
    points = tracks
    n_beads = points.shape[1]

    fits = {}
    axis_ratios = np.full(n_beads, np.nan)
    row_spans = np.full(n_beads, np.nan)
    rejected_reason = {}

    for bead in range(n_beads):
        v = tracks[:, bead, 1]
        finite = np.isfinite(v)
        span = float(v[finite].max() - v[finite].min()) if finite.sum() >= 2 else 0.0
        row_spans[bead] = span

        if span < min_row_span_px:
            rejected_reason[bead] = (
                f"row span {span:.1f} px < {min_row_span_px} px"
            )
            continue

        try:
            center, q, condition = fit_ellipse(points[:, bead])
        except ValueError as exc:
            rejected_reason[bead] = f"ellipse fit failed: {exc}"
            continue

        eigenvalues = np.linalg.eigvalsh(q)
        axis_ratio = float(np.sqrt(eigenvalues[0] / eigenvalues[1]))
        axis_ratios[bead] = axis_ratio

        if axis_ratio < min_axis_ratio:
            rejected_reason[bead] = (
                f"axis ratio {axis_ratio:.4g} < {min_axis_ratio}"
            )
            continue

        fits[bead] = (center, q, condition)

    used_ids = np.array(sorted(fits))
    rejected_ids = np.setdiff1d(np.arange(n_beads), used_ids)

    if len(used_ids) < 3:
        raise ValueError(
            f"Only {len(used_ids)} usable beads (need >= 3). "
            f"Rejected {rejected_ids.tolist()}: {rejected_reason}"
        )

    return BeadSelection(
        used_ids=used_ids,
        rejected_ids=rejected_ids,
        axis_ratios=axis_ratios,
        row_spans=row_spans,
        fits=fits,
        points_used=points[:, used_ids],
        rejected_reason=rejected_reason,
    )


# =====================================================================
# 步骤 2：探测器面内旋转角 eta（绕给定旋转中心）
# =====================================================================

def estimate_inplane_angle(selection, rotation_center):
    axis_points = np.array([selection.fits[b][0] for b in selection.used_ids])
    _, singular_values, directions = np.linalg.svd(
        axis_points - axis_points.mean(axis=0)
    )
    if singular_values[0] < 1e-6:
        raise ValueError("Beads must span different axial heights.")

    direction = directions[0]
    if direction[1] < 0:
        direction = -direction
    eta_deg = float(np.rad2deg(np.arctan2(direction[0], direction[1])))

    quadratic_forms = np.array([selection.fits[b][1] for b in selection.used_ids])
    rot = rotation(eta_deg)
    c = np.asarray(rotation_center, dtype=float)

    centers_aligned = (axis_points - c) @ rot.T + c
    forms_aligned = rot @ quadratic_forms @ rot.T
    return eta_deg, centers_aligned, forms_aligned


# =====================================================================
# 步骤 3：SDD 与主点（像素域解，SDD 转 mm）
# =====================================================================

def estimate_sdd_and_principal_point(centers_aligned, forms_aligned,
                                     pixel_size):
    uc, vc = centers_aligned.T
    a = forms_aligned[:, 0, 0]
    b = forms_aligned[:, 1, 1]

    v_shift = vc.mean()
    w = vc - v_shift

    i, j = np.triu_indices(len(uc), 1)
    matrix = np.column_stack([2 * (w[i] - w[j]), a[i] / b[i] - a[j] / b[j]])
    target = w[i] ** 2 - w[j] ** 2 - (1 / b[i] - 1 / b[j])

    solution, condition = scaled_lstsq(matrix, target, "SDD / principal point")
    v0_px = float(solution[0] + v_shift)
    d_squared = float(solution[1])
    if d_squared <= 0:
        raise ValueError(
            "Nonphysical SDD: "
            f"sdd^2={d_squared:.4g}, condition={condition:.3g}. "
            "Check tracks, tilt, and bead layout."
        )

    d_px = float(np.sqrt(d_squared))
    sdd_mm = d_px * float(np.sqrt(pixel_size[0] * pixel_size[1]))
    u0_px = float(uc.mean())
    ellipse_params = np.column_stack([uc, vc, a, b])
    return sdd_mm, u0_px, v0_px, d_px, ellipse_params


# =====================================================================
# 步骤 4：SOD 与球的三维坐标
# =====================================================================

def recover_phase(bead_track, angles, d_px, u0_px, v0_px):
    u, v = (bead_track - [u0_px, v0_px]).T
    c, s = np.cos(angles), np.sin(angles)
    n = len(angles)

    matrix = np.vstack([
        np.column_stack([u * c + d_px * s, -u * s + d_px * c, np.zeros(n)]),
        np.column_stack([v * c, -v * s, np.full(n, -d_px)]),
    ])
    xyz, _ = scaled_lstsq(matrix, np.r_[-u, -v], "bead phase")
    return float(np.arctan2(xyz[1], xyz[0]))


def estimate_sod_and_beads(points, angles, rot, d_px, u0_px, v0_px,
                           pair_i, pair_j, known_distances,
                           centers_aligned, forms_aligned):
    n_beads = points.shape[1]
    aligned = points @ rot.T

    vc = centers_aligned[:, 1]
    a = forms_aligned[:, 0, 0]
    b = forms_aligned[:, 1, 1]

    rho = 1 / np.sqrt(1 + a * d_px ** 2)
    zeta = (vc - v0_px) * (1 - rho ** 2) / d_px

    bead_phases = np.empty(n_beads)
    bead_normalized = np.empty((n_beads, 3))
    for bead in range(n_beads):
        phase = recover_phase(aligned[:, bead], angles, d_px, u0_px, v0_px)
        bead_phases[bead] = phase
        bead_normalized[bead] = [rho[bead] * np.cos(phase),
                                 rho[bead] * np.sin(phase),
                                 zeta[bead]]

    separations = np.linalg.norm(
        bead_normalized[pair_i] - bead_normalized[pair_j], axis=1
    )
    if np.any(separations < 1e-10):
        raise ValueError("A known bead pair collapsed to the same position.")

    sod = float(np.dot(separations, known_distances) /
                np.dot(separations, separations))
    return bead_normalized * sod, bead_phases, sod


# =====================================================================
# 步骤 5：联合精修
# =====================================================================

def refine_solution(geometry, beads, angles, pixel_size,
                    tracks_used, pair_i, pair_j, known_distances,
                    center_sigma_px, distance_sigma_mm):
    if not (np.isfinite(center_sigma_px) and center_sigma_px > 0 and
            np.isfinite(distance_sigma_mm) and distance_sigma_mm > 0):
        raise ValueError("Uncertainty scales must be positive and finite.")

    from scipy.optimize import least_squares
    n_beads = beads.shape[0]

    initial = np.r_[geometry.sdd_mm, geometry.sod_mm,
                    geometry.u0_px, geometry.v0_px,
                    np.deg2rad(geometry.eta_deg), beads.ravel()]

    def unpack(parameters):
        g = Geometry(parameters[0], parameters[1], parameters[2],
                     parameters[3], float(np.rad2deg(parameters[4])))
        return g, parameters[5:].reshape(n_beads, 3)

    def residual(parameters):
        g, xyz = unpack(parameters)
        predicted = project(g, xyz, angles, pixel_size)
        spacing = np.linalg.norm(xyz[pair_i] - xyz[pair_j], axis=1)
        return np.r_[(predicted - tracks_used).ravel() / center_sigma_px,
                     (spacing - known_distances) / distance_sigma_mm]

    lower = np.full(initial.size, -np.inf)
    lower[:2] = 1e-6
    optimized = least_squares(residual, initial, bounds=(lower, np.inf),
                              x_scale="jac", max_nfev=500,
                              ftol=1e-11, xtol=1e-11, gtol=1e-9)
    if not optimized.success:
        raise ValueError(f"Refinement did not converge: {optimized.message}")

    geometry, beads = unpack(optimized.x)
    jacobian = optimized.jac / np.maximum(
        np.linalg.norm(optimized.jac, axis=0), 1e-30
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    optimization = {
        "success": True,
        "nfev": int(optimized.nfev),
        "scaled_jacobian_condition": float(singular[0] / singular[-1]),
    }
    return geometry, beads, optimization


# =====================================================================
# 步骤 6：残差与诊断
# =====================================================================

def collect_warnings(geometry, angles, selection, residual_px):
    warnings = []
    used, rejected = selection.used_ids, selection.rejected_ids

    if len(rejected):
        warnings.append(
            f"Rejected {len(rejected)} bead(s) {rejected.tolist()} as "
            f"degenerate/too-flat; used {len(used)} of {len(selection.axis_ratios)}."
        )
    if len(used) < 4:
        warnings.append("Fewer than 4 usable beads: calibration may be weakly constrained.")
    if np.any(selection.axis_ratios[used] < 0.05):
        warnings.append("Some used ellipses are very flat: noise sensitive.")
    if np.any(selection.row_spans[used] < 40):
        warnings.append("Some used beads have small v-span: SDD regression may be ill-conditioned.")
    coverage = abs(np.rad2deg(np.sum(np.diff(np.unwrap(angles)))))
    if coverage < 300:
        warnings.append("Limited angular coverage: ellipse initialization is noise sensitive.")
    if np.sqrt(np.mean(residual_px ** 2)) > 0.5:
        warnings.append("Large reprojection error: inspect tracking, angles, and tilt.")
    if geometry.sdd_mm <= geometry.sod_mm:
        warnings.append("SDD <= SOD: recovered geometry is nonphysical.")
    return warnings


def compute_diagnostics(geometry, beads, angles, tracks_used, predicted,
                        pair_i, pair_j, known_distances,
                        selection, optimization):
    residual_px = predicted - tracks_used
    distance_error = (np.linalg.norm(beads[pair_i] - beads[pair_j], axis=1)
                      - known_distances)
    return {
        "n_beads_nominal": int(len(selection.axis_ratios)),
        "n_beads_used": int(len(selection.used_ids)),
        "beads_used_index": selection.used_ids.tolist(),
        "beads_rejected_index": selection.rejected_ids.tolist(),
        "beads_rejected_reason": {
            int(b): selection.rejected_reason[b]
            for b in selection.rejected_ids.tolist()
        },
        "axis_ratios": selection.axis_ratios.tolist(),
        "row_spans": selection.row_spans.tolist(),
        "n_known_pairs": int(len(pair_i)),
        "rmse_px": float(np.sqrt(np.mean(residual_px ** 2))),
        "max_point_error_px": float(np.linalg.norm(residual_px, axis=-1).max()),
        "rmse_per_bead_px": np.sqrt(np.mean(residual_px ** 2, axis=(0, 2))).tolist(),
        "distance_rmse_mm": float(np.sqrt(np.mean(distance_error ** 2))),
        "optimization": optimization,
        "warnings": collect_warnings(geometry, angles, selection, residual_px),
    }


# =====================================================================
# 组装结果
# =====================================================================

def build_result(geometry, beads, bead_phases, selection,
                 diagnostics, pixel_size, angles):
    steps = np.diff(np.unwrap(angles))
    uniform = np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9)
    return {
        "stages": {
            "stage1_ellipse_and_inplane_angle": {
                "eta_deg": float(geometry.eta_deg),
                "used_bead_ids": selection.used_ids.tolist(),
                "rejected_bead_ids": selection.rejected_ids.tolist(),
            },
            "stage2_sdd_and_principal_point": {
                "sdd_mm": float(geometry.sdd_mm),
                "principal_point_px": [float(geometry.u0_px), float(geometry.v0_px)],
            },
            "stage3_sod_and_bead_positions": {
                "sod_mm": float(geometry.sod_mm),
                "beads_mm": np.asarray(beads).tolist(),
                "bead_phases_rad": np.asarray(bead_phases).tolist(),
            },
            "stage4_joint_refinement": {
                "enabled": diagnostics.get("optimization") is not None,
                "details": diagnostics.get("optimization"),
            },
        },
        "geometry": asdict(geometry),
        "beads_mm": beads.tolist(),
        "beads_used_index": selection.used_ids.tolist(),
        "beads_rejected_index": selection.rejected_ids.tolist(),
        "beads_rejected_reason": {
            int(b): selection.rejected_reason[b]
            for b in selection.rejected_ids.tolist()
        },
        "pixel_size_mm": np.asarray(pixel_size).tolist(),
        "bead_phases_rad": bead_phases.tolist(),
        "delta_beta_deg": float(np.rad2deg(steps[0])) if uniform else None,
        "angular_coverage_deg": float(abs(np.rad2deg(np.sum(steps)))),
        "angle_step_source": "supplied per-view angles; not independently calibrated",
        "model": "flat panel, circular orbit, constant geometry, zero out-of-plane tilt",
        "initializer": "general ellipse fits and total-least-squares center line",
        "diagnostics": diagnostics,
    }


# =====================================================================
# 主流程
# =====================================================================

def calibrate(tracks_px, angles_rad, pixel_size_mm, distances_mm, *,
              refine=False, center_sigma_px=0.1, distance_sigma_mm=0.001,
              min_axis_ratio=0.02, min_row_span_px=30.0, image_size=None,
              max_rotation_iterations=5, rotation_tolerance_px=1e-4,
              min_pair_index_gap=2):
    """从跟踪到的 BB 球中心轨迹标定平板圆轨迹 CBCT 几何。

    主点迭代：先用 image_center 作旋转中心，再用解出的主点重转，
    直到主点位移小于 rotation_tolerance_px 或到 max_rotation_iterations。

    球对过滤：只保留 |i-j| >= min_pair_index_gap 的已知正距离对。
    """
    tracks, angles, pixel_size, distances = validate(
        tracks_px, angles_rad, pixel_size_mm, distances_mm
    )

    if image_size is None:
        raise ValueError("image_size=(rows, cols) is required for centered in-plane rotation.")
    rows, cols = image_size
    image_center = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])

    # --- 1. 选球 ---
    selection = select_valid_beads(
        tracks, pixel_size, min_axis_ratio, min_row_span_px
    )
    used_ids = selection.used_ids
    points_px = selection.points_used
    n_beads = points_px.shape[1]

    # 已知正距离对，按索引差过滤
    all_i, all_j = np.triu_indices(n_beads, 1)
    d_full = distances[used_ids[all_i], used_ids[all_j]]
    known_mask = np.isfinite(d_full) & (d_full > 0)
    gap_mask = np.abs(all_i - all_j) >= min_pair_index_gap
    keep_mask = known_mask & gap_mask
    pair_i = all_i[keep_mask]
    pair_j = all_j[keep_mask]
    known_distances = d_full[keep_mask]
    if len(pair_i) == 0:
        raise ValueError(
            "No known positive bead separations after bead selection and "
            f"min_pair_index_gap={min_pair_index_gap} filtering."
        )

    # --- 2/3. 迭代：用解出的主点做旋转中心 ---
    rotation_center = image_center.copy()
    eta_deg, centers_aligned, forms_aligned = estimate_inplane_angle(
        selection, rotation_center
    )
    sdd_mm, u0_px, v0_px, d_px, ellipse_params = \
        estimate_sdd_and_principal_point(
            centers_aligned, forms_aligned, pixel_size
        )

    rotation_iterations = 0
    for _ in range(max_rotation_iterations):
        new_center = np.array([u0_px, v0_px], dtype=float)
        shift = float(np.linalg.norm(new_center - rotation_center))
        if shift < rotation_tolerance_px:
            break
        rotation_center = new_center
        eta_deg, centers_aligned, forms_aligned = estimate_inplane_angle(
            selection, rotation_center
        )
        sdd_mm, u0_px, v0_px, d_px, ellipse_params = \
            estimate_sdd_and_principal_point(
                centers_aligned, forms_aligned, pixel_size
            )
        rotation_iterations += 1

    rot = rotation(eta_deg)

    # --- 4. SOD 与球坐标 ---
    beads, bead_phases, sod = estimate_sod_and_beads(
        points_px, angles, rot, d_px, u0_px, v0_px,
        pair_i, pair_j, known_distances,
        centers_aligned, forms_aligned,
    )

    geometry = Geometry(sdd_mm, sod, u0_px, v0_px, eta_deg)

    # --- 5. 可选精修 ---
    optimization = None
    if refine:
        geometry, beads, optimization = refine_solution(
            geometry, beads, angles, pixel_size,
            tracks[:, used_ids], pair_i, pair_j, known_distances,
            center_sigma_px, distance_sigma_mm,
        )

    # --- 6. 残差与诊断 ---
    predicted = project(geometry, beads, angles, pixel_size)
    diagnostics = compute_diagnostics(
        geometry, beads, angles, tracks[:, used_ids], predicted,
        pair_i, pair_j, known_distances, selection, optimization,
    )
    diagnostics["rotation_center_px"] = rotation_center.tolist()
    diagnostics["rotation_iterations"] = int(rotation_iterations)
    diagnostics["min_pair_index_gap"] = int(min_pair_index_gap)

    result = build_result(geometry, beads, bead_phases, selection,
                          diagnostics, pixel_size, angles)
    return result, predicted


# =====================================================================
# 演示数据
# =====================================================================

def demo_data(noise_px=0.1, seed=2026, drop_bead=None):
    truth = Geometry(300.0, 180.0, 255.5, 127.5, 2.0)
    beads = np.column_stack([np.full(5, 24.0), np.full(5, 8.0),
                             np.arange(5) * 6.0 - 10.0])
    if drop_bead is not None:
        beads[drop_bead, 2] = 0.0
    angles = np.arange(360) * (2 * np.pi / 360)
    pitch = np.array([0.25, 0.25])
    clean = project(truth, beads, angles, pitch)
    tracks = clean + np.random.default_rng(seed).normal(0, noise_px, clean.shape)
    distances = np.linalg.norm(beads[:, None] - beads[None, :], axis=-1)
    return {"tracks_px": tracks, "angles_rad": angles, "pixel_size_mm": pitch,
            "distances_mm": distances, "image_size": (256, 512),
            "min_row_span_px": 0.0}, truth


# =====================================================================
# 绘图
# =====================================================================

def plot_result(path, data, predicted, used_ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tracks = data["tracks_px"][:, used_ids]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for k in range(tracks.shape[1]):
        line, = axes[0].plot(predicted[:, k, 0], predicted[:, k, 1], lw=1.4,
                             label=f"BB {used_ids[k]}")
        axes[0].scatter(tracks[::6, k, 0], tracks[::6, k, 1], s=5,
                        color=line.get_color(), alpha=0.5)
    axes[0].set(xlabel="u (pixel)", ylabel="v (pixel, upward)",
                title="Measured centers and calibrated trajectories", aspect="equal")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                   ncol=min(5, tracks.shape[1]), frameon=False)
    error = predicted - tracks
    for coordinate, color in [(0, "#247ba0"), (1, "#c8553d")]:
        axes[1].hist(error[..., coordinate].ravel(), bins=40, alpha=0.6,
                     color=color, label=["u", "v"][coordinate])
    axes[1].set(xlabel="Reprojection residual (pixel)", ylabel="Count",
                title="Residuals (per coordinate)")
    axes[1].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# =====================================================================
# 命令行
# =====================================================================

'''Legacy command-line wrapper; call ``calibrate`` from Python instead.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--input", type=Path, help="NPZ with tracks, angles, pitches, distances")
    parser.add_argument("--output", type=Path, default=Path("out/cbct_calibration"))
    parser.add_argument("--refine", action="store_true", help="SciPy joint reprojection refinement")
    parser.add_argument("--plot", action="store_true", help="Write a Matplotlib diagnostic figure")
    parser.add_argument("--noise-px", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-axis-ratio", type=float, default=0.02)
    parser.add_argument("--min-row-span-px", type=float, default=30.0)
    parser.add_argument("--drop-bead", type=int, default=None)
    parser.add_argument("--center-sigma-px", type=float, default=0.1)
    parser.add_argument("--distance-sigma-mm", type=float, default=0.001)
    parser.add_argument("--image-rows", type=int, default=None,
                        help="Detector rows (for eta rotation center)")
    parser.add_argument("--image-cols", type=int, default=None,
                        help="Detector cols (for eta rotation center)")
    parser.add_argument("--max-rotation-iterations", type=int, default=5,
                        help="Max iterations for principal-point rotation center")
    parser.add_argument("--rotation-tolerance-px", type=float, default=1e-4,
                        help="Stop when principal point moves less than this (px)")
    parser.add_argument("--min-pair-index-gap", type=int, default=2,
                        help="Exclude bead pairs with |i-j| below this value (2 = skip adjacent)")
    args = parser.parse_args()

    truth = None
    if args.demo:
        if not np.isfinite(args.noise_px) or args.noise_px < 0:
            parser.error("--noise-px must be finite and nonnegative")
        data, truth = demo_data(args.noise_px, args.seed, args.drop_bead)
        image_size = (512, 512)
    else:
        with np.load(args.input, allow_pickle=False) as archive:
            data = {key: archive[key] for key in
                    ("tracks_px", "angles_rad", "pixel_size_mm", "distances_mm")}
        if args.image_rows is None or args.image_cols is None:
            parser.error("--image-rows and --image-cols are required for --input")
        image_size = (args.image_rows, args.image_cols)

    result, predicted = calibrate(
        **data, refine=args.refine,
        center_sigma_px=args.center_sigma_px,
        distance_sigma_mm=args.distance_sigma_mm,
        min_axis_ratio=args.min_axis_ratio,
        min_row_span_px=args.min_row_span_px,
        image_size=image_size,
        max_rotation_iterations=args.max_rotation_iterations,
        rotation_tolerance_px=args.rotation_tolerance_px,
        min_pair_index_gap=args.min_pair_index_gap,
    )
    if truth:
        result["simulation_truth"] = asdict(truth)
        result["simulation_noise_sigma_px"] = args.noise_px

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "observations.npz", **data)
    np.savez_compressed(args.output / "reprojection.npz", predicted_px=predicted,
                        residual_px=predicted - data["tracks_px"][:, result["beads_used_index"]])
    (args.output / "calibration.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if args.plot:
        plot_result(args.output / "calibration.png", data, predicted,
                    result["beads_used_index"])

    print(json.dumps({"geometry": result["geometry"],
                      "diagnostics": result["diagnostics"]},
                     indent=2, allow_nan=False))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
'''
