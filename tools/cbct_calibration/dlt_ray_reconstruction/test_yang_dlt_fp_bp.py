"""FP/BP reconstruction test using Yang DLT projection matrices.

Edit CONFIG and run this file from an IDE.  It uses only the projection
matrices produced by Yang calibration; no simulator geometry is used in the
projector.  The detector is deliberately sampled coarsely so the complete
test is practical in pure Python/SciPy.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import lsqr

from dlt_siddon import build_ray_matrix, load_projection_matrices


CONFIG = {
    "report_path": Path(
        "out/cbct_calibration/yang_standard_raw/yang_report.json"),
    "output_directory": Path("out/cbct_calibration/dlt_ray_reconstruction"),
    "pixel_size_mm": 0.417,
    "detector_samples": 48,
    "detector_u_range_px": (256.0, 768.0),
    "detector_v_range_px": (256.0, 768.0),
    "volume_shape_zyx": (32, 32, 32),
    "voxel_size_mm": 3.0,
    "volume_centre_xyz_mm": (0.0, 0.0, 0.0),
    "view_count": 36,
    "max_reprojection_rmse_px": 2.0,
    "max_sdd_error_mm": 20.0,
    "lsqr_iterations": 20,
}


def choose_views(report: dict, config: dict) -> np.ndarray:
    poses = report["poses"]
    rmse = np.asarray([pose["reprojection_rmse_px"] for pose in poses])
    sdd = np.asarray([pose["source_detector_distance_mm"] for pose in poses])
    good = np.isfinite(rmse) & np.isfinite(sdd)
    good &= rmse <= config["max_reprojection_rmse_px"]
    good &= np.abs(sdd - 770.0) <= config["max_sdd_error_mm"]
    candidates = np.flatnonzero(good)
    if len(candidates) < config["view_count"]:
        raise ValueError(f"only {len(candidates)} stable views are available")
    # Even angular spacing keeps the small demonstration well-conditioned.
    wanted = np.linspace(0, len(candidates) - 1,
                         config["view_count"]).round().astype(int)
    return candidates[wanted]


def make_phantom(shape: tuple[int, int, int], spacing: float) -> np.ndarray:
    """Create a compact object in the phantom-coordinate reconstruction frame."""
    z, y, x = np.indices(shape, dtype=float)
    centre = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    xyz = np.stack([(x - centre[2]) * spacing,
                    (y - centre[1]) * spacing,
                    (z - centre[0]) * spacing], axis=-1)
    radius = np.linalg.norm(xyz, axis=-1)
    volume = (radius <= 38.0).astype(float)
    for point, bead_radius, value in (
            ((-15.0, -12.0, 8.0), 8.0, 1.4),
            ((16.0, 10.0, -10.0), 6.0, 1.8),
            ((0.0, 20.0, 20.0), 5.0, 2.0)):
        distance = np.linalg.norm(xyz - np.asarray(point), axis=-1)
        volume[distance <= bead_radius] = value
    return volume


def save_slice(path: Path, volume: np.ndarray, title: str) -> None:
    mid = volume.shape[0] // 2
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(volume[mid], cmap="gray", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(config: dict = CONFIG) -> dict:
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(config["report_path"]).read_text(encoding="utf-8"))
    frame_ids = choose_views(report, config)
    projections = load_projection_matrices(config["report_path"], frame_ids)
    samples = config["detector_samples"]
    us = np.linspace(*config["detector_u_range_px"], samples)
    vs = np.linspace(*config["detector_v_range_px"], samples)
    detector_uv = np.asarray([(u, v) for v in vs for u in us])
    shape = tuple(config["volume_shape_zyx"])
    spacing = float(config["voxel_size_mm"])
    centre = np.asarray(config["volume_centre_xyz_mm"], dtype=float)
    print("building Siddon ray matrix ...")
    matrix = build_ray_matrix(projections, detector_uv, shape, spacing, centre)
    phantom = make_phantom(shape, spacing)
    truth = phantom.ravel()
    forward = np.asarray(matrix @ truth).ravel()
    rng = np.random.default_rng(1234)
    probe_volume = rng.normal(size=truth.size)
    probe_projection = rng.normal(size=forward.size)
    lhs = float(np.dot(matrix @ probe_volume, probe_projection))
    rhs = float(np.dot(probe_volume, matrix.T @ probe_projection))
    adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-12)
    backprojection = np.asarray(matrix.T @ forward).reshape(shape)
    solution = lsqr(matrix, forward, iter_lim=int(config["lsqr_iterations"]),
                    atol=1.0e-5, btol=1.0e-5, show=False)
    reconstruction = solution[0].reshape(shape)
    reconstructed_projection = np.asarray(matrix @ solution[0]).ravel()
    projection_rmse = float(np.sqrt(np.mean((reconstructed_projection - forward) ** 2)))
    correlation = float(np.corrcoef(truth, solution[0])[0, 1])
    np.save(output / "phantom.npy", phantom)
    np.save(output / "forward_projection.npy", forward.reshape(len(projections), samples, samples))
    np.save(output / "backprojection.npy", backprojection)
    np.save(output / "reconstruction.npy", reconstruction)
    save_slice(output / "phantom_slice.png", phantom, "known phantom")
    save_slice(output / "backprojection_slice.png", backprojection, "unfiltered BP (A^T A x)")
    save_slice(output / "reconstruction_slice.png", reconstruction,
               f"LSQR reconstruction, {len(projections)} views")
    result = {
        "frames": frame_ids.tolist(),
        "ray_count": int(matrix.shape[0]),
        "voxel_count": int(matrix.shape[1]),
        "nonzero_path_segments": int(matrix.nnz),
        "adjoint_relative_error": float(adjoint_error),
        "forward_projection_min_max": [float(forward.min()), float(forward.max())],
        "reconstruction_projection_rmse": projection_rmse,
        "truth_reconstruction_correlation": correlation,
        "lsqr_iterations_requested": int(config["lsqr_iterations"]),
        "lsqr_iterations_used": int(solution[2]),
        "note": "Projection matrices are from Yang calibration; rays use Siddon voxel path lengths.",
    }
    (output / "fp_bp_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()

