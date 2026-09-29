"""Validate Cho's analytic eta equations on a RAW-derived 360-view data set.

Run this file directly from an IDE.  The input is the indexed sub-pixel bead
array produced by ``cho_raw_example.py`` from the projection RAW.  Keeping the
image preprocessing result as the input makes this validation fast and, more
importantly, ensures that it exercises measured centroids rather than ideal
projected points.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cho_analytic import calibrate_frame_cho
from cho_calibration import ChoConfig


@dataclass(frozen=True)
class ValidationConfig:
    points_path: Path
    source_report_path: Path
    output_directory: Path
    ring_radius_mm: float = 50.0
    ring_half_spacing_mm: float = 50.0
    beads_per_ring: int = 12
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    reference_eta_deg: float | None = 3.0


CONFIG = ValidationConfig(
    points_path=Path(
        "out/cbct_calibration/cho_reference/double_ring_px10_py15_pz20/"
        "points_indexed.npy"),
    source_report_path=Path(
        "out/cbct_calibration/cho_reference/double_ring_px10_py15_pz20/"
        "cho_report.json"),
    output_directory=Path(
        "out/cbct_calibration/cho_reference/actual_data_eta_validation"),
)


def _wrap_line_angle_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    """Map an undirected line angle to [-90, 90) degrees."""
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 90.0) % 180.0 - 90.0


def _circular_summary_deg(values_deg: np.ndarray) -> dict:
    """Summarize 180-degree-periodic detector in-plane angles."""
    values = _wrap_line_angle_deg(values_deg)
    doubled = np.deg2rad(2.0 * values)
    mean = 0.5 * np.arctan2(np.mean(np.sin(doubled)),
                            np.mean(np.cos(doubled)))
    mean_deg = float(_wrap_line_angle_deg(np.rad2deg(mean)))
    residual = _wrap_line_angle_deg(values - mean_deg)
    return {
        "circular_mean_deg": mean_deg,
        "circular_std_deg": float(np.sqrt(np.mean(residual ** 2))),
        "median_deg": float(np.median(values)),
        "minimum_deg": float(np.min(values)),
        "maximum_deg": float(np.max(values)),
    }


def _save_figure(path: Path, analytic_eta: np.ndarray,
                 dlt_eta: np.ndarray, eta_difference: np.ndarray,
                 reprojection_rmse: np.ndarray, line_rms: np.ndarray,
                 reference_eta_deg: float | None) -> None:
    frames = np.arange(len(analytic_eta))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                constrained_layout=True)
    axes[0].plot(frames, analytic_eta, label="Cho Eqs. (14)-(16)", lw=1.2)
    axes[0].plot(frames, dlt_eta, label="DLT decomposition (mod 180 deg)",
                 lw=1.0, alpha=0.8)
    if reference_eta_deg is not None:
        axes[0].axhline(reference_eta_deg, color="black", ls="--", lw=1.0,
                        label=f"simulation reference {reference_eta_deg:g} deg")
    axes[0].set_ylabel("eta / deg")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(frames, eta_difference, color="tab:purple", lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("analytic - DLT / deg")
    axes[1].grid(alpha=0.25)

    axes[2].plot(frames, reprojection_rmse, label="DLT reprojection RMSE",
                 lw=1.0)
    axes[2].plot(frames, line_rms, label="opposite-line intersection RMS",
                 lw=1.0)
    axes[2].set(xlabel="view", ylabel="error / px")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(config: ValidationConfig = CONFIG) -> dict:
    points = np.load(config.points_path)
    expected_shape = (2 * config.beads_per_ring, 2)
    if points.ndim != 3 or tuple(points.shape[1:]) != expected_shape:
        raise ValueError(
            f"expected [views,{expected_shape[0]},2], got {points.shape}")

    source_report = json.loads(
        config.source_report_path.read_text(encoding="utf-8"))
    cho = ChoConfig(
        ring_radius_mm=config.ring_radius_mm,
        ring_half_spacing_mm=config.ring_half_spacing_mm,
        beads_per_ring=config.beads_per_ring,
        pixel_size_mm=config.pixel_size_mm,
        max_nfev=500,
    )

    analytic_eta = []
    dlt_eta = []
    reprojection_rmse = []
    line_rms = []
    failures = []
    for frame_id, measured in enumerate(points):
        try:
            result = calibrate_frame_cho(measured, cho)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failures.append({"frame": frame_id, "reason": str(exc)})
            continue
        analytic_eta.append(result.detector_in_plane_angle_deg)
        dlt_eta.append(result.complete_geometry["cho_eta_deg"])
        reprojection_rmse.append(
            result.complete_geometry["reprojection_rmse_px"])
        line_rms.append(result.line_fit_rms_px)

    analytic_eta = _wrap_line_angle_deg(np.asarray(analytic_eta))
    dlt_eta = _wrap_line_angle_deg(np.asarray(dlt_eta))
    eta_difference = _wrap_line_angle_deg(analytic_eta - dlt_eta)
    reprojection_rmse = np.asarray(reprojection_rmse)
    line_rms = np.asarray(line_rms)
    if not len(analytic_eta):
        raise RuntimeError("all actual-data frames failed Cho calibration")

    report = {
        "method": "Cho_2005_Eqs_14_to_16_actual_RAW_centroids",
        "raw_path": source_report.get("raw_path"),
        "points_path": str(config.points_path.resolve()),
        "input_is_ideal_projection": False,
        "input_description": (
            "sub-pixel indexed centroids extracted from the projection RAW"),
        "input_views": int(len(points)),
        "successful_views": int(len(analytic_eta)),
        "failed_views": failures,
        "angle_periodicity": "eta comparison is modulo 180 degrees",
        "analytic_eta": _circular_summary_deg(analytic_eta),
        "dlt_eta": _circular_summary_deg(dlt_eta),
        "analytic_minus_dlt": _circular_summary_deg(eta_difference),
        "dlt_reprojection_rmse_px": {
            "mean": float(np.mean(reprojection_rmse)),
            "median": float(np.median(reprojection_rmse)),
            "maximum": float(np.max(reprojection_rmse)),
        },
        "opposite_line_intersection_rms_px": {
            "mean": float(np.mean(line_rms)),
            "median": float(np.median(line_rms)),
            "maximum": float(np.max(line_rms)),
        },
        "reference_eta_deg_report_only": config.reference_eta_deg,
        "per_view": {
            "analytic_eta_deg": analytic_eta.tolist(),
            "dlt_eta_deg_modulo_180": dlt_eta.tolist(),
            "analytic_minus_dlt_deg": eta_difference.tolist(),
            "dlt_reprojection_rmse_px": reprojection_rmse.tolist(),
            "opposite_line_intersection_rms_px": line_rms.tolist(),
        },
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    report_path = config.output_directory / "cho_eta_actual_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    _save_figure(config.output_directory / "cho_eta_actual_data.png",
                 analytic_eta, dlt_eta, eta_difference,
                 reprojection_rmse, line_rms, config.reference_eta_deg)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "input_views": result["input_views"],
        "successful_views": result["successful_views"],
        "analytic_eta": result["analytic_eta"],
        "dlt_eta": result["dlt_eta"],
        "analytic_minus_dlt": result["analytic_minus_dlt"],
        "dlt_reprojection_rmse_px": result["dlt_reprojection_rmse_px"],
    }, indent=2, ensure_ascii=False))
