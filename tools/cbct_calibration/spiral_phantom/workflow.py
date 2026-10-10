"""Linear marked-spiral calibration workflow and result visualization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..identifiable_phantom.workflow import calibrate_indexed
from .phantom import spiral_points
from .tracker import SpiralTrackerConfig, track_stack


@dataclass(frozen=True)
class SpiralCalibrationConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    rows: int = 1024
    cols: int = 1024
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    threshold: float = 5.0
    gate_radius_px: float = 28.0
    min_pixels: int = 4
    marker_area_ratio: float = 1.35
    max_dlt_rmse_px: float = 1.0
    max_nfev: int = 1200
    radius_mm: float = 50.0
    spacing_mm: float = 5.0
    phase_step_deg: float = 30.0
    bead_count: int = 24
    marker_below_mm: float = 10.0


def _tracker_config(config: SpiralCalibrationConfig) -> SpiralTrackerConfig:
    values = asdict(config)
    values.pop("max_nfev")
    return SpiralTrackerConfig(**values)


def _save_result_plot(report: dict, path: Path) -> None:
    cameras = report["dlt_cameras"]
    source = np.asarray([item["source_phantom_mm"] for item in cameras])
    rmse = np.asarray([item["reprojection_rmse_px"] for item in cameras])
    sdd = np.asarray([item["sdd_mm"] for item in cameras])
    principal = np.asarray([item["principal_point_px"] for item in cameras])
    machine = report["joint_fit"]["machine"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    axis = axes[0, 0]; axis.remove()
    axis = figure.add_subplot(2, 3, 1, projection="3d")
    axis.plot(source[:, 0], source[:, 1], source[:, 2], lw=.8)
    axis.scatter(source[::12, 0], source[::12, 1], source[::12, 2], s=8)
    center = (source.max(axis=0) + source.min(axis=0)) / 2
    half_width = max(float(np.ptp(source, axis=0).max()) * .55, 1.0)
    axis.set_xlim(center[0] - half_width, center[0] + half_width)
    axis.set_ylim(center[1] - half_width, center[1] + half_width)
    axis.set_zlim(center[2] - half_width, center[2] + half_width)
    axis.set_box_aspect((1, 1, 1))
    axis.set(title="DLT source trajectory in phantom coordinates",
             xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
    axes[0, 1].plot(rmse); axes[0, 1].set(title="Per-view DLT residual", xlabel="view", ylabel="RMSE / px")
    axes[0, 2].plot(sdd); axes[0, 2].set(title="Per-view physical-camera SDD", xlabel="view", ylabel="SDD / mm")
    axes[1, 0].plot(principal[:, 0], label="u0"); axes[1, 0].plot(principal[:, 1], label="v0")
    axes[1, 0].set(title="Per-view principal point", xlabel="view", ylabel="pixel"); axes[1, 0].legend()
    labels = ["SID", "SDD", "offset U", "offset V"]
    values = [machine["sid_mm"], machine["sdd_mm"], machine["offset_u_mm"], machine["offset_v_mm"]]
    bars = axes[1, 1].bar(labels, values)
    axes[1, 1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1, 1].set(title="Fixed-source joint fit: distances", ylabel="mm")
    labels = ["tilt U", "tilt V", "tilt N"]
    values = [machine["tilt_u_deg"], machine["tilt_v_deg"], machine["tilt_n_deg"]]
    bars = axes[1, 2].bar(labels, values)
    axes[1, 2].bar_label(bars, fmt="%.3f", padding=3)
    axes[1, 2].set(title="Fixed-source joint fit: detector pose", ylabel="degree")
    for item in axes.flat[1:]: item.grid(alpha=.2)
    figure.savefig(path, dpi=180); plt.close(figure)


def _save_phantom_plot(points: np.ndarray, path: Path) -> None:
    """Show the physical helix and its off-axis axial marker."""
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    ordinary, marker = points[:-1], points[-1]
    axis.plot(ordinary[:, 0], ordinary[:, 1], ordinary[:, 2],
              color="tab:blue", lw=1.0, alpha=.7)
    axis.scatter(ordinary[:, 0], ordinary[:, 1], ordinary[:, 2],
                 color="tab:blue", s=30, label="ordinary beads")
    axis.scatter(*marker, color="tab:red", s=100, marker="*",
                 label="axial marker M")
    for index, point in enumerate(ordinary):
        axis.text(*point, str(index), fontsize=7)
    axis.text(*marker, "M", fontsize=9)
    axis.set(title="Marked helical-bead phantom",
             xlabel="x / mm", ylabel="y / mm", zlabel="z / mm")
    axis.legend(loc="upper left")
    figure.savefig(path, dpi=180); plt.close(figure)


def calibrate_spiral_phantom(config: SpiralCalibrationConfig) -> dict:
    """RAW -> tracking/IDs -> DLT -> source circle -> fixed-source joint fit."""
    tracking = track_stack(_tracker_config(config))
    local = spiral_points(config.radius_mm, config.spacing_mm,
                          config.phase_step_deg, config.bead_count,
                          config.marker_below_mm)
    report = calibrate_indexed(
        tracking.points_px, config.pixel_size_mm, (config.rows, config.cols),
        config.max_nfev, local_points=local, usable_mask=tracking.usable)
    report.update({
        "workflow": "marked_spiral -> image_tracking -> indexed_3d_2d -> per_view_dlt -> source_circle -> fixed_source_joint_fit",
        "input": {**asdict(config), "raw_path": str(config.raw_path),
                  "output_directory": str(config.output_directory)},
        "phantom": {"point_count": len(local), "ordinary_bead_count": config.bead_count,
                    "marker_id": config.bead_count,
                    "points_mm": local.tolist()},
        "tracking": tracking.report,
    })
    config.output_directory.mkdir(parents=True, exist_ok=True)
    _save_phantom_plot(local, config.output_directory / "phantom_layout.png")
    (config.output_directory / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _save_result_plot(report, config.output_directory / "calibration.png")
    return report
