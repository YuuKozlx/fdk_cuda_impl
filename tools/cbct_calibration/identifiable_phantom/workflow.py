"""Calibration workflow for a phantom whose 3-D point IDs are identifiable.

The current example uses two unequal rings, a phase shift and a marker bead,
but the calibration principle is not limited to a double-ring shape.

RAW -> image-only tracking -> ordered centroids -> per-view DLT -> equivalent
geometry.  The tracker never receives scanner geometry or phantom pose.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

# 允许从本目录直接点击运行示例，同时复用上一级公共算法模块。
from .dlt import DltConfig, calibrate_view
from .joint_fit import fit
from .phantom import marked_points
from .source_circle import fit_source_circle
from .tracker import FormalTrackerConfig, track_stack


def calibrate_indexed(points_px, pixel_size_mm=(0.417, 0.417),
                      image_shape=(1024, 1024), max_nfev=1200,
                      marker_points_mm=None) -> dict:
    """DLT cameras -> source circle -> fixed-source joint calibration."""
    points = np.asarray(points_px, dtype=float)
    local = marked_points(marker_points_mm)
    dlt_config = DltConfig(pixel_size_mm=tuple(pixel_size_mm))
    cameras = [calibrate_view(frame, dlt_config, local) for frame in points]
    circle = fit_source_circle(np.asarray([c.source_phantom_mm for c in cameras]))
    joint = fit(points, pixel_size_mm, image_shape, max_nfev, local)
    return {
        "workflow": "indexed_points -> per_view_dlt -> source_circle -> fixed_source_joint_fit",
        "dlt_cameras": [{"projection_matrix": c.projection_matrix.tolist(),
                         "source_phantom_mm": c.source_phantom_mm.tolist(),
                         "sdd_mm": c.sdd_mm,
                         "principal_point_px": c.principal_point_px.tolist(),
                         "reprojection_rmse_px": c.reprojection_rmse_px}
                        for c in cameras],
        "source_circle": {"center_phantom_mm": circle.center_phantom_mm.tolist(),
                          "axis_phantom": circle.axis_phantom.tolist(),
                          "radius_mm": circle.radius_mm,
                          "radial_rms_mm": circle.radial_rms_mm,
                          "axial_rms_mm": circle.axial_rms_mm},
        "joint_fit": joint,
    }


@dataclass(frozen=True)
class IdentifiablePhantomConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    rows: int = 1024
    cols: int = 1024
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    threshold: float = 5.0
    gate_radius_px: float = 28.0
    min_pixels: int = 4
    marker_area_ratio: float = 1.45
    max_dlt_rmse_px: float = 1.0
    marker_points_mm: tuple[tuple[float, float, float], ...] = ((50.0, 0.0, 80.0),)


def preprocess_identifiable_phantom(config: IdentifiablePhantomConfig):
    """Extract all 25 indexed centroids without scanner-geometry seeds."""
    tracker = FormalTrackerConfig(
        raw_path=config.raw_path,
        output_directory=config.output_directory,
        views=config.views,
        rows=config.rows,
        cols=config.cols,
        threshold=config.threshold,
        pixel_size_mm=config.pixel_size_mm,
        gate_radius_px=config.gate_radius_px,
        min_pixels=config.min_pixels,
        marker_area_ratio=config.marker_area_ratio,
        max_dlt_rmse_px=config.max_dlt_rmse_px,
        marker_points_mm=config.marker_points_mm,
    )
    return track_stack(tracker)


def mirror_correspondence() -> np.ndarray:
    """Reflection ambiguity left by one marker on the phantom symmetry plane."""
    return np.array([
        0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
        11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 24,
    ], dtype=int)


def calibrate_identifiable_phantom(config: IdentifiablePhantomConfig) -> dict:
    """Fit reconstruction-oriented equivalent geometry with source fixed to zero."""
    tracking = preprocess_identifiable_phantom(config)
    if not np.all(tracking.observed):
        missing = int(np.size(tracking.observed) - np.count_nonzero(tracking.observed))
        raise RuntimeError(
            f"formal tracking contains {missing} missing observations; "
            "the dense joint fit requires a complete stack")
    identity = np.arange(tracking.points_px.shape[1], dtype=int)
    correspondence_candidates = []
    maps = [("as_tracked", identity)]
    if len(config.marker_points_mm) == 1:
        maps.append(("mirror_relabelled", mirror_correspondence()))
    for name, indices in maps:
        candidate_points = tracking.points_px[:, indices]
        candidate_report = calibrate_indexed(
            candidate_points, config.pixel_size_mm, (config.rows, config.cols),
            marker_points_mm=config.marker_points_mm)
        correspondence_candidates.append((
            candidate_report["joint_fit"]["diagnostics"]["rmse_px"], name, indices,
            candidate_points, candidate_report))
    _, correspondence_name, correspondence, canonical_points, report = min(
        correspondence_candidates, key=lambda item: item[0])
    canonical_observed = tracking.observed[:, correspondence]
    report.update({
        "workflow": "identifiable_phantom_equivalent_geometry",
        "input": {**asdict(config), "raw_path": str(config.raw_path),
                   "output_directory": str(config.output_directory)},
        "preprocessing": {"mode": "formal_image_only_tracking",
                          "uses_system_geometry": False,
                          "uses_scan_angles": False,
                          "uses_phantom_pose": False,
                          "views": int(tracking.points_px.shape[0]),
                          "point_count": int(tracking.points_px.shape[1]),
                          "anchor_frame": tracking.report["anchor_frame"],
                          "anchor_camera_rmse_px": tracking.report[
                              "anchor_initialization"][
                                  "selected_physical_camera_rmse_px"],
                          "tracking_dlt_rmse_mean_px": tracking.report[
                              "dlt_validation"]["rmse_mean_px"],
                          "tracking_dlt_rmse_max_px": tracking.report[
                              "dlt_validation"]["rmse_max_px"],
                          "selected_correspondence": correspondence_name,
                          "correspondence_candidate_rmse_px": {
                              name: float(rmse)
                              for rmse, name, _, _, _ in correspondence_candidates
                          },
                          "correspondence_index": correspondence.tolist(),
                          "fixed_source_offset_mm": [0.0, 0.0, 0.0],
                          "fixed_offset_n_mm": 0.0},
    })
    config.output_directory.mkdir(parents=True, exist_ok=True)
    np.save(config.output_directory / "points.npy", canonical_points)
    np.save(config.output_directory / "observed.npy", canonical_observed)
    (config.output_directory / "calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return report
