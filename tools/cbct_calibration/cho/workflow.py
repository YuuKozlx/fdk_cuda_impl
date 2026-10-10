"""Callable Cho workflow: tracking -> PIC/DLT -> source circle -> joint fit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..common import save_array, save_json
from .dlt import ChoConfig, calibrate_stack_masked
from .joint_fit import fit_with_handedness_search
from .source_circle import _jsonable, analyze_poses
from .pic import calibrate_stack_pic
from .tracker import track_raw_stack


@dataclass(frozen=True)
class ChoWorkflowConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    image_shape: tuple[int, int] = (1024, 1024)
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    ring_radius_mm: float = 50.0
    ring_half_spacing_mm: float = 50.0
    beads_per_ring: int = 12
    threshold: float = 5.0
    marker_points_mm: tuple[tuple[float, float, float], ...] = ((50.0, 0.0, 80.0),)

    def algorithm_config(self) -> ChoConfig:
        return ChoConfig(ring_radius_mm=self.ring_radius_mm,
                         ring_half_spacing_mm=self.ring_half_spacing_mm,
                         beads_per_ring=self.beads_per_ring,
                         pixel_size_mm=self.pixel_size_mm,
                         marker_points_mm=self.marker_points_mm,
                         robust_scale_px=0.15, max_nfev=500)


def calibrate_indexed_points(points_px: np.ndarray, config: ChoConfig,
                             image_shape: tuple[int, int] = (1024, 1024),
                             frame_ids: np.ndarray | None = None,
                             total_views: int | None = None,
                             usable_mask: np.ndarray | None = None) -> dict:
    """Calibrate an indexed stack without performing image preprocessing."""
    points = np.asarray(points_px, dtype=float)
    usable = (np.isfinite(points).all(axis=2) if usable_mask is None else
              np.asarray(usable_mask, dtype=bool) & np.isfinite(points).all(axis=2))
    if usable.shape != points.shape[:2]:
        raise ValueError("usable_mask must have shape (views, targets)")
    points, joint, handedness = fit_with_handedness_search(
        points, config, image_shape, frame_ids, total_views, usable)
    if handedness["selected"] == "ring_order_reversed":
        n = int(config.beads_per_ring)
        reverse = np.r_[np.mod(-np.arange(n), n),
                        n + np.mod(-np.arange(n), n)]
        usable = usable[:, reverse]
    poses = calibrate_stack_masked(points, config, usable)
    complete = np.all(usable, axis=1)
    pic_frames = [None] * len(points)
    if np.any(complete):
        complete_frames = calibrate_stack_pic(points[complete], config)
        for frame, value in zip(np.flatnonzero(complete), complete_frames):
            pic_frames[int(frame)] = value
    system = analyze_poses(poses, image_shape, config.pixel_size_mm)
    return {"method": "Cho_2005_PIC_plus_DLT_and_fixed_source_joint_fit",
            "image_coordinate_convention": "fdk-test: u-right, v-down",
            "correspondence_handedness": handedness,
            "usable_count_per_view": usable.sum(axis=1).astype(int).tolist(),
            "pic_skipped_frames": np.flatnonzero(~complete).astype(int).tolist(),
            "points_indexed_canonical": points,
            "pic": pic_frames,
            "dlt_projection_matrices": [pose.projection_matrix.tolist() for pose in poses],
            "source_circle_geometry": _jsonable(system),
            "fixed_source_joint_fit": asdict(joint)}


def calibrate_raw(config: ChoWorkflowConfig) -> dict:
    """Run the full Cho image-to-geometry chain and save stable outputs."""
    algorithm = config.algorithm_config()
    tracking = track_raw_stack(
        config.raw_path, algorithm, threshold=config.threshold,
        views=config.views, image_shape=config.image_shape)
    report = calibrate_indexed_points(tracking.points_px, algorithm, config.image_shape,
                                      tracking.frame_ids, config.views,
                                      tracking.usable)
    points = report.pop("points_indexed_canonical")
    report["input"] = {"raw_path": str(config.raw_path), "views": config.views,
                        "image_shape": list(config.image_shape),
                        "pixel_size_mm": list(config.pixel_size_mm)}
    report["tracking"] = {"valid_frame_ids": tracking.frame_ids.tolist(),
                          "rejected_frames": tracking.rejected_frames,
                          "diagnostics": tracking.diagnostics,
                          "observed": tracking.observed.tolist(),
                          "merged": tracking.merged.tolist(),
                          "usable": tracking.usable.tolist()}
    save_array(config.output_directory / "points_indexed.npy", points)
    save_json(config.output_directory / "calibration.json", report)
    return report
