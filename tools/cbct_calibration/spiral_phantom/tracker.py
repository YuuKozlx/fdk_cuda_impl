"""Image-only tracking and numbering for the marked helical-bead phantom.

No scanner geometry, scan angle, or phantom pose is used here.  The known
helix layout is used only to score candidate 3-D/2-D label correspondences.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ..identifiable_phantom.dlt import DltConfig, calibrate_view
from ..math.dlt import normalized_dlt, project_points
from .phantom import spiral_points


@dataclass(frozen=True)
class SpiralTrackerConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    rows: int = 1024
    cols: int = 1024
    threshold: float = 5.0
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    gate_radius_px: float = 28.0
    min_pixels: int = 4
    marker_area_ratio: float = 1.35
    max_dlt_rmse_px: float = 1.0
    radius_mm: float = 50.0
    spacing_mm: float = 5.0
    phase_step_deg: float = 30.0
    bead_count: int = 24
    marker_below_mm: float = 10.0


@dataclass
class Component:
    pixels: np.ndarray
    weights: np.ndarray
    centroid: np.ndarray
    area: int


@dataclass
class TrackingResult:
    points_px: np.ndarray
    observed: np.ndarray
    merged: np.ndarray
    usable: np.ndarray
    report: dict


def _world(config: SpiralTrackerConfig) -> np.ndarray:
    return spiral_points(config.radius_mm, config.spacing_mm,
                         config.phase_step_deg, config.bead_count,
                         config.marker_below_mm)


def _load_raw(config: SpiralTrackerConfig) -> np.memmap:
    expected = config.views * config.rows * config.cols * 4
    if not config.raw_path.exists():
        raise FileNotFoundError(config.raw_path)
    if config.raw_path.stat().st_size != expected:
        raise ValueError(f"RAW size mismatch: expected {expected}, got "
                         f"{config.raw_path.stat().st_size}")
    return np.memmap(config.raw_path, dtype="<f4", mode="r",
                     shape=(config.views, config.rows, config.cols))


def detect_components(image: np.ndarray, threshold: float,
                      min_pixels: int = 4) -> list[Component]:
    labels, count = ndimage.label(
        np.asarray(image) > threshold, structure=np.ones((3, 3), dtype=int))
    result = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) < min_pixels:
            continue
        pixels = np.column_stack([xx, yy]).astype(float)
        weights = np.maximum(np.asarray(image)[yy, xx].astype(float) - threshold,
                             1.0e-6)
        centroid = np.sum(pixels * weights[:, None], axis=0) / np.sum(weights)
        result.append(Component(pixels, weights, centroid, len(xx)))
    return result


def choose_anchor(raw: np.ndarray, config: SpiralTrackerConfig) -> tuple[int, list[dict]]:
    """Find a view in which all beads are separated and the marker is distinct."""
    expected = config.bead_count + 1
    scores = []
    for frame in range(config.views):
        components = detect_components(raw[frame], config.threshold,
                                       config.min_pixels)
        item = {"frame": frame, "component_count": len(components),
                "anchor_eligible": False}
        if len(components) == expected:
            areas = np.array([c.area for c in components], dtype=float)
            ratio = float(areas.max() / max(np.median(areas), 1.0))
            item.update(marker_area_ratio=ratio,
                        anchor_score=float(np.median(areas) + 25 * min(ratio, 4)),
                        anchor_eligible=ratio >= config.marker_area_ratio)
        scores.append(item)
    eligible = [item for item in scores if item["anchor_eligible"]]
    if not eligible:
        raise ValueError(f"no frame contains {expected} separated targets")
    selected = max(eligible, key=lambda item: item["anchor_score"])
    return int(selected["frame"]), scores


def _candidate_orders(components: list[Component], bead_count: int):
    """Enumerate all distinct 1-D projection orders and marker candidates."""
    marker_candidates = np.argsort([-c.area for c in components])[:4]
    # A point order changes only when the sorting axis crosses a pairwise
    # perpendicular direction.  A dense angular sweep is simpler and gives
    # the same useful set for 25 image points.
    angles = np.deg2rad(np.arange(0.0, 180.0, 2.0))
    seen = set()
    for marker_id in marker_candidates:
        marker = components[int(marker_id)].centroid
        ordinary = np.asarray([c.centroid for i, c in enumerate(components)
                               if i != marker_id], dtype=float)
        for angle in angles:
            axis = np.array([np.cos(angle), np.sin(angle)])
            order = np.argsort(ordinary @ axis)
            for reverse in (False, True):
                index = order[::-1] if reverse else order
                key = (int(marker_id), *index.tolist())
                if key in seen:
                    continue
                seen.add(key)
                yield np.vstack([ordinary[index], marker])


def initialize_indices(image: np.ndarray, config: SpiralTrackerConfig
                       ) -> tuple[np.ndarray, dict]:
    """Number one separated view by DLT residual, not by scanner truth."""
    components = detect_components(image, config.threshold, config.min_pixels)
    expected = config.bead_count + 1
    if len(components) != expected:
        raise ValueError(f"anchor needs {expected} components, found {len(components)}")
    world = _world(config)
    projective = []
    for candidate in _candidate_orders(components, config.bead_count):
        try:
            projection, _ = normalized_dlt(world, candidate)
            error = project_points(world, projection) - candidate
            projective.append((float(np.sqrt(np.mean(error * error))), candidate))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    projective.sort(key=lambda item: item[0])
    camera_config = DltConfig(pixel_size_mm=config.pixel_size_mm,
                              robust_scale_px=0.15, max_nfev=300)
    refined = []
    for _, candidate in projective[:48]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pose = calibrate_view(candidate, camera_config, world)
            refined.append((pose.reprojection_rmse_px, candidate))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    if not refined:
        raise RuntimeError("no physical camera candidate survived")
    rmse, points = min(refined, key=lambda item: item[0])
    if rmse > config.max_dlt_rmse_px:
        raise RuntimeError(f"anchor numbering failed: RMSE={rmse:.4f} px")
    marker_distances = np.linalg.norm(
        np.asarray([c.centroid for c in components]) - points[-1], axis=1)
    marker_component = int(np.argmin(marker_distances))
    return points, {
        "component_count": len(components),
        "candidate_count": len(projective),
        "physical_camera_candidate_count": len(refined),
        "selected_physical_camera_rmse_px": float(rmse),
        "marker_component": marker_component,
        "marker_area_rank": int(np.where(
            np.argsort([-c.area for c in components]) == marker_component)[0][0]),
        "numbering_rule": "marker candidate + projected-axis order, selected by physical-camera RMSE",
        "uses_system_geometry": False,
    }


def _extract_from_predictions(image: np.ndarray, predicted: np.ndarray,
                              config: SpiralTrackerConfig):
    """Assign foreground pixels to predicted IDs, including merged components."""
    foreground = np.asarray(image) > config.threshold
    labels, _ = ndimage.label(foreground, structure=np.ones((3, 3), dtype=int))
    yy, xx = np.nonzero(foreground)
    output = np.full_like(predicted, np.nan, dtype=float)
    valid = np.zeros(len(predicted), dtype=bool)
    finite_seeds = np.isfinite(predicted).all(axis=1)
    if not len(xx) or not finite_seeds.any():
        return output, valid, {"foreground_pixels": len(xx), "valid_count": 0,
                               "merged_ids": []}
    pixels = np.column_stack([xx, yy]).astype(float)
    values = np.maximum(np.asarray(image)[yy, xx] - config.threshold, 1.0e-6)
    owner = np.empty(len(pixels), dtype=np.int32)
    distance = np.empty(len(pixels), dtype=float)
    for start in range(0, len(pixels), 65536):
        stop = min(start + 65536, len(pixels))
        squared = np.sum((pixels[start:stop, None] - predicted[None]) ** 2,
                         axis=2)
        squared[:, ~finite_seeds] = np.inf
        owner[start:stop] = np.argmin(squared, axis=1)
        distance[start:stop] = np.sqrt(np.min(squared, axis=1))
    accepted = distance <= config.gate_radius_px
    nearest = ndimage.distance_transform_edt(
        ~foreground, return_distances=False, return_indices=True)
    seed_labels = np.zeros(len(predicted), dtype=int)
    for target in np.flatnonzero(finite_seeds):
        u, v = np.rint(predicted[target]).astype(int)
        if 0 <= u < labels.shape[1] and 0 <= v < labels.shape[0]:
            ny, nx = nearest[:, v, u]
            if np.linalg.norm(predicted[target] - [nx, ny]) <= config.gate_radius_px:
                seed_labels[target] = int(labels[ny, nx])
    merged = np.zeros(len(predicted), dtype=bool)
    for label in np.unique(seed_labels[seed_labels > 0]):
        members = np.flatnonzero(seed_labels == label)
        if len(members) > 1:
            merged[members] = True
    counts = []
    for target in range(len(predicted)):
        selected = accepted & (owner == target)
        count = int(np.count_nonzero(selected)); counts.append(count)
        if count < config.min_pixels:
            continue
        weights = values[selected]
        output[target] = np.sum(pixels[selected] * weights[:, None], axis=0) / np.sum(weights)
        valid[target] = True
    return output, valid, {"foreground_pixels": len(pixels),
                           "valid_count": int(valid.sum()),
                           "pixel_counts_min": int(min(counts)),
                           "pixel_counts_max": int(max(counts)),
                           "merged_ids": np.flatnonzero(merged).tolist()}


def _track_direction(raw: np.ndarray, anchor: int, anchor_points: np.ndarray,
                    direction: int, config: SpiralTrackerConfig):
    points = np.full((config.views, len(anchor_points), 2), np.nan)
    observed = np.zeros((config.views, len(anchor_points)), dtype=bool)
    merged = np.zeros_like(observed)
    points[anchor] = anchor_points; observed[anchor] = True
    previous, previous_previous = anchor_points.copy(), None
    active = np.ones(len(anchor_points), dtype=bool)
    diagnostics = []
    for frame in range(anchor + direction,
                       config.views if direction > 0 else -1, direction):
        predicted = (previous if previous_previous is None else
                     previous + previous - previous_previous)
        inside = (active & np.isfinite(predicted).all(axis=1) &
                  (predicted[:, 0] >= 0) & (predicted[:, 0] < config.cols) &
                  (predicted[:, 1] >= 0) & (predicted[:, 1] < config.rows))
        measured, valid, stats = _extract_from_predictions(
            raw[frame], predicted, config)
        valid &= inside
        frame_merged = np.zeros(len(anchor_points), dtype=bool)
        frame_merged[stats["merged_ids"]] = True
        frame_merged &= inside
        # A merged component is not an independent centroid measurement, but
        # it is positive evidence that the targets still exist.  Keep those
        # tracks alive and use the motion prediction only to bridge this frame.
        measured[frame_merged & ~valid] = predicted[frame_merged & ~valid]
        active &= valid | frame_merged
        measured[~active] = np.nan
        points[frame], observed[frame] = measured, active
        merged[frame] = active & frame_merged
        diagnostics.append({"frame": frame, "direction": direction,
                            "active_count": int(active.sum()), **stats})
        previous_previous, previous = previous, measured
    return points, observed, merged, diagnostics


def _save_diagnostics(path: Path, raw: np.ndarray, points: np.ndarray,
                      anchor: int, dlt_rmse: list[float | None]):
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    axes[0].imshow(raw[anchor], cmap="gray", vmin=0, vmax=21)
    axes[0].scatter(points[anchor, :-1, 0], points[anchor, :-1, 1],
                    facecolors="none", edgecolors="cyan", s=36)
    axes[0].scatter(*points[anchor, -1], facecolors="none", edgecolors="red", s=80)
    for index, (u, v) in enumerate(points[anchor]):
        axes[0].text(u + 3, v, "M" if index == len(points[anchor]) - 1 else str(index),
                     color="yellow", fontsize=7)
    axes[0].set(title=f"Anchor frame {anchor}: inferred IDs")
    for index in range(points.shape[1] - 1):
        axes[1].plot(points[:, index, 0], points[:, index, 1], lw=.7)
    axes[1].plot(points[:, -1, 0], points[:, -1, 1], color="red", lw=1.5)
    axes[1].invert_yaxis(); axes[1].grid(alpha=.2)
    axes[1].set(title="Tracked detector trajectories", xlabel="u / px", ylabel="row / px")
    values = np.array([np.nan if x is None else x for x in dlt_rmse])
    axes[2].plot(values); axes[2].axhline(1, color="red", ls="--", lw=.8)
    axes[2].grid(alpha=.2)
    axes[2].set(title="Per-frame DLT validation", xlabel="view", ylabel="RMSE / px")
    figure.savefig(path, dpi=180); plt.close(figure)


def track_stack(config: SpiralTrackerConfig) -> TrackingResult:
    raw = _load_raw(config)
    anchor, anchor_scan = choose_anchor(raw, config)
    anchor_points, anchor_info = initialize_indices(raw[anchor], config)
    forward, forward_valid, forward_merged, forward_diag = _track_direction(
        raw, anchor, anchor_points, +1, config)
    backward, backward_valid, backward_merged, backward_diag = _track_direction(
        raw, anchor, anchor_points, -1, config)
    points = np.where(forward_valid[..., None], forward, backward)
    observed = forward_valid | backward_valid
    merged = np.where(forward_valid, forward_merged, backward_merged)
    points[anchor] = anchor_points; observed[anchor] = True
    merged[anchor] = False
    usable = observed & ~merged
    world = _world(config)
    dlt_rmse = []
    for frame in range(config.views):
        valid = usable[frame]
        if valid.sum() < 8:
            dlt_rmse.append(None); continue
        projection, _ = normalized_dlt(world[valid], points[frame, valid])
        error = project_points(world[valid], projection) - points[frame, valid]
        dlt_rmse.append(float(np.sqrt(np.mean(error * error))))
    finite = np.array([x for x in dlt_rmse if x is not None])
    if not len(finite) or finite.max() > config.max_dlt_rmse_px:
        raise RuntimeError(f"tracking DLT validation failed; max={finite.max() if len(finite) else None}")
    report = {
        "method": "marked_spiral_image_only_tracker",
        "raw_path": str(config.raw_path),
        "uses_system_geometry": False,
        "uses_scan_angles": False,
        "uses_phantom_pose": False,
        "anchor_frame": anchor,
        "anchor_initialization": anchor_info,
        "observed_count_per_frame": observed.sum(axis=1).tolist(),
        "observed_count_per_target": observed.sum(axis=0).tolist(),
        "merged_count_per_frame": merged.sum(axis=1).tolist(),
        "usable_count_per_frame": usable.sum(axis=1).tolist(),
        "dlt_validation": {"valid_frame_count": len(finite),
                           "rmse_mean_px": float(finite.mean()),
                           "rmse_max_px": float(finite.max()),
                           "rmse_per_frame_px": dlt_rmse},
        "forward_diagnostics": forward_diag,
        "backward_diagnostics": backward_diag,
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    np.save(config.output_directory / "points.npy", points)
    np.save(config.output_directory / "observed.npy", observed)
    np.save(config.output_directory / "merged.npy", merged)
    np.save(config.output_directory / "usable.npy", usable)
    (config.output_directory / "tracking.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_diagnostics(config.output_directory / "tracking.png", raw, points,
                      anchor, dlt_rmse)
    return TrackingResult(points, observed, merged, usable, report)
