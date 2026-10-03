"""Formal image-only tracker for the marked unequal double-ring phantom.

The tracker does not use SID, SDD, detector offsets, detector tilts, phantom
pose, or scan angle.  Those quantities are calibration outputs.  The only
geometric prior used here is the known phantom point layout, which is needed
to resolve the initial 2-D/3-D labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from ..math.dlt import normalized_dlt, project_points
from .dlt import DltConfig, calibrate_view
from .phantom import marked_points


@dataclass(frozen=True)
class FormalTrackerConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    rows: int = 1024
    cols: int = 1024
    threshold: float = 5.0
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    gate_radius_px: float = 28.0
    min_pixels: int = 4
    marker_area_ratio: float = 1.45
    max_dlt_rmse_px: float = 1.0


@dataclass
class TrackingResult:
    points_px: np.ndarray
    observed: np.ndarray
    report: dict


@dataclass
class Component:
    pixels: np.ndarray
    weights: np.ndarray
    centroid: np.ndarray
    area: int
    peak: float


def _load_raw(config: FormalTrackerConfig) -> np.memmap:
    expected = config.views * config.rows * config.cols * 4
    if not config.raw_path.exists():
        raise FileNotFoundError(config.raw_path)
    if config.raw_path.stat().st_size != expected:
        raise ValueError(
            f"RAW size mismatch: expected {expected} bytes, "
            f"got {config.raw_path.stat().st_size}")
    return np.memmap(config.raw_path, dtype="<f4", mode="r",
                     shape=(config.views, config.rows, config.cols))


def detect_components(image: np.ndarray, threshold: float) -> list[Component]:
    """Detect bright target components and intensity-weighted centroids."""
    image = np.asarray(image)
    mask = image > float(threshold)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    components: list[Component] = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) < 4:
            continue
        pixels = np.column_stack([xx, yy]).astype(float)
        values = np.maximum(image[yy, xx].astype(float) - threshold, 1.0e-6)
        centroid = np.sum(pixels * values[:, None], axis=0) / np.sum(values)
        components.append(Component(
            pixels=pixels,
            weights=values,
            centroid=centroid,
            area=len(xx),
            peak=float(np.max(image[yy, xx])),
        ))
    return components


def _ring_order(centres: np.ndarray) -> np.ndarray:
    centre = np.mean(centres, axis=0)
    angle = np.arctan2(centres[:, 1] - centre[1],
                       centres[:, 0] - centre[0])
    return centres[np.argsort(angle)]


def _initial_index_candidates(components: list[Component]) -> list[np.ndarray]:
    """Enumerate label ambiguities using only anchor-frame image geometry."""
    if len(components) != 25:
        raise ValueError(
            f"anchor frame must have 25 separated components, found {len(components)}")

    marker_id = int(np.argmax([item.area for item in components]))
    marker = components[marker_id].centroid
    ordinary = [item.centroid for i, item in enumerate(components)
                if i != marker_id]
    ordinary = np.asarray(ordinary, dtype=float)
    # At a usable anchor frame the two projected rings form two separated
    # bands.  The DLT search below still enumerates ring exchange and winding.
    vertical_order = np.argsort(ordinary[:, 1])
    ring_a = _ring_order(ordinary[vertical_order[:12]])
    ring_b = _ring_order(ordinary[vertical_order[12:]])
    candidates = []
    for swap in (False, True):
        base_a, base_b = (ring_a, ring_b) if not swap else (ring_b, ring_a)
        for reverse_a in (False, True):
            for reverse_b in (False, True):
                a = base_a[::-1] if reverse_a else base_a
                b = base_b[::-1] if reverse_b else base_b
                # The marker lies axially above upper-ring target zero.  Its
                # nearest member in the ring assigned as "upper" fixes that
                # ring's cyclic phase without scanner geometry.
                nearest_marker = int(np.argmin(np.linalg.norm(a - marker, axis=1)))
                aa = np.roll(a, -nearest_marker, axis=0)
                for shift_b in range(12):
                    bb = np.roll(b, shift_b, axis=0)
                    ordered = np.empty((25, 2), dtype=float)
                    ordered[:24:2] = aa
                    ordered[1:24:2] = bb
                    ordered[24] = marker
                    candidates.append(ordered)
    return candidates


def initialize_indices(image: np.ndarray, threshold: float = 5.0,
                       pixel_size_mm: tuple[float, float] = (0.417, 0.417),
                       max_rmse_px: float = 5.0) -> tuple[np.ndarray, dict]:
    """Choose anchor labels using a physically constrained flat-panel camera."""
    components = detect_components(image, threshold)
    world = marked_points()
    best = None
    candidate_count = 0
    camera_config = DltConfig(
        pixel_size_mm=pixel_size_mm,
        robust_loss="soft_l1",
        robust_scale_px=0.15,
        max_nfev=300,
    )
    for candidate in _initial_index_candidates(components):
        candidate_count += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pose = calibrate_view(candidate, camera_config, world)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        rmse = float(pose.reprojection_rmse_px)
        if best is None or rmse < best[0]:
            best = (rmse, candidate, pose)
    if best is None or best[0] > max_rmse_px:
        raise ValueError(
            f"could not establish anchor labels; best RMSE="
            f"{best[0] if best else None}")
    return best[1], {
        "component_count": len(components),
        "candidate_count": candidate_count,
        "selected_physical_camera_rmse_px": best[0],
        "marker_component": "largest separated component",
        "upper_ring_phase": "upper target zero is nearest the axial marker",
        "camera_constraints": "zero skew, known pixel pitch, one physical SDD",
        "chirality": (
            "single-view mirror ambiguity retained; rotation direction is "
            "selected by the multi-view joint calibration"),
    }


def choose_anchor(raw: np.ndarray, threshold: float,
                  marker_area_ratio: float) -> tuple[int, list[dict]]:
    """Select a frame with all 25 targets separated, preferring clear targets."""
    scores = []
    for frame in range(int(raw.shape[0])):
        components = detect_components(np.asarray(raw[frame]), threshold)
        if len(components) != 25:
            scores.append({"frame": frame, "component_count": len(components),
                           "anchor_eligible": False})
            continue
        areas = np.asarray([item.area for item in components], dtype=float)
        marker = float(np.max(areas))
        ordinary = np.delete(areas, int(np.argmax(areas)))
        ratio = marker / max(float(np.median(ordinary)), 1.0)
        score = float(np.median(ordinary)) + 25.0 * min(ratio, 4.0)
        scores.append({"frame": frame, "component_count": 25,
                       "marker_area_ratio": ratio, "anchor_score": score,
                       "anchor_eligible": ratio >= marker_area_ratio})
    eligible = [item for item in scores if item["anchor_eligible"]]
    if not eligible:
        raise ValueError("no frame contains 25 separated target components")
    anchor = max(eligible, key=lambda item: item["anchor_score"])
    return int(anchor["frame"]), scores


def _extract_from_predictions(image: np.ndarray, predicted: np.ndarray,
                              threshold: float, gate_radius_px: float,
                              min_pixels: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split ordinary and merged components by nearest predicted pixel seed."""
    image = np.asarray(image)
    yy, xx = np.nonzero(image > threshold)
    if len(xx) == 0:
        return (np.full_like(predicted, np.nan), np.zeros(len(predicted), bool),
                {"foreground_pixels": 0, "valid_count": 0})
    pixels = np.column_stack([xx, yy]).astype(float)
    values = np.maximum(image[yy, xx].astype(float) - threshold, 1.0e-6)
    owner = np.empty(len(pixels), dtype=np.int32)
    distance = np.empty(len(pixels), dtype=float)
    for start in range(0, len(pixels), 65536):
        stop = min(start + 65536, len(pixels))
        delta = pixels[start:stop, None, :] - predicted[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        owner[start:stop] = np.argmin(squared, axis=1)
        distance[start:stop] = np.sqrt(np.min(squared, axis=1))
    accepted = distance <= gate_radius_px
    output = np.full_like(predicted, np.nan, dtype=float)
    valid = np.zeros(len(predicted), dtype=bool)
    counts = np.zeros(len(predicted), dtype=int)
    for target in range(len(predicted)):
        selected = accepted & (owner == target)
        counts[target] = int(np.count_nonzero(selected))
        if counts[target] < min_pixels:
            continue
        weights = values[selected]
        output[target] = np.sum(pixels[selected] * weights[:, None], axis=0) / np.sum(weights)
        valid[target] = True
    return output, valid, {
        "foreground_pixels": int(len(pixels)),
        "valid_count": int(np.count_nonzero(valid)),
        "pixel_counts_min": int(counts.min()),
        "pixel_counts_median": float(np.median(counts)),
        "pixel_counts_max": int(counts.max()),
    }


def _track_direction(raw: np.ndarray, anchor_frame: int,
                    anchor_points: np.ndarray, direction: int,
                    config: FormalTrackerConfig) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    views, targets = raw.shape[0], anchor_points.shape[0]
    points = np.full((views, targets, 2), np.nan, dtype=float)
    observed = np.zeros((views, targets), dtype=bool)
    diagnostics = []
    points[anchor_frame] = anchor_points
    observed[anchor_frame] = True
    previous = anchor_points.copy()
    previous_previous = None
    active = np.ones(targets, dtype=bool)
    frame = anchor_frame + direction
    while 0 <= frame < views:
        if previous_previous is None:
            predicted = previous.copy()
        else:
            predicted = previous + (previous - previous_previous)
        predicted = np.asarray(predicted, dtype=float)
        valid_prediction = (
            active & np.isfinite(predicted).all(axis=1) &
            (predicted[:, 0] >= 0.0) & (predicted[:, 0] < config.cols) &
            (predicted[:, 1] >= 0.0) & (predicted[:, 1] < config.rows))
        measured, valid, stats = _extract_from_predictions(
            np.asarray(raw[frame]), predicted, config.threshold,
            config.gate_radius_px, config.min_pixels)
        valid &= valid_prediction
        # A target that leaves the panel or cannot be measured is permanently
        # inactive for this direction; it is never reassigned to another ID.
        active &= valid
        measured[~active] = np.nan
        points[frame] = measured
        observed[frame] = active
        if np.any(active):
            jump = np.linalg.norm(measured[active] - predicted[active], axis=1)
            jump_max = float(np.max(jump))
            jump_rms = float(np.sqrt(np.mean(jump * jump)))
        else:
            jump_max = float("nan")
            jump_rms = float("nan")
        diagnostics.append({
            "frame": frame,
            "direction": direction,
            "active_count": int(np.count_nonzero(active)),
            "prediction_max_jump_px": jump_max,
            "prediction_rmse_px": jump_rms,
            **stats,
        })
        previous_previous = previous
        previous = measured
        frame += direction
    return points, observed, diagnostics


def _save_diagnostics(path: Path, raw: np.ndarray, points: np.ndarray,
                      anchor_frame: int, anchor_scan: list[dict],
                      dlt_rmse: list[float | None]) -> None:
    difficult_frame = min(anchor_scan, key=lambda item: item["component_count"])["frame"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for axis, frame, title in (
            (axes[0, 0], anchor_frame,
             f"Anchor frame {anchor_frame}: automatic 3-D IDs"),
            (axes[0, 1], difficult_frame,
             f"Frame {difficult_frame}: merged targets retained")):
        axis.imshow(np.asarray(raw[frame]), cmap="gray", vmin=0, vmax=21)
        axis.scatter(points[frame, :-1, 0], points[frame, :-1, 1], s=45,
                     facecolors="none", edgecolors="cyan", linewidths=1.0)
        axis.scatter(points[frame, -1, 0], points[frame, -1, 1], s=80,
                     facecolors="none", edgecolors="red", linewidths=1.5,
                     label="marker ID 24")
        if frame == anchor_frame:
            for target, (u, v) in enumerate(points[frame]):
                axis.text(u + 4, v, str(target), color="yellow", fontsize=7)
        axis.set(xlim=(250, 820), ylim=(1024, 250), title=title)
        axis.legend(loc="upper right", fontsize=8)

    for target in range(points.shape[1] - 1):
        axes[1, 0].plot(points[:, target, 0], points[:, target, 1],
                        lw=0.7, alpha=0.7)
    axes[1, 0].plot(points[:, -1, 0], points[:, -1, 1], color="red",
                    lw=1.5, label="marker ID 24")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="Tracked detector trajectories",
                   xlabel="u / px", ylabel="image row / px")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    values = np.asarray([np.nan if value is None else value
                         for value in dlt_rmse], dtype=float)
    axes[1, 1].plot(values, lw=1.0)
    axes[1, 1].axhline(1.0, color="red", ls="--", lw=0.8,
                       label="1 px acceptance threshold")
    axes[1, 1].set(title="Independent per-frame DLT validation",
                   xlabel="view", ylabel="point RMSE / px")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def track_stack(config: FormalTrackerConfig) -> TrackingResult:
    """Run anchor discovery, label initialization, and bidirectional tracking."""
    raw = _load_raw(config)
    anchor_frame, anchor_scan = choose_anchor(
        raw, config.threshold, config.marker_area_ratio)
    anchor_points, anchor_info = initialize_indices(
        np.asarray(raw[anchor_frame]), config.threshold, config.pixel_size_mm)
    forward, forward_valid, forward_diag = _track_direction(
        raw, anchor_frame, anchor_points, +1, config)
    backward, backward_valid, backward_diag = _track_direction(
        raw, anchor_frame, anchor_points, -1, config)
    points = np.where(forward_valid[..., None], forward, backward)
    observed = forward_valid | backward_valid
    # Anchor is valid in both passes. Prefer forward values where both exist.
    points[anchor_frame] = anchor_points
    observed[anchor_frame] = True
    world = marked_points()
    dlt_rmse = []
    dlt_max_error = []
    for frame in range(config.views):
        valid = observed[frame]
        if np.count_nonzero(valid) < 8:
            dlt_rmse.append(None)
            dlt_max_error.append(None)
            continue
        projection, _ = normalized_dlt(world[valid], points[frame, valid])
        error = np.linalg.norm(
            project_points(world[valid], projection) - points[frame, valid],
            axis=1)
        dlt_rmse.append(float(np.sqrt(np.mean(error * error))))
        dlt_max_error.append(float(np.max(error)))
    finite_rmse = np.asarray([value for value in dlt_rmse if value is not None])
    if not len(finite_rmse):
        raise RuntimeError("tracking left no frame with enough points for DLT validation")
    if float(np.max(finite_rmse)) > config.max_dlt_rmse_px:
        frame = int(np.nanargmax(np.asarray([
            np.nan if value is None else value for value in dlt_rmse])))
        raise RuntimeError(
            f"tracking DLT validation failed at frame {frame}: "
            f"RMSE={dlt_rmse[frame]:.6g} px")
    report = {
        "method": "image_only_bidirectional_prediction_guided_tracker",
        "raw_path": str(config.raw_path),
        "uses_system_geometry": False,
        "uses_scan_angles": False,
        "uses_phantom_pose": False,
        "anchor_frame": anchor_frame,
        "anchor_selection": anchor_scan,
        "anchor_initialization": anchor_info,
        "views": int(config.views),
        "target_count": int(points.shape[1]),
        "observed_count_per_target": observed.sum(axis=0).tolist(),
        "observed_count_per_frame": observed.sum(axis=1).tolist(),
        "dlt_validation": {
            "valid_frame_count": int(len(finite_rmse)),
            "rmse_mean_px": float(np.mean(finite_rmse)),
            "rmse_median_px": float(np.median(finite_rmse)),
            "rmse_max_px": float(np.max(finite_rmse)),
            "rmse_per_frame_px": dlt_rmse,
            "max_point_error_per_frame_px": dlt_max_error,
        },
        "forward_diagnostics": forward_diag,
        "backward_diagnostics": backward_diag,
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    np.save(config.output_directory / "points.npy", points)
    np.save(config.output_directory / "observed.npy", observed)
    (config.output_directory / "tracking_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_diagnostics(config.output_directory / "tracking_diagnostics.png",
                      raw, points, anchor_frame, anchor_scan, dlt_rmse)
    return TrackingResult(points, observed, report)
