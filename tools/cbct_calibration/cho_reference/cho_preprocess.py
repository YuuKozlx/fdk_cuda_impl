"""Preprocessing layer for the Cho two-ring phantom.

This module converts a projection RAW stack into indexed sub-pixel bead
centres.  It deliberately contains no camera calibration or geometry fitting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from cho_calibration import ChoConfig, normalized_dlt, phantom_points, project_points


def detect_frame(image: np.ndarray, threshold: float = 5.0,
                 split_merged: bool = True,
                 predicted: np.ndarray | None = None) -> np.ndarray:
    mask = image > threshold
    labels, count = ndimage.label(mask)
    components = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 5:
            continue
        weights = np.maximum(image[ys, xs].astype(float) - threshold, 1.0e-6)
        centroid = np.array([np.sum(xs * weights) / np.sum(weights),
                             np.sum(ys * weights) / np.sum(weights)])
        components.append((xs, ys, weights, centroid))
    if predicted is not None and len(components) in (23, 24):
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != (24, 2):
            raise ValueError("predicted centers must have shape (24,2)")
        component_centers = np.asarray([item[3] for item in components])
        if len(components) == 23:
            best = None
            areas = np.asarray([len(item[0]) for item in components])
            typical = float(np.median(areas))
            for duplicate in range(23):
                virtual = np.vstack([component_centers, component_centers[duplicate]])
                cost = np.sum((predicted[:, None, :] - virtual[None, :, :]) ** 2, axis=2)
                rows, cols = linear_sum_assignment(cost)
                score = float(cost[rows, cols].sum())
                score += 4.0 * max(0.0, typical * 1.25 - areas[duplicate]) ** 2
                if best is None or score < best[0]:
                    best = (score, duplicate, cols)
            _, duplicate, assignment = best
            virtual_to_component = np.arange(24)
            virtual_to_component[23] = duplicate
        else:
            cost = np.sum((predicted[:, None, :] - component_centers[None, :, :]) ** 2, axis=2)
            _, assignment = linear_sum_assignment(cost)
            duplicate = -1
            virtual_to_component = np.arange(24)
        output = np.empty((24, 2), dtype=float)
        for component in range(len(components)):
            bead_ids = np.flatnonzero(virtual_to_component[assignment] == component)
            xs, ys, weights, centroid = components[component]
            if len(bead_ids) == 1:
                output[bead_ids[0]] = centroid
            elif len(bead_ids) == 2 and component == duplicate:
                seeds = predicted[bead_ids]
                pixels = np.column_stack([xs, ys])
                owner = np.argmin(np.sum((pixels[:, None, :] - seeds[None, :, :]) ** 2, axis=2), axis=1)
                for branch, bead_id in enumerate(bead_ids):
                    selected = owner == branch
                    if selected.sum() < 2:
                        output[bead_id] = seeds[branch]
                    else:
                        w = weights[selected]
                        output[bead_id] = np.sum(pixels[selected] * w[:, None], axis=0) / np.sum(w)
            else:
                raise ValueError("trajectory-guided component assignment is ambiguous")
        return output
    centers = []
    for xs, ys, weights, centroid in components:
        if split_merged and len(xs) >= 25:
            local = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            peak_mask = ((local == ndimage.maximum_filter(local, size=3)) &
                         (local > max(threshold + 2.0, 8.0)))
            peak_labels, peak_count = ndimage.label(peak_mask)
            peaks = []
            for peak in range(1, peak_count + 1):
                py, px = np.where(peak_labels == peak)
                if len(px):
                    k = np.argmax(local[py, px])
                    peaks.append(np.array([px[k] + xs.min(), py[k] + ys.min()], dtype=float))
            if len(peaks) == 2:
                peaks = np.asarray(peaks)
                distance = np.sum((np.column_stack([xs, ys])[:, None, :] - peaks[None, :, :]) ** 2, axis=2)
                for branch in range(2):
                    selected = np.argmin(distance, axis=1) == branch
                    if selected.sum() < 3:
                        break
                    w = weights[selected]
                    xx, yy = xs[selected], ys[selected]
                    centers.append([np.sum(xx * w) / np.sum(w), np.sum(yy * w) / np.sum(w)])
                else:
                    continue
        centers.append(centroid)
    if len(centers) != 24:
        raise ValueError(f"expected 24 beads, found {len(centers)}")
    return np.asarray(centers)


def ring_order(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(points[:, 1])
    first, second = points[order[:12]], points[order[12:]]
    out = []
    for ring in (first, second):
        centre = ring.mean(axis=0)
        angles = np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0])
        out.append(ring[np.argsort(angles)])
    return out[0], out[1]


def reorder_candidates(ring: np.ndarray, reverse: bool, shift: int) -> np.ndarray:
    sequence = ring[::-1] if reverse else ring
    return np.roll(sequence, shift, axis=0)


def initialize_indices(points: np.ndarray, config: ChoConfig) -> np.ndarray:
    r0, r1 = ring_order(points)
    world = phantom_points(config)
    best = None
    for swap_rings in (False, True):
        rings = (r1, r0) if swap_rings else (r0, r1)
        for rev0 in (False, True):
            for rev1 in (False, True):
                for shift0 in range(12):
                    for shift1 in range(12):
                        image = np.vstack([reorder_candidates(rings[0], rev0, shift0),
                                            reorder_candidates(rings[1], rev1, shift1)])
                        p, _ = normalized_dlt(world, image)
                        predicted = project_points(world, p)
                        score = float(np.sqrt(np.mean((predicted - image) ** 2)))
                        if best is None or score < best[0]:
                            best = (score, image)
    if best is None or best[0] > 5.0:
        raise ValueError(f"could not establish bead indices; best RMSE={best[0] if best else None}")
    print(f"initial correspondence DLT RMSE={best[0]:.6g} pixel")
    return best[1]


def frame_component_stats(image: np.ndarray, threshold: float) -> dict:
    labels, _ = ndimage.label(image > threshold)
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas >= 5]
    return {
        "raw_component_count": int(len(areas)),
        "component_area_median": float(np.median(areas)) if len(areas) else 0.0,
        "component_area_max": int(areas.max()) if len(areas) else 0,
    }


def track_raw_stack(raw_path: Path, config: ChoConfig, threshold: float = 5.0,
                    views: int = 360,
                    image_shape: tuple[int, int] = (1024, 1024)
                    ) -> tuple[np.ndarray, list[int], list[dict], list[dict]]:
    """Return indexed observations and preprocessing diagnostics.

    The point array has shape ``(valid_views, 24, 2)`` and is the interface
    consumed by the geometric calibration layer.
    """
    raw = np.memmap(raw_path, dtype=np.float32, mode="r",
                    shape=(views, image_shape[0], image_shape[1]))
    frame0 = detect_frame(np.asarray(raw[0]), threshold, split_merged=False)
    indexed0 = initialize_indices(frame0, config)
    stack = [indexed0]
    frame_ids = [0]
    rejected = []
    diagnostics = [{**frame_component_stats(np.asarray(raw[0]), threshold),
                    "frame": 0, "split_used": False,
                    "prediction_max_error_px": 0.0, "assigned_count": 24,
                    "rejected": False}]
    for i in range(1, views):
        if len(stack) >= 2:
            gap = max(frame_ids[-1] - frame_ids[-2], 1)
            predicted = stack[-1] + (stack[-1] - stack[-2]) / gap * (i - frame_ids[-1])
        else:
            predicted = stack[-1]
        image = np.asarray(raw[i])
        stats = frame_component_stats(image, threshold)
        try:
            assigned = detect_frame(image, threshold, predicted=predicted)
        except ValueError as exc:
            item = {"frame": i, "reason": str(exc)}
            rejected.append(item)
            diagnostics.append({**stats, "frame": i, "split_used": False,
                                "assigned_count": 0, "rejected": True,
                                "reason": str(exc)})
            continue
        err = np.linalg.norm(assigned - predicted, axis=1)
        diagnostics.append({**stats, "frame": i,
                            "split_used": stats["raw_component_count"] == 23,
                            "prediction_max_error_px": float(err.max()),
                            "prediction_rmse_px": float(np.sqrt(np.mean(err ** 2))),
                            "assigned_count": 24, "rejected": False})
        stack.append(assigned)
        frame_ids.append(i)
    return np.asarray(stack), frame_ids, diagnostics, rejected
