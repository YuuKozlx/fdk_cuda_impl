"""Preprocessing layer for the Cho two-ring phantom.

This module converts a projection RAW stack into indexed sub-pixel bead
centres.  It deliberately contains no camera calibration or geometry fitting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from ..math.dlt import normalized_dlt, project_points
from .dlt import ChoConfig, phantom_points


def _components(image: np.ndarray, threshold: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract weighted connected components without assigning identities."""
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
    return components


def detect_frame(image: np.ndarray, threshold: float = 5.0,
                 split_merged: bool = True,
                 predicted: np.ndarray | None = None,
                 expected_beads: int = 24,
                 components: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    if components is None:
        components = _components(image, threshold)
    expected = int(expected_beads)
    if expected < 12 or expected % 2:
        raise ValueError("expected_beads must be an even number >= 12")
    if predicted is not None and len(components) in (expected - 1, expected):
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != (expected, 2):
            raise ValueError(f"predicted centers must have shape ({expected},2)")
        component_centers = np.asarray([item[3] for item in components])
        if len(components) == expected - 1:
            best = None
            areas = np.asarray([len(item[0]) for item in components])
            typical = float(np.median(areas))
            for duplicate in range(expected - 1):
                virtual = np.vstack([component_centers, component_centers[duplicate]])
                cost = np.sum((predicted[:, None, :] - virtual[None, :, :]) ** 2, axis=2)
                rows, cols = linear_sum_assignment(cost)
                score = float(cost[rows, cols].sum())
                score += 4.0 * max(0.0, typical * 1.25 - areas[duplicate]) ** 2
                if best is None or score < best[0]:
                    best = (score, duplicate, cols)
            _, duplicate, assignment = best
            virtual_to_component = np.arange(expected)
            virtual_to_component[-1] = duplicate
        else:
            cost = np.sum((predicted[:, None, :] - component_centers[None, :, :]) ** 2, axis=2)
            _, assignment = linear_sum_assignment(cost)
            duplicate = -1
            virtual_to_component = np.arange(expected)
        output = np.empty((expected, 2), dtype=float)
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
    if predicted is not None:
        raise ValueError(
            f"expected {expected - 1} or {expected} ring components after "
            f"marker removal, found {len(components)}")
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
    if len(centers) != expected:
        raise ValueError(f"expected {expected} beads, found {len(centers)}")
    return np.asarray(centers)


def initialize_marker_indices(components, config: ChoConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Identify an external marker and index the remaining ring beads.

    The marker is *not* assumed to be the largest projected component.  Each
    component is tentatively removed and the ordinary Cho correspondence
    search scores the remaining two-ring set by DLT reprojection error.
    """
    expected = 2 * int(config.beads_per_ring) + 1
    if len(components) != expected:
        raise ValueError(
            f"expected {expected} targets ({expected - 1} ring beads + marker), "
            f"found {len(components)}")
    centres = np.asarray([item[3] for item in components])
    best = None
    for marker in range(len(components)):
        try:
            indexed = initialize_indices(np.delete(centres, marker, axis=0), config)
        except ValueError:
            continue
        world = phantom_points(config)
        p, _ = normalized_dlt(world, indexed)
        score = float(np.sqrt(np.mean((project_points(world, p) - indexed) ** 2)))
        if best is None or score < best[0]:
            best = (score, marker, indexed)
    if best is None:
        raise ValueError("could not identify the external marker")
    score, marker, indexed = best
    indexed, line_score = _canonicalize_with_marker(
        indexed, centres[marker], config.beads_per_ring)
    return indexed, centres[marker], float(max(score, line_score))


def ring_order(points: np.ndarray, beads_per_ring: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(beads_per_ring)
    if np.asarray(points).shape != (2 * n, 2):
        raise ValueError(f"points must have shape ({2 * n},2)")
    order = np.argsort(points[:, 1])
    first, second = points[order[:n]], points[order[n:]]
    out = []
    for ring in (first, second):
        centre = ring.mean(axis=0)
        angles = np.arctan2(ring[:, 1] - centre[1], ring[:, 0] - centre[0])
        out.append(ring[np.argsort(angles)])
    return out[0], out[1]


def reorder_candidates(ring: np.ndarray, reverse: bool, shift: int) -> np.ndarray:
    sequence = ring[::-1] if reverse else ring
    return np.roll(sequence, shift, axis=0)


def _canonicalize_with_marker(indexed: np.ndarray, marker: np.ndarray,
                              beads_per_ring: int) -> tuple[np.ndarray, float]:
    """Rotate ring indices so bead zero shares the marker generatrix."""
    n = int(beads_per_ring)
    rings = [np.asarray(indexed[:n]), np.asarray(indexed[n:])]
    best = None
    for i, a in enumerate(rings[0]):
        for j, b in enumerate(rings[1]):
            line = b - a
            norm = float(np.linalg.norm(line))
            if norm < 1.0:
                continue
            perp = abs(line[0] * (marker - a)[1] - line[1] * (marker - a)[0]) / norm
            t = float(np.dot(marker - a, line) / np.dot(line, line))
            penalty = 0.0 if t < 0.0 or t > 1.0 else norm * min(t, 1.0 - t)
            score = perp + penalty
            if best is None or score < best[0]:
                best = (score, i, j, t)
    if best is None:
        raise ValueError("marker does not define a common generatrix")
    score, i, j, t = best
    first, second = np.roll(rings[0], -i, axis=0), np.roll(rings[1], -j, axis=0)
    if t < 0.0:
        first, second = second, first
    return np.vstack([first, second]), float(score)


def initialize_indices(points: np.ndarray, config: ChoConfig) -> np.ndarray:
    n = int(config.beads_per_ring)
    r0, r1 = ring_order(points, n)
    world = phantom_points(config)
    best = None
    for swap_rings in (False, True):
        rings = (r1, r0) if swap_rings else (r0, r1)
        for rev0 in (False, True):
            for rev1 in (False, True):
                for shift0 in range(n):
                    for shift1 in range(n):
                        image = np.vstack([reorder_candidates(rings[0], rev0, shift0),
                                            reorder_candidates(rings[1], rev1, shift1)])
                        p, _ = normalized_dlt(world, image)
                        predicted = project_points(world, p)
                        score = float(np.sqrt(np.mean((predicted - image) ** 2)))
                        if best is None or score < best[0]:
                            best = (score, image)
    if best is None or best[0] > 5.0:
        raise ValueError(f"could not establish bead indices; best RMSE={best[0] if best else None}")
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

    The point array has shape ``(valid_views, 2*N, 2)`` and is the interface
    consumed by the geometric calibration layer.
    """
    raw = np.memmap(raw_path, dtype=np.float32, mode="r",
                    shape=(views, image_shape[0], image_shape[1]))
    expected = 2 * int(config.beads_per_ring)
    components0 = _components(np.asarray(raw[0]), threshold)
    indexed0, marker0, marker_score = initialize_marker_indices(components0, config)
    stack = [indexed0]
    frame_ids = [0]
    rejected = []
    diagnostics = [{**frame_component_stats(np.asarray(raw[0]), threshold),
                    "frame": 0, "split_used": False,
                    "prediction_max_error_px": 0.0, "assigned_count": expected,
                    "marker_center_px": marker0.tolist(),
                    "marker_selection_rmse_px": marker_score,
                    "rejected": False}]
    marker_centres = [marker0]
    for i in range(1, views):
        if len(stack) >= 2:
            gap = max(frame_ids[-1] - frame_ids[-2], 1)
            predicted = stack[-1] + (stack[-1] - stack[-2]) / gap * (i - frame_ids[-1])
        else:
            predicted = stack[-1]
        image = np.asarray(raw[i])
        components = _components(image, threshold)
        marker_prediction = marker_centres[-1]
        if len(marker_centres) >= 2:
            marker_prediction = marker_centres[-1] + (marker_centres[-1] - marker_centres[-2])
        # Remove the external marker only when its predicted track is visible.
        # If it is out of frame, the remaining N/N-1 components are still a
        # valid ring-only observation and the normal Cho splitter handles them.
        marker_component = None
        if components:
            distances = np.asarray([np.linalg.norm(item[3] - marker_prediction)
                                    for item in components])
            candidate = int(np.argmin(distances))
            if distances[candidate] <= 45.0:
                marker_component = components[candidate]
                components = [item for j, item in enumerate(components) if j != candidate]
        stats = frame_component_stats(image, threshold)
        try:
            assigned = detect_frame(image, threshold, predicted=predicted,
                                    expected_beads=expected,
                                    components=components)
        except ValueError as exc:
            item = {"frame": i, "reason": str(exc)}
            rejected.append(item)
            diagnostics.append({**stats, "frame": i, "split_used": False,
                                "assigned_count": 0, "rejected": True,
                                "reason": str(exc)})
            continue
        err = np.linalg.norm(assigned - predicted, axis=1)
        diagnostics.append({**stats, "frame": i,
                            "split_used": stats["raw_component_count"] == expected - 1,
                            "prediction_max_error_px": float(err.max()),
                            "prediction_rmse_px": float(np.sqrt(np.mean(err ** 2))),
                            "assigned_count": expected, "rejected": False})
        if marker_component is not None:
            marker_centres.append(marker_component[3])
        else:
            marker_centres.append(marker_prediction)
        diagnostics[-1]["marker_center_px"] = marker_centres[-1].tolist()
        diagnostics[-1]["marker_detected"] = marker_component is not None
        stack.append(assigned)
        frame_ids.append(i)
    return np.asarray(stack), frame_ids, diagnostics, rejected
