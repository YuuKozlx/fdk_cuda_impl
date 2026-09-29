"""Image preprocessing for the standard Yang 6+6 target phantom.

The geometric solver is deliberately not used here.  This module only turns
a RAW projection stack into twelve consistently indexed sub-pixel image
points.  A cylinder replaces bead 1 on each ring and supplies the phase mark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment


@dataclass
class Component:
    pixels: np.ndarray
    weights: np.ndarray
    centroid: np.ndarray
    area: int


def detect_components(image: np.ndarray, threshold: float) -> list[Component]:
    """Return high-contrast target components with intensity-weighted centres."""
    array = np.asarray(image)
    labels, count = ndimage.label(
        array > float(threshold), structure=np.ones((3, 3), dtype=int))
    components: list[Component] = []
    for label in range(1, count + 1):
        yy, xx = np.nonzero(labels == label)
        if len(xx) < 3:
            continue
        pixels = np.column_stack([xx, yy]).astype(float)
        weights = np.maximum(array[yy, xx].astype(float) - threshold, 1.0e-6)
        centroid = np.sum(pixels * weights[:, None], axis=0) / np.sum(weights)
        components.append(Component(pixels, weights, centroid, len(xx)))
    return components


def _split_rings(components: list[Component]) -> tuple[list[Component], list[Component]]:
    if len(components) != 12:
        raise ValueError(f"initial view needs 12 separated targets, found {len(components)}")
    ordered = sorted(components, key=lambda item: item.centroid[1])
    return ordered[:6], ordered[6:]


def _marker_first(ring: list[Component]) -> np.ndarray:
    centres = np.asarray([item.centroid for item in ring])
    ring_centre = centres.mean(axis=0)
    angles = np.arctan2(centres[:, 1] - ring_centre[1],
                        centres[:, 0] - ring_centre[0])
    order = np.argsort(angles)
    marker = int(np.argmax([item.area for item in ring]))
    marker_position = int(np.flatnonzero(order == marker)[0])
    return np.roll(centres[order], -marker_position, axis=0)


def initial_index_candidates(components: list[Component]) -> list[np.ndarray]:
    """Enumerate the remaining ring/winding ambiguities after marker detection."""
    ring_a, ring_b = _split_rings(components)
    a = _marker_first(ring_a)
    b = _marker_first(ring_b)
    reverse = np.array([0, 5, 4, 3, 2, 1])
    candidates = []
    for swap in (False, True):
        first, second = (a, b) if not swap else (b, a)
        for reverse_first in (False, True):
            for reverse_second in (False, True):
                x = first[reverse] if reverse_first else first
                y = second[reverse] if reverse_second else second
                candidates.append(np.vstack([x, y]))
    return candidates


def _assign_components(components: list[Component], predicted: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Assign 12 target identities and mark targets hidden in a merged blob."""
    count = len(components)
    if count not in (11, 12):
        raise ValueError(f"expected 11 or 12 target components, found {count}")
    centres = np.asarray([item.centroid for item in components])
    if count == 12:
        cost = np.sum((predicted[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        rows, columns = linear_sum_assignment(cost)
        output = np.empty((12, 2), dtype=float)
        output[rows] = centres[columns]
        valid = np.ones(12, dtype=bool)
        return output, valid, ()

    # Try each component as the duplicated (merged) observation.  Assignment
    # distance identifies the only target pair whose predicted tracks meet.
    best = None
    areas = np.asarray([item.area for item in components], dtype=float)
    typical_area = float(np.median(areas))
    for duplicate in range(11):
        virtual = np.vstack([centres, centres[duplicate]])
        cost = np.sum((predicted[:, None, :] - virtual[None, :, :]) ** 2, axis=2)
        rows, columns = linear_sum_assignment(cost)
        score = float(cost[rows, columns].sum())
        # A merged component should be larger than an ordinary spherical bead.
        score += max(0.0, 1.35 * typical_area - areas[duplicate]) ** 2
        if best is None or score < best[0]:
            best = (score, duplicate, rows, columns)
    assert best is not None
    _, duplicate, rows, columns = best
    virtual_to_component = np.arange(12)
    virtual_to_component[11] = duplicate
    assigned_component = virtual_to_component[columns]
    output = np.asarray(predicted, dtype=float).copy()
    valid = np.ones(12, dtype=bool)
    merged_ids = tuple(int(x) for x in rows[assigned_component == duplicate])
    if len(merged_ids) != 2:
        raise ValueError("could not identify the two targets inside the merged component")
    for target, component in zip(rows, assigned_component):
        if target not in merged_ids:
            output[target] = centres[component]
    valid[list(merged_ids)] = False
    return output, valid, merged_ids


def _fill_periodic_gaps(points: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Fill only occluded samples using periodic cubic target trajectories."""
    result = points.copy()
    views = len(points)
    for target in range(12):
        valid_frames = np.flatnonzero(observed[:, target])
        missing_frames = np.flatnonzero(~observed[:, target])
        if len(missing_frames) == 0:
            continue
        if len(valid_frames) < 8:
            raise ValueError(f"target {target} has too few visible views")
        extended_frames = np.concatenate([
            valid_frames - views, valid_frames, valid_frames + views])
        for coordinate in range(2):
            values = points[valid_frames, target, coordinate]
            extended_values = np.tile(values, 3)
            spline = CubicSpline(extended_frames, extended_values)
            result[missing_frames, target, coordinate] = spline(missing_frames)
    return result


def track_standard_stack(raw: np.ndarray, threshold: float,
                         initial_points: np.ndarray
                         ) -> tuple[np.ndarray, list[dict]]:
    """Track twelve indexed targets and interpolate complete-overlap intervals."""
    views = int(raw.shape[0])
    points = np.empty((views, 12, 2), dtype=float)
    observed = np.ones((views, 12), dtype=bool)
    points[0] = np.asarray(initial_points, dtype=float)
    diagnostics = [{
        "frame": 0, "component_count": 12, "merged_target_ids": [],
        "interpolated": False,
    }]
    for frame in range(1, views):
        if frame >= 2:
            predicted = points[frame - 1] + (points[frame - 1] - points[frame - 2])
        else:
            predicted = points[frame - 1]
        components = detect_components(np.asarray(raw[frame]), threshold)
        assigned, valid, merged_ids = _assign_components(components, predicted)
        error = np.linalg.norm(assigned[valid] - predicted[valid], axis=1)
        if len(error) and float(error.max()) > 25.0:
            raise ValueError(
                f"frame {frame}: target assignment jumped {float(error.max()):.3f} px")
        points[frame] = assigned
        observed[frame] = valid
        diagnostics.append({
            "frame": frame,
            "component_count": len(components),
            "merged_target_ids": list(merged_ids),
            "interpolated": bool(merged_ids),
            "prediction_max_error_px": float(error.max()) if len(error) else 0.0,
        })
    return _fill_periodic_gaps(points, observed), diagnostics

