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


def external_marker_index_candidates(components: list[Component], marker_count: int = 1,
                                     ring_points_3d: np.ndarray | None = None,
                                     marker_points_3d: np.ndarray | None = None):
    """Return Yang ring candidates after removing an external marker.

    The marker, upper ring zero and lower ring zero lie on one phantom
    generatrix.  Their image points are therefore collinear; this projective
    invariant is used instead of projected area to locate the marker.
    """
    expected = 12 + int(marker_count)
    if marker_count not in (0, 1, 2):
        raise ValueError("marker_count must be 0, 1 or 2")
    if len(components) != expected:
        raise ValueError(f"initial view needs {expected} targets, found {len(components)}")
    centres = np.asarray([item.centroid for item in components])
    if marker_count == 0:
        remaining = centres
        order = np.argsort(remaining[:, 1])
        rings = []
        for ring in (remaining[order[:6]], remaining[order[6:]]):
            center = ring.mean(axis=0)
            angle = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
            rings.append(ring[np.argsort(angle)])
        candidates = []
        for swap in (False, True):
            first, second = rings if not swap else rings[::-1]
            for reverse_first in (False, True):
                for reverse_second in (False, True):
                    a = first[[0, 5, 4, 3, 2, 1]] if reverse_first else first
                    b = second[[0, 5, 4, 3, 2, 1]] if reverse_second else second
                    for shift_first in range(6):
                        for shift_second in range(6):
                            candidates.append(np.vstack([
                                np.roll(a, shift_first, axis=0),
                                np.roll(b, shift_second, axis=0)]))
        return candidates, [np.empty((0, 2)) for _ in candidates], {"marker_count": 0}
    if marker_count == 2:
        if ring_points_3d is None or marker_points_3d is None:
            raise ValueError("two-marker initialization needs the known 3-D target points")
        from itertools import combinations, permutations
        from ..math.dlt import normalized_dlt, project_points
        ring_world = np.asarray(ring_points_3d, dtype=float)
        marker_world = np.asarray(marker_points_3d, dtype=float)
        if ring_world.shape != (12, 3) or marker_world.shape != (2, 3):
            raise ValueError("two-marker Yang model must contain 12 ring and 2 marker points")

        # Area ranks candidates only; the full 14-point DLT residual assigns
        # marker identity, ring phase, winding and upper/lower correspondence.
        marker_sets = list(combinations(range(len(components)), 2))
        marker_sets.sort(key=lambda ids: -sum(components[i].area for i in ids))
        marker_sets = marker_sets[:16]
        scored = []
        reverse = np.array([0, 5, 4, 3, 2, 1])
        for marker_ids in marker_sets:
            ring_ids = [i for i in range(len(components)) if i not in marker_ids]
            remaining = centres[ring_ids]
            order = np.argsort(remaining[:, 1])
            base_rings = []
            for ring in (remaining[order[:6]], remaining[order[6:]]):
                center = ring.mean(axis=0)
                angle = np.arctan2(ring[:, 1] - center[1], ring[:, 0] - center[0])
                base_rings.append(ring[np.argsort(angle)])
            for swap in (False, True):
                first, second = (base_rings if not swap else base_rings[::-1])
                for reverse_first in (False, True):
                    for reverse_second in (False, True):
                        a = first[reverse] if reverse_first else first
                        b = second[reverse] if reverse_second else second
                        for shift_first in range(6):
                            for shift_second in range(6):
                                rings = np.vstack([np.roll(a, shift_first, axis=0),
                                                   np.roll(b, shift_second, axis=0)])
                                for marker_order in permutations(range(2)):
                                    markers = centres[list(marker_ids)][list(marker_order)]
                                    world = np.vstack([ring_world, marker_world])
                                    image = np.vstack([rings, markers])
                                    projection, _ = normalized_dlt(world, image)
                                    rmse = float(np.sqrt(np.mean(
                                        (project_points(world, projection) - image) ** 2)))
                                    scored.append((rmse, rings, markers, marker_ids))
        scored.sort(key=lambda item: item[0])
        # Keep only distinct low-residual maps for the physical/PIC scorer.
        selected = scored[:16]
        return ([item[1] for item in selected],
                [item[2] for item in selected], {
                    "marker_count": 2,
                    "candidate_count_before_dlt": len(scored),
                    "candidate_dlt_rmse_px": [float(item[0]) for item in selected],
                    "marker_component_candidates": [list(map(int, item[3]))
                                                      for item in selected],
                })
    best = None
    for marker_id, marker in enumerate(centres):
        remaining = np.delete(centres, marker_id, axis=0)
        order = np.argsort(remaining[:, 1])
        rings = (remaining[order[:6]], remaining[order[6:]])
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
                    best = (score, marker_id, rings, i, j, t, perp)
    if best is None:
        raise ValueError("could not identify marker generatrix")
    score, marker_id, rings, i0, i1, t, perp = best
    if t < 0.0:
        lower, upper, lower_zero, upper_zero = rings[1], rings[0], i1, i0
    else:
        lower, upper, lower_zero, upper_zero = rings[0], rings[1], i0, i1

    def zero_first(ring, zero):
        c = ring.mean(axis=0)
        angles = np.arctan2(ring[:, 1] - c[1], ring[:, 0] - c[0])
        order = np.argsort(angles)
        return np.roll(ring[order], -int(np.flatnonzero(order == zero)[0]), axis=0)

    lower, upper = zero_first(lower, lower_zero), zero_first(upper, upper_zero)
    reverse = np.array([0, 5, 4, 3, 2, 1])
    candidates = [np.vstack([lower[reverse] if rl else lower,
                              upper[reverse] if ru else upper])
                  for rl in (False, True) for ru in (False, True)]
    marker = centres[marker_id][None, :]
    return candidates, [marker.copy() for _ in candidates], {
        "marker_component": int(marker_id),
        "marker_center_px": centres[marker_id].tolist(),
        "marker_area_px": int(components[marker_id].area),
        "generatrix_distance_px": float(perp),
        "generatrix_parameter": float(t),
        "selection_score_px": float(score),
    }


def _assign_components(components: list[Component], predicted: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Assign 12 target identities and mark targets hidden in a merged blob."""
    count = len(components)
    if count >= 12:
        centres = np.asarray([item.centroid for item in components])
        cost = np.sum((predicted[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        rows, columns = linear_sum_assignment(cost)
        if len(rows) != 12:
            raise ValueError("could not assign twelve ring targets")
        output = np.full((12, 2), np.nan, dtype=float)
        output[rows] = centres[columns]
        return output, np.ones(12, dtype=bool), ()
    if count != 11:
        raise ValueError(f"expected at least 11 target components, found {count}")
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


def _assign_pixels(image: np.ndarray, predicted: np.ndarray, threshold: float,
                   gate_radius_px: float = 24.0):
    """Split touching targets using predicted tracks as pixel-level seeds."""
    yy, xx = np.nonzero(np.asarray(image) > threshold)
    pixels = np.column_stack([xx, yy]).astype(float)
    values = np.maximum(np.asarray(image)[yy, xx].astype(float) - threshold, 1.0e-6)
    delta = pixels[:, None, :] - predicted[None, :, :]
    squared = np.sum(delta * delta, axis=2)
    owner = np.argmin(squared, axis=1)
    accepted = np.min(squared, axis=1) <= gate_radius_px ** 2
    output = np.asarray(predicted, dtype=float).copy()
    valid = np.zeros(12, dtype=bool)
    for target in range(12):
        selected = accepted & (owner == target)
        if np.count_nonzero(selected) < 3:
            continue
        weight = values[selected]
        output[target] = np.sum(pixels[selected] * weight[:, None], axis=0) / np.sum(weight)
        valid[target] = True
    return output, valid


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


def _track_ordered_stack(raw: np.ndarray, threshold: float,
                         initial_points: np.ndarray,
                         initial_marker: np.ndarray | None = None,
                         initial_markers: np.ndarray | None = None
                         ) -> tuple[np.ndarray, list[dict]]:
    """Track twelve indexed targets and interpolate complete-overlap intervals."""
    views = int(raw.shape[0])
    points = np.empty((views, 12, 2), dtype=float)
    observed = np.ones((views, 12), dtype=bool)
    points[0] = np.asarray(initial_points, dtype=float)
    if initial_markers is None and initial_marker is not None:
        initial_markers = np.asarray(initial_marker, dtype=float).reshape(1, 2)
    marker_track = [] if initial_markers is None else [np.asarray(initial_markers, dtype=float)]
    diagnostics = [{
        "frame": 0, "component_count": 12 + (0 if initial_markers is None else len(initial_markers)),
        "marker_center_px": None if initial_markers is None else marker_track[0].tolist(),
        "marker_detected": initial_markers is not None, "merged_target_ids": [],
        "interpolated": False,
    }]
    for frame in range(1, views):
        if frame >= 2:
            predicted = points[frame - 1] + (points[frame - 1] - points[frame - 2])
        else:
            predicted = points[frame - 1]
        components = detect_components(np.asarray(raw[frame]), threshold)
        marker = None
        if initial_markers is not None:
            marker_prediction = marker_track[-1]
            if len(marker_track) >= 2:
                marker_prediction = marker_track[-1] + (marker_track[-1] - marker_track[-2])
            marker = marker_prediction.copy()
            selected = []
            if len(components) > 12:
                cost = np.sum((marker_prediction[:, None, :] -
                               np.asarray([item.centroid for item in components])[None, :, :]) ** 2, axis=2)
                rows, marker_ids = linear_sum_assignment(cost)
                visible_count = min(len(marker_prediction), len(components) - 12)
                pairs = sorted((float(np.sqrt(cost[r, c])), int(r), int(c))
                               for r, c in zip(rows, marker_ids))
                selected = [(r, c) for distance, r, c in pairs[:visible_count]
                            if distance <= 45.0]
                for r, c in selected:
                    marker[r] = components[c].centroid
                components = [item for i, item in enumerate(components)
                              if i not in {c for _, c in selected}]
            marker_track.append(marker)
            if not selected:
                marker = None
        assigned, valid, merged_ids = _assign_components(components, predicted)
        error = np.linalg.norm(assigned[valid] - predicted[valid], axis=1)
        if len(error) and float(error.max()) > 25.0:
            assigned, valid = _assign_pixels(np.asarray(raw[frame]), predicted, threshold)
            merged_ids = tuple(int(x) for x in np.flatnonzero(~valid))
            error = np.linalg.norm(assigned[valid] - predicted[valid], axis=1)
            if not np.all(valid) or (len(error) and float(error.max()) > 25.0):
                raise ValueError(
                    f"frame {frame}: target assignment jumped "
                    f"{float(error.max()) if len(error) else float('nan'):.3f} px")
        points[frame] = assigned
        observed[frame] = valid
        diagnostics.append({
            "frame": frame,
            "component_count": len(components) + int(marker is not None),
            "marker_center_px": None if marker is None else marker.tolist(),
            "marker_detected": marker is not None,
            "merged_target_ids": list(merged_ids),
            "interpolated": bool(merged_ids),
            "prediction_max_error_px": float(error.max()) if len(error) else 0.0,
        })
    return _fill_periodic_gaps(points, observed), diagnostics


class _FrameOrder:
    """Read-only frame remapping without copying a multi-gigabyte RAW stack."""
    def __init__(self, raw, indices):
        self.raw = raw
        self.indices = np.asarray(indices, dtype=int)
        self.shape = (len(self.indices), *raw.shape[1:])

    def __getitem__(self, item):
        return self.raw[int(self.indices[item])]


def track_standard_stack(raw: np.ndarray, threshold: float,
                         initial_points: np.ndarray,
                         initial_marker: np.ndarray | None = None,
                         initial_markers: np.ndarray | None = None
                         ) -> tuple[np.ndarray, list[dict]]:
    """Track from frame zero in both directions and merge two half-turns."""
    views = int(raw.shape[0])
    split = views // 2
    forward_ids = np.arange(0, split + 1)
    backward_ids = np.r_[0, np.arange(views - 1, split, -1)]
    forward, forward_diag = _track_ordered_stack(
        _FrameOrder(raw, forward_ids), threshold, initial_points,
        initial_marker, initial_markers)
    backward, backward_diag = _track_ordered_stack(
        _FrameOrder(raw, backward_ids), threshold, initial_points,
        initial_marker, initial_markers)
    points = np.empty((views, 12, 2), dtype=float)
    points[forward_ids] = forward
    points[backward_ids] = backward
    diagnostics = []
    for item in forward_diag:
        converted = dict(item)
        converted["frame"] = int(forward_ids[int(item["frame"])])
        converted["tracking_direction"] = 1
        diagnostics.append(converted)
    for item in backward_diag[1:]:
        converted = dict(item)
        converted["frame"] = int(backward_ids[int(item["frame"])])
        converted["tracking_direction"] = -1
        diagnostics.append(converted)
    diagnostics.sort(key=lambda item: item["frame"])
    return points, diagnostics
