"""Steel-ball detection and visibility filtering for flat-panel projections.

The detector intentionally has no OpenCV dependency: it uses thresholded
8-connected components, a circularity gate, and intensity-weighted moments.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
from itertools import combinations, permutations
from pathlib import Path

import numpy as np


@dataclass
class Candidate:
    u_px: float
    v_px: float
    area_px: int
    radius_px: float
    circularity: float
    touches_border: bool
    peak: float


def _components(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        pixels = []
        while stack:
            r, c = stack.pop()
            pixels.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not (dr or dc):
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        yield np.asarray(pixels, dtype=int)


def robust_threshold(image, sigma=6.0):
    values = np.asarray(image, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    noise = 1.4826 * mad
    if noise < np.finfo(float).eps:
        noise = float(np.std(values))
    if noise < np.finfo(float).eps:
        raise ValueError("Flat image: cannot estimate a steel-ball threshold.")
    return median + sigma * noise


def detect_frame(image, *, threshold=None, threshold_sigma=6.0,
                 min_radius_px=1.5, max_radius_px=100.0,
                 min_circularity=0.55, border_margin_px=0,
                 reject_border=True, v_origin="up"):
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Each projection must be a finite 2-D image.")
    threshold = robust_threshold(image, threshold_sigma) if threshold is None else float(threshold)
    mask = image >= threshold
    candidates = []
    for pixels in _components(mask):
        area = len(pixels)
        radius = np.sqrt(area / np.pi)
        if radius < min_radius_px or radius > max_radius_px:
            continue
        rows, cols = pixels.T
        border = (rows.min() <= border_margin_px or cols.min() <= border_margin_px or
                  rows.max() >= image.shape[0] - 1 - border_margin_px or
                  cols.max() >= image.shape[1] - 1 - border_margin_px)
        if border and reject_border:
            continue
        boundary = 0
        for r, c in pixels:
            boundary += sum(not (0 <= r + dr < image.shape[0] and
                                 0 <= c + dc < image.shape[1] and mask[r + dr, c + dc])
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        circularity = 4 * np.pi * area / max(boundary * boundary, 1)
        if circularity < min_circularity:
            continue
        values = np.maximum(image[rows, cols] - threshold, 0.0)
        weight = values.sum()
        if weight <= 0:
            continue
        v = float(np.dot(rows, values) / weight)
        if v_origin == "up":
            v = image.shape[0] - 1 - v
        elif v_origin != "down":
            raise ValueError("v_origin must be 'up' or 'down'.")
        candidates.append(Candidate(
            float(np.dot(cols, values) / weight), v,
            area, float(radius), float(circularity), bool(border),
            float(image[rows, cols].max())))
    return candidates


def track_candidates(frame_candidates, *, expected_count=None, max_jump_px=40.0):
    """Track candidates from the first frame; missing/out-of-panel balls become NaN."""
    if not frame_candidates:
        raise ValueError("No steel-ball candidates were detected in the first frame.")
    first = sorted(frame_candidates[0], key=lambda c: (c.v_px, c.u_px))
    if expected_count is not None:
        first = first[:int(expected_count)]
    if len(first) < 3:
        raise ValueError("Fewer than three usable steel balls remain in the first frame.")
    tracks = np.full((len(frame_candidates), len(first), 2), np.nan, dtype=float)
    tracks[0] = [[c.u_px, c.v_px] for c in first]
    # Copy: updating the prediction for the next frame must not mutate frame 0.
    previous = np.asarray(tracks[0]).copy()
    active = np.ones(len(first), dtype=bool)
    for frame, candidates in enumerate(frame_candidates[1:], 1):
        available = list(range(len(candidates)))
        assignment = [-1] * len(first)
        active_indices = np.flatnonzero(active).tolist()
        # A lost track is terminal for this calibration run. It is never
        # reactivated by a later candidate, which prevents ID jumps at borders.
        if not active_indices:
            continue
        # Small candidate sets make an exact assignment clearer than greedy IDs.
        if len(available) <= len(active_indices) and len(available) <= 8:
            best = None
            for tracks_subset in combinations(active_indices, len(available)):
                for chosen in permutations(available):
                    distances = [np.linalg.norm(previous[i] - [candidates[j].u_px, candidates[j].v_px])
                                 for i, j in zip(tracks_subset, chosen)]
                    score = sum(distances)
                    if best is None or score < best[0]:
                        best = (score, tracks_subset, chosen, distances)
            if best:
                for i, (j, distance) in zip(best[1], zip(best[2], best[3])):
                    if distance <= max_jump_px:
                        assignment[i] = j
        else:
            for i in sorted(active_indices, key=lambda index: min(
                    (np.linalg.norm(previous[index] - [c.u_px, c.v_px])
                     for c in candidates), default=np.inf)):
                if not available:
                    break
                j = min(available, key=lambda candidate: np.linalg.norm(
                    previous[i] - [candidates[candidate].u_px, candidates[candidate].v_px]))
                if np.linalg.norm(previous[i] - [candidates[j].u_px, candidates[j].v_px]) <= max_jump_px:
                    assignment[i] = j
                    available.remove(j)
        for i, j in enumerate(assignment):
            if j >= 0:
                tracks[frame, i] = [candidates[j].u_px, candidates[j].v_px]
                previous[i] = tracks[frame, i]
            elif active[i]:
                active[i] = False
    visible = np.isfinite(tracks).all(axis=(0, 2))
    if visible.sum() < 3:
        raise ValueError("All steel-ball tracks are invalid or fewer than three remain after border filtering.")
    return tracks[:, visible], visible


def detect_and_track(stack, **kwargs):
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("Projection stack must have shape (views, rows, cols).")
    expected_count = kwargs.pop("expected_count", None)
    max_jump_px = kwargs.pop("max_jump_px", 40.0)
    candidates = [detect_frame(frame, **kwargs) for frame in stack]
    tracks, visible = track_candidates(candidates, expected_count=expected_count,
                                       max_jump_px=max_jump_px)
    return tracks, visible, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy projection stack [views, rows, cols]")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-sigma", type=float, default=6.0)
    parser.add_argument("--min-radius-px", type=float, default=1.5)
    parser.add_argument("--max-radius-px", type=float, default=100.0)
    parser.add_argument("--border-margin-px", type=int, default=0)
    parser.add_argument("--max-jump-px", type=float, default=40.0)
    args = parser.parse_args()
    stack = np.load(args.input, allow_pickle=False)
    options = vars(args).copy()
    options.pop("input"); options.pop("output")
    tracks, visible, candidates = detect_and_track(stack, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tracks_px=tracks, visible=visible,
                        candidate_count=np.array([len(c) for c in candidates]))
    print({"output": str(args.output), "tracks_shape": tracks.shape,
           "visible": visible.tolist(), "candidates_per_view": [len(c) for c in candidates]})


if __name__ == "__main__":
    main()
