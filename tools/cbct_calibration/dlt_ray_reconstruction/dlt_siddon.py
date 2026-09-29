"""Small, readable DLT-to-ray-driven projector.

The projection matrix determines a camera centre and one ray per detector
sample.  Siddon's voxel-box intersections provide the path lengths used by
both the forward projector and its exact sparse transpose.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy import sparse


def load_projection_matrices(report_path: Path, frame_ids: np.ndarray
                             ) -> list[np.ndarray]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    geometries = report["reconstruction_geometries"]
    return [np.asarray(geometries[int(frame)]["projection_matrix"], dtype=float)
            for frame in frame_ids]


def camera_from_dlt(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return camera centre C and the inverse 3x3 projection block."""
    p = np.asarray(projection, dtype=float)
    if p.shape != (3, 4):
        raise ValueError("projection matrix must have shape (3,4)")
    block = p[:, :3]
    inverse = np.linalg.inv(block)
    centre = -inverse @ p[:, 3]
    return centre, inverse


def ray_from_pixel(projection: np.ndarray, u: float, v: float,
                   volume_centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre, inverse = camera_from_dlt(projection)
    direction = inverse @ np.array([u, v, 1.0], dtype=float)
    direction /= np.linalg.norm(direction)
    # A projective matrix has an arbitrary sign.  Orient the ray toward the
    # reconstruction volume so t increases from the source into the object.
    if np.dot(direction, np.asarray(volume_centre) - centre) < 0.0:
        direction = -direction
    return centre, direction


def siddon_segments(source: np.ndarray, direction: np.ndarray,
                    shape: tuple[int, int, int], spacing_mm: float,
                    volume_centre: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened voxel indices and ray path lengths in millimetres."""
    # Coordinate order is z,y,x for the volume array, but the ray is x,y,z.
    half = 0.5 * np.asarray(shape, dtype=float)[::-1] * spacing_mm
    centre = np.asarray(volume_centre, dtype=float)
    bounds_min = centre - half
    bounds_max = centre + half
    source = np.asarray(source, dtype=float)
    direction = np.asarray(direction, dtype=float)
    near, far = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-14:
            if source[axis] < bounds_min[axis] or source[axis] > bounds_max[axis]:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
            continue
        t0 = (bounds_min[axis] - source[axis]) / direction[axis]
        t1 = (bounds_max[axis] - source[axis]) / direction[axis]
        near = max(near, min(t0, t1))
        far = min(far, max(t0, t1))
    if not np.isfinite(near) or far <= max(near, 0.0):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    near = max(near, 0.0)
    crossings = [near, far]
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-14:
            continue
        count = shape[::-1][axis]
        planes = bounds_min[axis] + spacing_mm * np.arange(1, count)
        values = (planes - source[axis]) / direction[axis]
        crossings.extend(values[(values > near) & (values < far)].tolist())
    t = np.unique(np.asarray(crossings, dtype=float))
    if len(t) < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    lengths = np.diff(t)
    mid = 0.5 * (t[:-1] + t[1:])
    xyz = source[None, :] + mid[:, None] * direction[None, :]
    ijk = np.floor((xyz - bounds_min[None, :]) / spacing_mm).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(shape[::-1])[None, :]), axis=1)
    ijk = ijk[inside]
    lengths = lengths[inside]
    # Convert x,y,z indices into a C-order z,y,x flattened volume index.
    flat = ((ijk[:, 2] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 0])
    return flat.astype(np.int64), lengths.astype(float)


def build_ray_matrix(projections: list[np.ndarray], detector_uv: np.ndarray,
                     volume_shape: tuple[int, int, int], spacing_mm: float,
                     volume_centre: np.ndarray
                     ) -> sparse.csr_matrix:
    """Build a sparse ray-driven system matrix A."""
    detector_uv = np.asarray(detector_uv, dtype=float)
    if detector_uv.ndim != 2 or detector_uv.shape[1] != 2:
        raise ValueError("detector_uv must have shape (rays_per_view,2)")
    rows, cols, values = [], [], []
    ray_count = len(projections) * len(detector_uv)
    row = 0
    for projection in projections:
        for u, v in detector_uv:
            source, direction = ray_from_pixel(
                projection, u, v, volume_centre)
            indices, lengths = siddon_segments(
                source, direction, volume_shape, spacing_mm, volume_centre)
            rows.extend([row] * len(indices))
            cols.extend(indices.tolist())
            values.extend(lengths.tolist())
            row += 1
    matrix = sparse.coo_matrix((values, (rows, cols)),
                               shape=(ray_count, int(np.prod(volume_shape))))
    return matrix.tocsr()

