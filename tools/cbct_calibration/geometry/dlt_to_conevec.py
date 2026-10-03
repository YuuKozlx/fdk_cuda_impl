"""Exact conversion between a projective camera and cone-vector ray geometry."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConeVector:
    """Source and detector pixel-plane vectors in one world coordinate frame.

    ``detector_pixel(u,v) = det_s + u * r_u + v * r_v``.
    ``det_s`` is the centre of pixel coordinate (0,0), not an image corner.
    """

    source: np.ndarray
    det_s: np.ndarray
    r_u: np.ndarray
    r_v: np.ndarray
    gauge_scale: float
    source_detector_distance_mm: float | None


def conevec_normal(cone: ConeVector) -> np.ndarray:
    """Return the right-handed detector normal from the pixel vectors."""
    normal = np.cross(np.asarray(cone.r_u, dtype=float),
                      np.asarray(cone.r_v, dtype=float))
    length = float(np.linalg.norm(normal))
    if length < 1.0e-14:
        raise ValueError("detector pixel vectors are collinear")
    return normal / length


def reindex_conevec(cone: ConeVector, image_shape: tuple[int, int],
                    *, flip_u: bool = False, flip_v: bool = False) -> ConeVector:
    """Reindex image pixels by updating the complete 3-D detector plane."""
    rows, columns = map(int, image_shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("image_shape must be positive")
    det_s = np.asarray(cone.det_s, dtype=float).copy()
    r_u = np.asarray(cone.r_u, dtype=float).copy()
    r_v = np.asarray(cone.r_v, dtype=float).copy()
    if flip_u:
        det_s += (columns - 1) * r_u
        r_u = -r_u
    if flip_v:
        det_s += (rows - 1) * r_v
        r_v = -r_v
    result = ConeVector(np.asarray(cone.source, dtype=float).copy(), det_s,
                        r_u, r_v, cone.gauge_scale,
                        cone.source_detector_distance_mm)
    conevec_normal(result)
    return result


def _validate_projection(projection: np.ndarray) -> np.ndarray:
    p = np.asarray(projection, dtype=float)
    if p.shape != (3, 4) or not np.isfinite(p).all():
        raise ValueError("projection must be a finite 3x4 matrix")
    if abs(np.linalg.det(p[:, :3])) < 1.0e-14:
        raise ValueError("projection has a singular 3x3 camera block")
    return p


def dlt_to_conevec(projection: np.ndarray, *, sdd_mm: float | None = None,
                   point_toward: np.ndarray | None = None) -> ConeVector:
    """Convert ``P`` to exactly equivalent source/detector vectors.

    A DLT matrix fixes rays but not which plane transverse to those rays is
    called the physical detector.  With ``sdd_mm=None`` this function chooses
    the algebraic plane supplied by ``M^-1``.  Supplying SDD scales that plane
    to the requested perpendicular source-to-plane distance without changing
    a single ray.  ``point_toward`` resolves the arbitrary projective sign.
    """
    p = _validate_projection(projection)
    inverse = np.linalg.inv(p[:, :3])
    source = -inverse @ p[:, 3]
    r_u, r_v, base = inverse[:, 0], inverse[:, 1], inverse[:, 2]

    sign = 1.0
    if point_toward is not None and np.dot(base, np.asarray(point_toward) - source) < 0.0:
        sign = -1.0
    r_u, r_v, base = sign * r_u, sign * r_v, sign * base

    scale = 1.0
    if sdd_mm is not None:
        if not np.isfinite(sdd_mm) or sdd_mm <= 0.0:
            raise ValueError("sdd_mm must be positive and finite")
        normal = np.cross(r_u, r_v)
        normal /= np.linalg.norm(normal)
        plane_distance = abs(float(np.dot(base, normal)))
        if plane_distance < 1.0e-14:
            raise ValueError("DLT detector plane passes through the source")
        scale = float(sdd_mm) / plane_distance
        r_u, r_v, base = scale * r_u, scale * r_v, scale * base
    return ConeVector(source, source + base, r_u, r_v, scale,
                      None if sdd_mm is None else float(sdd_mm))


def conevec_to_dlt(cone: ConeVector) -> np.ndarray:
    """Return a 3x4 camera matrix for a cone-vector geometry."""
    source = np.asarray(cone.source, dtype=float)
    base = np.asarray(cone.det_s, dtype=float) - source
    basis = np.column_stack([cone.r_u, cone.r_v, base])
    if abs(np.linalg.det(basis)) < 1.0e-14:
        raise ValueError("cone vectors do not span a detector camera basis")
    world_to_detector = np.linalg.inv(basis)
    return np.column_stack([world_to_detector, -world_to_detector @ source])


def ray_from_conevec(cone: ConeVector, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    direction = (np.asarray(cone.det_s) + u * np.asarray(cone.r_u)
                 + v * np.asarray(cone.r_v) - np.asarray(cone.source))
    direction /= np.linalg.norm(direction)
    return np.asarray(cone.source), direction
