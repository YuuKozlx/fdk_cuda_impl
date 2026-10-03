"""Normalized DLT and projective point utilities shared by calibration workflows."""
from __future__ import annotations

import numpy as np
from scipy.linalg import rq


EPS = 1.0e-12


def _normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=float)
    center = values.mean(axis=0)
    distance = np.linalg.norm(values - center, axis=1).mean()
    if distance < EPS:
        raise ValueError("degenerate 2-D point set")
    scale = np.sqrt(2.0) / distance
    transform = np.array([[scale, 0.0, -scale * center[0]],
                          [0.0, scale, -scale * center[1]],
                          [0.0, 0.0, 1.0]])
    homogeneous = np.column_stack([values, np.ones(len(values))])
    normalized = homogeneous @ transform.T
    return normalized[:, :2], transform


def _normalize_points_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(points, dtype=float)
    center = values.mean(axis=0)
    distance = np.linalg.norm(values - center, axis=1).mean()
    if distance < EPS:
        raise ValueError("degenerate 3-D point set")
    scale = np.sqrt(3.0) / distance
    transform = np.array([
        [scale, 0.0, 0.0, -scale * center[0]],
        [0.0, scale, 0.0, -scale * center[1]],
        [0.0, 0.0, scale, -scale * center[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])
    homogeneous = np.column_stack([values, np.ones(len(values))])
    normalized = homogeneous @ transform.T
    return normalized[:, :3], transform


def normalized_dlt(points_3d_mm: np.ndarray,
                   points_px: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate ``q ~ P X`` with Hartley-normalized DLT."""
    world = np.asarray(points_3d_mm, dtype=float)
    image = np.asarray(points_px, dtype=float)
    if world.ndim != 2 or world.shape[1] != 3 or image.shape != (len(world), 2):
        raise ValueError("expected matching arrays (N,3) and (N,2)")
    if len(world) < 6 or not np.isfinite(world).all() or not np.isfinite(image).all():
        raise ValueError("DLT needs at least six finite correspondences")

    world_normalized, world_transform = _normalize_points_3d(world)
    image_normalized, image_transform = _normalize_points_2d(image)
    rows = []
    for point, (u, v) in zip(world_normalized, image_normalized):
        homogeneous = np.r_[point, 1.0]
        rows.append(np.r_[homogeneous, np.zeros(4), -u * homogeneous])
        rows.append(np.r_[np.zeros(4), homogeneous, -v * homogeneous])
    _, singular, vt = np.linalg.svd(np.asarray(rows))
    normalized_projection = vt[-1].reshape(3, 4)
    projection = (np.linalg.inv(image_transform) @ normalized_projection
                  @ world_transform)
    projection /= max(np.linalg.norm(projection[2, :3]), EPS)
    return projection, {
        "condition": float(singular[0] / max(singular[-2], EPS)),
        "nullspace_gap": float(singular[-2] / max(singular[-1], EPS)),
    }


def project_points(points_3d_mm: np.ndarray,
                   projection: np.ndarray) -> np.ndarray:
    """Project finite 3-D points through a ``3x4`` camera matrix."""
    points = np.asarray(points_3d_mm, dtype=float)
    homogeneous = (np.column_stack([points, np.ones(len(points))])
                   @ np.asarray(projection, dtype=float).T)
    if np.any(np.abs(homogeneous[:, 2]) < EPS):
        raise ValueError("point projects at infinity")
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def decompose_projection(projection: np.ndarray,
                         pixel_size_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero-skew physical ``K, R, t`` from a projective camera."""
    matrix = np.asarray(projection, dtype=float)
    pixel = np.asarray(pixel_size_mm, dtype=float)
    intrinsic, rotation = rq(matrix[:, :3])
    signs = np.sign(np.diag(intrinsic))
    signs[signs == 0] = 1.0
    diagonal = np.diag(signs)
    intrinsic, rotation = intrinsic @ diagonal, diagonal @ rotation
    if np.linalg.det(rotation) < 0:
        intrinsic, rotation = -intrinsic, -rotation
    scale = intrinsic[2, 2]
    if abs(scale) < EPS:
        raise ValueError("invalid projection scale")
    intrinsic /= scale
    scaled_projection = matrix / scale
    sdd = 0.5 * (abs(intrinsic[0, 0]) * pixel[0]
                 + abs(intrinsic[1, 1]) * pixel[1])
    physical_intrinsic = np.array([
        [sdd / pixel[0], 0.0, intrinsic[0, 2]],
        [0.0, sdd / pixel[1], intrinsic[1, 2]],
        [0.0, 0.0, 1.0],
    ])
    translation = np.linalg.solve(intrinsic, scaled_projection[:, 3])
    return physical_intrinsic, rotation, translation
