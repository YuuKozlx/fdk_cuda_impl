"""Fit the DLT source trajectory and build a phantom-to-scanner seed."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SourceCircle:
    center_phantom_mm: np.ndarray
    axis_phantom: np.ndarray
    radial0_phantom: np.ndarray
    tangent0_phantom: np.ndarray
    radius_mm: float
    radial_rms_mm: float
    axial_rms_mm: float


def fit_source_circle(source_points_phantom_mm: np.ndarray) -> SourceCircle:
    points = np.asarray(source_points_phantom_mm, dtype=float)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    axis = vt[-1]
    if np.dot(np.cross(points[0] - centroid, points[1] - centroid), axis) < 0:
        axis = -axis
    radial0 = points[0] - centroid
    radial0 -= axis * np.dot(radial0, axis)
    radial0 /= np.linalg.norm(radial0)
    tangent0 = np.cross(axis, radial0)
    xy = np.column_stack([(points - centroid) @ radial0,
                          (points - centroid) @ tangent0])
    design = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    a, b, _ = np.linalg.lstsq(design, -np.sum(xy * xy, axis=1), rcond=None)[0]
    center = centroid - 0.5 * a * radial0 - 0.5 * b * tangent0
    delta = points - center
    axial = delta @ axis
    planar = delta - axial[:, None] * axis
    radii = np.linalg.norm(planar, axis=1)
    radius = float(np.mean(radii))
    radial0 = planar[0] / np.linalg.norm(planar[0])
    tangent0 = np.cross(axis, radial0)
    return SourceCircle(center, axis, radial0, tangent0, radius,
                        float(np.sqrt(np.mean((radii - radius) ** 2))),
                        float(np.sqrt(np.mean(axial ** 2))))


def scanner_alignment(circle: SourceCircle) -> tuple[np.ndarray, np.ndarray]:
    """Map the fitted circle to radial -Y, tangent +X, axial +Z at view zero."""
    axis = circle.axis_phantom.copy()
    radial = circle.radial0_phantom.copy()
    if axis[2] < 0:
        axis, radial = -axis, -radial
    tangent = np.cross(axis, radial); tangent /= np.linalg.norm(tangent)
    radial = np.cross(tangent, axis)
    phantom_basis = np.column_stack([radial, tangent, axis])
    scanner_basis = np.column_stack([[0.0, -1.0, 0.0],
                                     [1.0, 0.0, 0.0],
                                     [0.0, 0.0, 1.0]])
    rotation = scanner_basis @ phantom_basis.T
    return rotation, -rotation @ circle.center_phantom_mm
