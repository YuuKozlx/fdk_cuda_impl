"""Right-handed 3-D rotations and rigid frame transforms.

This module contains only the numerical operation ``x_target = R x_source + t``.
Names such as ``Yang phantom frame`` or ``fdk-test frame`` belong in the
coordinate adapters, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RigidTransform:
    """A proper right-handed transform for row-wise 3-D points."""

    rotation: np.ndarray
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation must be (3,3) and translation (3,)")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) <= 0.0:
            raise ValueError("rigid transform must preserve handedness")

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply ``target = rotation @ source + translation`` to points."""
        values = np.asarray(points, dtype=float)
        if values.shape[-1:] != (3,):
            raise ValueError("points must have final dimension 3")
        return values @ np.asarray(self.rotation).T + np.asarray(self.translation)

    def inverse(self) -> "RigidTransform":
        """Return the transform from target coordinates back to source."""
        rotation = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        return RigidTransform(rotation.T, -rotation.T @ translation)

    def homogeneous_matrix(self) -> np.ndarray:
        """Return the column-vector matrix ``[R t; 0 1]``."""
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = np.asarray(self.rotation, dtype=float)
        matrix[:3, 3] = np.asarray(self.translation, dtype=float)
        return matrix
