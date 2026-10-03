"""Explicit 3-D frame transforms and target-index correspondences.

Frame transforms and point ordering are deliberately separate concepts.  A
rigid transform changes coordinates; a correspondence only says which model
target is represented by each observed/indexed point.  Keeping them separate
prevents a ring swap from being mistaken for a detector-axis or handedness
conversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..math.rigid import RigidTransform


@dataclass(frozen=True)
class PointCorrespondence:
    """Mapping from observed/indexed target order to model point order.

    ``model_indices[k]`` is the model-point row that corresponds to observed
    row ``k``.  This is intentionally not part of a coordinate-frame
    transform.
    """

    model_indices: tuple[int, ...]
    name: str = "unnamed"

    def __post_init__(self) -> None:
        indices = np.asarray(self.model_indices, dtype=int)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("model_indices must be a non-empty 1-D sequence")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("model_indices must not contain duplicates")
        if np.any(indices < 0):
            raise ValueError("model_indices must be non-negative")

    def apply(self, model_points: np.ndarray) -> np.ndarray:
        points = np.asarray(model_points, dtype=float)
        indices = np.asarray(self.model_indices, dtype=int)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("model_points must have shape (N, 3)")
        if np.any(indices >= len(points)):
            raise ValueError("correspondence index exceeds model point count")
        return points[indices].copy()


@dataclass(frozen=True)
class PhantomFrameTransform(RigidTransform):
    """Rigid map ``target = R @ source + t`` for row-wise point arrays."""

    source_frame: str = "source"
    target_frame: str = "target"

    def inverse(self) -> "PhantomFrameTransform":
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation, dtype=float)
        return PhantomFrameTransform(
            r.T, -r.T @ t, self.target_frame, self.source_frame)


def transform_indexed_model_points(
        model_points: np.ndarray,
        frame_transform: PhantomFrameTransform,
        correspondence: PointCorrespondence | None = None) -> np.ndarray:
    """Transform model coordinates, then arrange them in observation order.

    The operation order is explicit even though a row permutation and a
    rigid transform commute mathematically.  It documents the two independent
    contracts at the 3-D-to-2-D boundary: coordinate representation and
    target identity.
    """
    transformed = frame_transform.apply(model_points)
    if correspondence is None:
        correspondence = PointCorrespondence(
            tuple(range(len(transformed))), name="identity")
    return correspondence.apply(transformed)
