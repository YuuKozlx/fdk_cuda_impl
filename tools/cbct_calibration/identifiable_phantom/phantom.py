"""Known 3-D layout of the identifiable unequal-ring calibration phantom."""
from __future__ import annotations

import numpy as np


def marked_points(marker_points: tuple[tuple[float, float, float], ...] | None = None) -> np.ndarray:
    """Return 24 ring beads plus the off-plane marker in canonical ID order.

    IDs alternate upper/lower ring.  The unequal radii, 15-degree phase shift,
    and final marker make the 3-D/2-D correspondence observable from images.
    """
    points = []
    for index in range(12):
        upper = np.deg2rad(30.0 * index)
        lower = np.deg2rad(30.0 * index + 15.0)
        points.append([50.0 * np.cos(upper), 50.0 * np.sin(upper), 50.0])
        points.append([40.0 * np.cos(lower), 40.0 * np.sin(lower), -50.0])
    if marker_points is None:
        marker_points = ((50.0, 0.0, 80.0),)
    points.extend(np.asarray(marker_points, dtype=float).reshape((-1, 3)).tolist())
    return np.asarray(points, dtype=float)
