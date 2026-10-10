"""Known 3-D layout of the marked helical-bead calibration phantom."""
from __future__ import annotations

import numpy as np


def spiral_points(radius_mm: float = 50.0, spacing_mm: float = 5.0,
                  phase_step_deg: float = 30.0, bead_count: int = 24,
                  marker_below_mm: float = 10.0) -> np.ndarray:
    """Return ordinary beads P0..P(n-1), followed by the axial marker M."""
    index = np.arange(bead_count, dtype=float)
    phase = np.deg2rad(index * phase_step_deg)
    first_z = -0.5 * (bead_count - 1) * spacing_mm
    ordinary = np.column_stack([
        radius_mm * np.cos(phase),
        radius_mm * np.sin(phase),
        first_z + index * spacing_mm,
    ])
    marker = np.array([[radius_mm, 0.0, first_z - marker_below_mm]])
    return np.vstack([ordinary, marker])
