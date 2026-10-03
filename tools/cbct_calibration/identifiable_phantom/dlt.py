"""Per-view DLT camera estimation for the identifiable 25-point phantom."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..math.dlt import decompose_projection, normalized_dlt, project_points


@dataclass(frozen=True)
class DltConfig:
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    robust_loss: str = "soft_l1"
    robust_scale_px: float = 0.15
    max_nfev: int = 500


@dataclass
class DltCamera:
    projection_matrix: np.ndarray
    source_phantom_mm: np.ndarray
    sdd_mm: float
    principal_point_px: np.ndarray
    reprojection_rmse_px: float


def calibrate_view(points_px: np.ndarray, config: DltConfig,
                   points_3d_mm: np.ndarray) -> DltCamera:
    """Refine one normalized-DLT camera with physical rectangular pixels."""
    world = np.asarray(points_3d_mm, dtype=float)
    measured = np.asarray(points_px, dtype=float)
    pixel = np.asarray(config.pixel_size_mm, dtype=float)
    projection0, _ = normalized_dlt(world, measured)
    k0, r0, t0 = decompose_projection(projection0, pixel)
    initial = np.r_[np.log(k0[0, 0] * pixel[0]), k0[0, 2], k0[1, 2],
                    Rotation.from_matrix(r0).as_rotvec(), t0]

    def unpack(values):
        sdd = np.exp(values[0])
        k = np.array([[sdd / pixel[0], 0.0, values[1]],
                      [0.0, sdd / pixel[1], values[2]], [0.0, 0.0, 1.0]])
        return k, Rotation.from_rotvec(values[3:6]).as_matrix(), values[6:9]

    def residual(values):
        k, rotation, translation = unpack(values)
        return (project_points(world, k @ np.column_stack([rotation, translation]))
                - measured).ravel()

    result = least_squares(residual, initial, loss=config.robust_loss,
                           f_scale=config.robust_scale_px,
                           max_nfev=config.max_nfev,
                           ftol=1e-13, xtol=1e-13, gtol=1e-13)
    intrinsic, rotation, translation = unpack(result.x)
    projection = intrinsic @ np.column_stack([rotation, translation])
    error = project_points(world, projection) - measured
    return DltCamera(projection, -rotation.T @ translation,
                     float(intrinsic[0, 0] * pixel[0]),
                     np.array([intrinsic[0, 2], intrinsic[1, 2]]),
                     float(np.sqrt(np.mean(error * error))))
