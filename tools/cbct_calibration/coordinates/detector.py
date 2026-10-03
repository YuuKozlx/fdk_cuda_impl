"""Explicit adapters between paper detector coordinates and fdk-test pixels.

The public fdk-test convention is image coordinates: u increases by column,
v increases by row (downward). Pixel coordinates refer to pixel centres.
Paper methods may use a different detector-frame orientation while observing
the same image. Keep that change at the API boundary, not inside the solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..math.rigid import RigidTransform


@dataclass(frozen=True)
class DetectorConvention:
    """Complete right-handed 3-D detector convention plus image signs."""

    name: str
    u_sign: int = 1
    v_sign: int = 1
    u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    v_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    n_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)

    def __post_init__(self) -> None:
        if self.u_sign not in (-1, 1) or self.v_sign not in (-1, 1):
            raise ValueError("detector axis signs must be +1 or -1")
        u, v, n = map(lambda x: np.asarray(x, dtype=float),
                      (self.u_axis, self.v_axis, self.n_axis))
        if any(x.shape != (3,) for x in (u, v, n)):
            raise ValueError("detector axes must be 3-vectors")
        if any(abs(np.linalg.norm(x) - 1.0) > 1e-8 for x in (u, v, n)):
            raise ValueError("detector axes must be unit vectors")
        if not np.allclose(np.cross(u, v), n, atol=1e-8):
            raise ValueError("detector convention must satisfy U cross V = N")


FDK_TEST_CONVENTION = DetectorConvention("fdk-test: u-right, v-down")


def phantom_frame_homography(rotation: np.ndarray,
                             translation: np.ndarray | None = None) -> np.ndarray:
    """Return the homogeneous map from target-frame points to source-frame points.

    ``rotation`` is defined by ``x_target = rotation @ x_source`` for column
    vectors.  A projection matrix written in the source frame therefore uses
    ``[rotation.T, 0; 0, 1]`` on its right-hand side.  Keeping this operation
    here makes the direction explicit and prevents a paper/world transform
    from being silently applied twice in a workflow.
    """
    transform = RigidTransform(
        np.asarray(rotation, dtype=float),
        np.zeros(3) if translation is None else np.asarray(translation, dtype=float),
    )
    return transform.inverse().homogeneous_matrix()


def convert_projection_world_and_detector(
    projection_matrix: np.ndarray,
    image_shape: tuple[int, int],
    *,
    phantom_source_to_target: np.ndarray | None = None,
    phantom_translation: np.ndarray | None = None,
    source_detector: DetectorConvention = FDK_TEST_CONVENTION,
    target_detector: DetectorConvention = FDK_TEST_CONVENTION,
) -> np.ndarray:
    """Convert a paper camera to one canonical world/image convention.

    The returned matrix maps points in the target phantom/world frame to
    pixels in the target detector image convention.  Both transformations are
    applied exactly once:

    ``P_target = H_detector @ P_source @ H_world``.

    ``phantom_source_to_target`` must map source-frame 3-D column vectors to
    target-frame vectors.  The detector conversion is still performed through
    the complete cone-vector plane, so reversing an image row also updates
    ``detS``, ``detV`` and the derived right-handed normal together.
    """
    p = np.asarray(projection_matrix, dtype=float)
    if p.shape != (3, 4):
        raise ValueError("projection_matrix must have shape (3, 4)")
    if phantom_source_to_target is not None:
        p = p @ phantom_frame_homography(
            phantom_source_to_target, phantom_translation)
    if source_detector == target_detector:
        return p
    return convert_projection_matrix(p, image_shape, source_detector,
                                     target_detector)


def detector_frame_basis(convention: DetectorConvention) -> np.ndarray:
    """Return the 3x3 basis matrix with columns U, V, N."""
    return np.column_stack((convention.u_axis, convention.v_axis,
                            convention.n_axis)).astype(float)


def detector_frame_transform(source: DetectorConvention,
                             target: DetectorConvention = FDK_TEST_CONVENTION) -> np.ndarray:
    """Return the proper 3-D rotation from source detector frame to target."""
    return detector_frame_basis(target).T @ detector_frame_basis(source)


def convert_detector_frame_axes(axes: np.ndarray,
                                source: DetectorConvention,
                                target: DetectorConvention = FDK_TEST_CONVENTION) -> np.ndarray:
    """Convert physical detector vectors while preserving full 3-D axes."""
    values = np.asarray(axes, dtype=float)
    if values.shape[-1:] != (3,):
        raise ValueError("axes must have final dimension 3")
    return values @ detector_frame_transform(source, target).T


def convert_detector_points(
    points: np.ndarray,
    image_shape: tuple[int, int],
    source: DetectorConvention,
    target: DetectorConvention = FDK_TEST_CONVENTION,
) -> np.ndarray:
    """Convert ``[..., (u,v)]`` pixel-centre coordinates between conventions.

    ``image_shape`` is ``(rows, columns)``. The returned array is a copy.
    This supports the present axis-sign convention changes (no axis swap).
    """

    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError("points must have final dimension (u, v)")
    if len(image_shape) != 2 or image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError("image_shape must be positive (rows, columns)")

    rows, columns = map(int, image_shape)
    result = values.copy()
    if source.u_sign != target.u_sign:
        result[..., 0] = (columns - 1) - result[..., 0]
    if source.v_sign != target.v_sign:
        result[..., 1] = (rows - 1) - result[..., 1]
    return result


def detector_homography(
    image_shape: tuple[int, int],
    source: DetectorConvention,
    target: DetectorConvention = FDK_TEST_CONVENTION,
) -> np.ndarray:
    """Return the homogeneous pixel transform ``q_target ~ H q_source``.

    Pixel coordinates are centre-indexed, so an axis reflection is
    ``q' = (N - 1) - q``.  The matrix is useful for projection matrices and
    preserves the exact same convention as :func:`convert_detector_points`.
    """
    rows, columns = map(int, image_shape)
    if rows <= 0 or columns <= 0:
        raise ValueError("image_shape must be positive (rows, columns)")
    h = np.eye(3, dtype=float)
    if source.u_sign != target.u_sign:
        h[0, 0] = -1.0
        h[0, 2] = float(columns - 1)
    if source.v_sign != target.v_sign:
        h[1, 1] = -1.0
        h[1, 2] = float(rows - 1)
    return h


def convert_projection_matrix(
    projection_matrix: np.ndarray,
    image_shape: tuple[int, int],
    source: DetectorConvention,
    target: DetectorConvention = FDK_TEST_CONVENTION,
) -> np.ndarray:
    """Reindex a projection through its complete 3-D cone geometry."""
    p = np.asarray(projection_matrix, dtype=float)
    if p.shape != (3, 4):
        raise ValueError("projection_matrix must have shape (3, 4)")
    from ..geometry.dlt_to_conevec import (conevec_to_dlt, dlt_to_conevec,
                                           reindex_conevec)
    cone = dlt_to_conevec(p)
    converted = reindex_conevec(
        cone, image_shape,
        flip_u=source.u_sign != target.u_sign,
        flip_v=source.v_sign != target.v_sign)
    return conevec_to_dlt(converted)


def convert_pose_pixel_fields(
    pose: dict,
    image_shape: tuple[int, int],
    source: DetectorConvention,
    target: DetectorConvention = FDK_TEST_CONVENTION,
) -> dict:
    """Copy a serialized pose and convert fields expressed in image pixels."""
    result = dict(pose)
    for key in ("o_px", "d_px", "principal_point_px"):
        if key in result and result[key] is not None:
            result[key] = convert_detector_points(
                np.asarray(result[key]), image_shape, source, target).tolist()
    return result
