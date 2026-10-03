"""Coordinate convention adapters for calibration and reconstruction."""

from .detector import (
    DetectorConvention,
    FDK_TEST_CONVENTION,
    convert_detector_points,
    detector_homography,
    convert_projection_matrix,
    convert_pose_pixel_fields,
    detector_frame_basis,
    detector_frame_transform,
    convert_detector_frame_axes,
    phantom_frame_homography,
    convert_projection_world_and_detector,
)
from .phantom import (
    PointCorrespondence,
    PhantomFrameTransform,
    transform_indexed_model_points,
)

__all__ = [
    "DetectorConvention",
    "FDK_TEST_CONVENTION",
    "convert_detector_points",
    "detector_homography",
    "convert_projection_matrix",
    "convert_pose_pixel_fields",
    "detector_frame_basis",
    "detector_frame_transform",
    "convert_detector_frame_axes",
    "phantom_frame_homography",
    "convert_projection_world_and_detector",
    "PointCorrespondence",
    "PhantomFrameTransform",
    "transform_indexed_model_points",
]
