"""Yang-specific coordinate declarations built from common transform tools."""
from __future__ import annotations

import numpy as np

from ..coordinates.detector import (
    DetectorConvention,
    FDK_TEST_CONVENTION,
    convert_projection_world_and_detector,
)
from ..coordinates.phantom import PhantomFrameTransform, PointCorrespondence


YANG_PAPER_CONVENTION = DetectorConvention(
    "Yang PIC: u-right, v-up",
    v_sign=-1,
    u_axis=(1.0, 0.0, 0.0),
    v_axis=(0.0, 0.0, -1.0),
    n_axis=(0.0, 1.0, 0.0),
)

# Yang uses (circumferential-x, longitudinal-y, circumferential-z), while
# fdk-test uses (circumferential-x, circumferential-y, longitudinal-z).
YANG_PHANTOM_TO_FDK = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

YANG_PHANTOM_TO_FDK_FRAME = PhantomFrameTransform(
    rotation=YANG_PHANTOM_TO_FDK,
    source_frame="yang-paper-phantom",
    target_frame="fdk-test-phantom",
)

# The Yang RAW tracker emits the lower spatial ring first. This is target
# identity, not a coordinate transform.
YANG_TRACKER_RING_ORDER = PointCorrespondence(
    tuple(range(6, 12)) + tuple(range(6)),
    name="yang-tracker-first-ring-then-second-ring",
)


def yang_projection_to_fdk(projection_matrix: np.ndarray,
                           image_shape: tuple[int, int]) -> np.ndarray:
    """Convert one Yang paper camera to the fdk-test world/image frame."""
    return convert_projection_world_and_detector(
        projection_matrix,
        image_shape,
        phantom_source_to_target=YANG_PHANTOM_TO_FDK,
        source_detector=YANG_PAPER_CONVENTION,
        target_detector=FDK_TEST_CONVENTION,
    )
