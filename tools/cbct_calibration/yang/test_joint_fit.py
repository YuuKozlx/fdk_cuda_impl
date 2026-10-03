"""Regression tests for Yang's 3-D target/index boundary."""
from __future__ import annotations

import unittest

import numpy as np

from .coordinate_adapter import YANG_TRACKER_RING_ORDER
from .joint_fit import phantom_points, phantom_points_paper
from .pic import YangConfig


class YangTargetMapTests(unittest.TestCase):
    def test_tracker_ring_order_matches_canonical_z_order(self):
        config = YangConfig(ring_radius_mm=50.0, ring_half_spacing_mm=50.0)
        paper = phantom_points_paper(config)
        model_order = phantom_points(config)
        canonical = phantom_points(config, YANG_TRACKER_RING_ORDER)

        # The tracker contract is first ring then second ring.  In the FDK
        # frame this is z=-l then z=+l; the paper list alone has the reverse
        # order because YANG_PHANTOM_TO_FDK maps paper y to canonical -z.
        self.assertTrue(np.allclose(canonical[:6, 2], -50.0))
        self.assertTrue(np.allclose(canonical[6:, 2], 50.0))
        self.assertTrue(np.allclose(model_order[:6, 2], 50.0))
        self.assertTrue(np.allclose(model_order[6:, 2], -50.0))
        self.assertTrue(np.allclose(np.sort(canonical[:, 0]),
                                    np.sort(np.tile(paper[:6, 0], 2))))


if __name__ == "__main__":
    unittest.main()
