"""Regression tests for the fixed-source, fixed-offset-n joint gauge."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

CURRENT_DIRECTORY = Path(__file__).resolve().parent
COMMON_DIRECTORY = CURRENT_DIRECTORY.parent
for directory in (CURRENT_DIRECTORY, COMMON_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cho_calibration import ChoConfig, phantom_points
from fit_right_hand_joint import FIXED_PARAMETER_VALUES, fit, project


class ConstrainedJointTests(unittest.TestCase):
    def setUp(self):
        self.config = ChoConfig(
            ring_radius_mm=50.0,
            ring_half_spacing_mm=50.0,
            beads_per_ring=12,
            pixel_size_mm=(0.417, 0.417),
        )
        self.points = phantom_points(self.config)
        self.angles = np.arange(24) * 2.0 * np.pi / 24.0
        self.pixel = np.array(self.config.pixel_size_mm)
        self.image_center = np.array([511.5, 511.5])
        self.truth = np.array([
            440.0, 770.0, 2.085, 4.170,
            np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(3.0),
            np.deg2rad(2.0), 0.0, 0.0, 10.0, 15.0, 20.0,
        ])

    def test_state_has_no_source_or_normal_offset_slots(self):
        with self.assertRaisesRegex(ValueError, "13-value"):
            project(np.r_[self.truth, 0.0, 0.0], self.points, self.angles,
                    self.pixel, self.image_center)

    def test_noiseless_joint_fit_recovers_seven_parameters(self):
        tracks = project(self.truth, self.points, self.angles,
                         self.pixel, self.image_center)
        report, _, _ = fit(tracks, max_nfev=200)

        recovered = np.array([
            report["sid_mm"], report["sdd_mm"],
            report["offset_u_mm"], report["offset_v_mm"],
            report["tilt_u_deg"], report["tilt_v_deg"],
            report["tilt_n_deg"],
        ])
        expected = np.array([440.0, 770.0, 2.085, 4.170, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(recovered, expected, atol=1.0e-8)
        self.assertLess(report["rmse_px"], 1.0e-9)
        self.assertEqual(report["fixed_parameters"], FIXED_PARAMETER_VALUES)


if __name__ == "__main__":
    unittest.main()
