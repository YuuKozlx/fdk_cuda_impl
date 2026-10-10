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

from .dlt import DltConfig
from .phantom import marked_points
from .joint_fit import FIXED_PARAMETERS, fit, project


class ConstrainedJointTests(unittest.TestCase):
    def setUp(self):
        self.config = DltConfig(pixel_size_mm=(0.417, 0.417))
        self.points = marked_points()
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
        report = fit(tracks, max_nfev=200)
        machine = report["machine"]

        recovered = np.array([
            machine["sid_mm"], machine["sdd_mm"],
            machine["offset_u_mm"], machine["offset_v_mm"],
            machine["tilt_u_deg"], machine["tilt_v_deg"],
            machine["tilt_n_deg"],
        ])
        expected = np.array([440.0, 770.0, 2.085, 4.170, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(recovered, expected, atol=1.0e-8)
        self.assertLess(report["diagnostics"]["rmse_px"], 1.0e-9)
        self.assertEqual(
            {key: machine[key] for key in FIXED_PARAMETERS}, FIXED_PARAMETERS)

    def test_masked_corrupt_points_never_enter_dlt_or_joint_residual(self):
        tracks = project(self.truth, self.points, self.angles,
                         self.pixel, self.image_center)
        usable = np.ones(tracks.shape[:2], dtype=bool)
        for frame in range(len(tracks)):
            targets = np.arange(frame % 4)
            usable[frame, targets] = False
            tracks[frame, targets] += [80.0, -60.0]
        report = fit(tracks, max_nfev=200, usable_mask=usable)
        self.assertLess(report["diagnostics"]["rmse_px"], 1.0e-8)
        self.assertAlmostEqual(report["machine"]["sdd_mm"], 770.0, places=7)
        self.assertEqual(report["diagnostics"]["usable_observation_count"],
                         int(usable.sum()))

    def test_too_few_points_are_rejected_before_dlt(self):
        tracks = project(self.truth, self.points, self.angles,
                         self.pixel, self.image_center)
        usable = np.ones(tracks.shape[:2], dtype=bool)
        usable[3, 5:] = False
        with self.assertRaisesRegex(ValueError, "view 3 has only 5"):
            fit(tracks, usable_mask=usable)


if __name__ == "__main__":
    unittest.main()
