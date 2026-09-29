"""Tests for Cho's analytic eta and complete per-view geometry."""
from __future__ import annotations

import unittest

import numpy as np

from cho_analytic import calibrate_frame_cho
from cho_calibration import ChoConfig, phantom_points

import sys
from pathlib import Path

COMMON_DIRECTORY = Path(__file__).resolve().parent.parent
if str(COMMON_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(COMMON_DIRECTORY))
from fit_right_hand_joint import project


class ChoAnalyticTests(unittest.TestCase):
    def setUp(self):
        self.config = ChoConfig(
            ring_radius_mm=50.0,
            ring_half_spacing_mm=50.0,
            beads_per_ring=12,
            pixel_size_mm=(0.417, 0.417),
        )
        self.points = phantom_points(self.config)
        self.pixel = np.asarray(self.config.pixel_size_mm)
        self.center = np.array([511.5, 511.5])

    def make_frame(self, tilt_u=0.0, tilt_v=0.0, tilt_n=0.0):
        state = np.array([
            440.0, 770.0, 0.0, 0.0,
            np.deg2rad(tilt_u), np.deg2rad(tilt_v), np.deg2rad(tilt_n),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        return project(state, self.points, np.array([0.0]),
                       self.pixel, self.center)[0]

    def test_zero_rotation(self):
        result = calibrate_frame_cho(self.make_frame(), self.config)
        self.assertAlmostEqual(result.detector_in_plane_angle_deg, 0.0, places=9)
        self.assertAlmostEqual(result.complete_geometry["cho_eta_deg"], 0.0,
                               places=9)
        self.assertEqual(result.eta_diagnostics["method"],
                         "cho_2005_equations_14_16")

    def test_equations_14_16_and_complete_pose(self):
        result = calibrate_frame_cho(self.make_frame(0.0, 0.0, 3.0),
                                     self.config)
        # Cho's ellipse construction and the full camera decomposition are
        # retained separately.  The latter exactly recovers this noiseless
        # physical detector rotation; the former follows Eqs. (14)-(16).
        self.assertAlmostEqual(result.detector_in_plane_angle_deg,
                               2.960717633489, places=8)
        self.assertAlmostEqual(result.complete_geometry["cho_eta_deg"],
                               3.0, places=9)
        self.assertAlmostEqual(
            result.complete_geometry["source_detector_distance_mm"],
            770.0, places=8)
        self.assertLess(result.complete_geometry["reprojection_rmse_px"],
                        1.0e-9)
        self.assertEqual(result.ellipse_lines.shape, (2, 4))
        self.assertEqual(result.converging_point_pa_px.shape, (2,))

    def test_complete_pose_with_three_detector_rotations(self):
        result = calibrate_frame_cho(self.make_frame(1.0, 2.0, 3.0),
                                     self.config)
        geometry = result.complete_geometry
        self.assertAlmostEqual(geometry["cho_phi_deg"], -2.0003044086,
                               places=7)
        self.assertAlmostEqual(geometry["cho_theta_deg"], 0.9993907652,
                               places=7)
        self.assertAlmostEqual(geometry["cho_eta_deg"], 3.0349030365,
                               places=7)
        self.assertLess(geometry["reprojection_rmse_px"], 1.0e-9)

    def test_six_targets_per_ring(self):
        config = ChoConfig(
            ring_radius_mm=50.0,
            ring_half_spacing_mm=50.0,
            beads_per_ring=6,
            pixel_size_mm=(0.417, 0.417),
        )
        points = phantom_points(config)
        state = np.array([
            440.0, 770.0, 0.0, 0.0,
            np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(3.0),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        measured = project(state, points, np.array([0.0]),
                           self.pixel, self.center)[0]
        result = calibrate_frame_cho(measured, config)
        self.assertAlmostEqual(result.complete_geometry["cho_eta_deg"],
                               3.0349030365, places=7)
        self.assertEqual(result.opposite_pair_lines.shape, (6, 3))
        self.assertLess(result.complete_geometry["reprojection_rmse_px"],
                        1.0e-9)


if __name__ == "__main__":
    unittest.main()
