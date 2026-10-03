import unittest

import numpy as np

from .dlt import (ChoConfig, calibrate_stack_joint, calibrate_view,
                              phantom_points, project_points)
from .joint_fit import reverse_ring_handedness


class ChoCalibrationTests(unittest.TestCase):
    def test_reverse_handedness_preserves_marked_bead_zero(self):
        points = np.arange(2 * 12 * 2, dtype=float).reshape(1, 24, 2)
        reversed_points = reverse_ring_handedness(points, 12)
        np.testing.assert_array_equal(reversed_points[:, 0], points[:, 0])
        np.testing.assert_array_equal(reversed_points[:, 12], points[:, 12])
        np.testing.assert_array_equal(reversed_points[:, 1], points[:, 11])
        np.testing.assert_array_equal(reversed_points[:, 13], points[:, 23])

    def make_view(self, noise=0.0):
        config = ChoConfig(ring_radius_mm=50.0, ring_half_spacing_mm=80.0,
                           pixel_size_mm=(0.254, 0.317))
        points = phantom_points(config)
        sdd = 1600.0
        principal = np.array([511.5, 511.5]) + np.array([5.0, 10.0]) / np.array(config.pixel_size_mm)
        k = np.array([[sdd / config.pixel_size_mm[0], 0.0, principal[0]],
                      [0.0, sdd / config.pixel_size_mm[1], principal[1]],
                      [0.0, 0.0, 1.0]])
        angle = np.deg2rad([4.0, -3.0, 1.0])
        from scipy.spatial.transform import Rotation
        r = Rotation.from_rotvec(angle).as_matrix()
        t = np.array([70.0, -40.0, 1500.0])
        projection = k @ np.column_stack([r, t])
        measured = project_points(points, projection)
        if noise:
            measured += np.random.default_rng(42).normal(0.0, noise, measured.shape)
        return config, measured, np.linalg.inv(np.eye(1)) if False else projection, principal, sdd

    def test_noiseless_complete_geometry(self):
        config, measured, _, principal, sdd = self.make_view()
        pose = calibrate_view(measured, config)
        np.testing.assert_allclose(pose.principal_point_px, principal, atol=1e-7)
        self.assertAlmostEqual(pose.source_detector_distance_mm, sdd, places=6)
        self.assertLess(pose.reprojection_rmse_px, 1e-7)

    def test_noisy_all_beads_joint_fit(self):
        config, measured, _, principal, sdd = self.make_view(0.01)
        pose = calibrate_view(measured, config)
        self.assertLess(pose.reprojection_rmse_px, 0.02)
        # 24 points at 0.01 pixel centroid noise give a few-pixel principal
        # point uncertainty in this intentionally oblique configuration.
        self.assertLess(np.linalg.norm(pose.principal_point_px - principal), 3.0)
        self.assertLess(abs(pose.source_detector_distance_mm - sdd), 2.0)

    def test_joint_shared_intrinsics(self):
        config, _, _, principal, sdd = self.make_view()
        points = phantom_points(config)
        frames = []
        rng = np.random.default_rng(7)
        for i in range(24):
            q = 2.0 * np.pi * i / 24.0
            angle = np.array([0.03 * np.sin(2.0 * q),
                              0.02 * np.cos(3.0 * q), q])
            from scipy.spatial.transform import Rotation
            r = Rotation.from_rotvec(angle).as_matrix()
            t = np.array([8.0 * np.sin(q), 6.0 * np.cos(2.0 * q),
                          1500.0 + 3.0 * np.sin(3.0 * q)])
            k = np.array([[sdd / config.pixel_size_mm[0], 0.0, principal[0]],
                          [0.0, sdd / config.pixel_size_mm[1], principal[1]],
                          [0.0, 0.0, 1.0]])
            frames.append(project_points(points, k @ np.column_stack([r, t]))
                          + rng.normal(0, 0.01, (len(points), 2)))
        result = calibrate_stack_joint(np.asarray(frames), config)
        np.testing.assert_allclose(result.shared_offset_mm, [5.0, 10.0], atol=0.4)
        self.assertLess(abs(result.shared_sdd_mm - sdd), 0.5)


if __name__ == "__main__":
    unittest.main()
