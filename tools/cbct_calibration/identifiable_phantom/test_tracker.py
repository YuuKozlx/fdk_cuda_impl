"""Tests for image-only marked unequal-ring initialization and extraction."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for directory in (HERE, PARENT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from .phantom import marked_points
from .joint_fit import project
from .tracker import (_extract_from_predictions, initialize_indices)


class FormalTrackerTests(unittest.TestCase):
    def test_anchor_labels_are_resolved_by_physical_camera(self):
        world = marked_points()
        state = np.array([
            440.0, 770.0, 2.085, 4.17,
            np.deg2rad(1.0), np.deg2rad(2.0), np.deg2rad(3.0),
            np.deg2rad(2.0), 0.0, 0.0, 10.0, 15.0, 20.0,
        ])
        angle = np.array([107.0 * 2.0 * np.pi / 360.0])
        centres = project(state, world, angle, np.array([0.417, 0.417]),
                          np.array([511.5, 511.5]))[0]
        yy, xx = np.indices((1024, 1024))
        image = np.zeros((1024, 1024), dtype=np.float32)
        for target, (u, v) in enumerate(centres):
            sigma = 5.0 if target == 24 else 2.2
            image += 30.0 * np.exp(-((xx - u) ** 2 + (yy - v) ** 2)
                                    / (2.0 * sigma ** 2))
        indexed, diagnostics = initialize_indices(image, threshold=5.0)
        mirror = np.array([
            0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
            11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 24,
        ])
        direct_error = np.max(np.linalg.norm(indexed - centres, axis=1))
        mirror_error = np.max(
            np.linalg.norm(indexed - centres[mirror], axis=1))
        self.assertLess(min(direct_error, mirror_error), 0.05)
        self.assertEqual(diagnostics["candidate_count"], 96)
        self.assertLess(diagnostics["selected_physical_camera_rmse_px"], 0.05)

    def test_prediction_guided_pixels_split_touching_targets(self):
        yy, xx = np.indices((80, 80))
        image = (30.0 * np.exp(-((xx - 38.0) ** 2 + (yy - 40.0) ** 2) / 18.0)
                 + 30.0 * np.exp(-((xx - 43.0) ** 2 + (yy - 40.0) ** 2) / 18.0))
        predicted = np.array([[38.0, 40.0], [43.0, 40.0]])
        measured, valid, _ = _extract_from_predictions(
            image, predicted, threshold=5.0, gate_radius_px=12.0,
            min_pixels=4)
        self.assertTrue(np.all(valid))
        self.assertLess(measured[0, 0], measured[1, 0])
        self.assertLess(np.max(np.linalg.norm(measured - predicted, axis=1)), 1.5)


if __name__ == "__main__":
    unittest.main()
