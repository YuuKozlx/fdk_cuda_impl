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
from .tracker import (FormalTrackerConfig, _extract_from_predictions,
                      initialize_indices, _track_direction)


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

    def test_touching_targets_are_excluded_from_calibration(self):
        yy, xx = np.indices((80, 80))
        image = np.zeros((80, 80), dtype=np.float32)
        image[(xx - 38) ** 2 + (yy - 40) ** 2 <= 5 ** 2] = 20.0
        image[(xx - 44) ** 2 + (yy - 40) ** 2 <= 5 ** 2] = 20.0
        _, valid, stats = _extract_from_predictions(
            image, np.array([[38.0, 40.0], [44.0, 40.0]]),
            threshold=5.0, gate_radius_px=12.0, min_pixels=4)
        np.testing.assert_array_equal(valid, [True, True])
        self.assertEqual(stats["merged_ids"], [0, 1])

    def test_merged_frame_does_not_retire_tracks(self):
        raw = np.zeros((3, 40, 40), dtype=np.float32)
        raw[:, 19:22, 9:12] = 20.0
        raw[:, 19:22, 19:22] = 20.0
        raw[1, 20, 12:19] = 20.0
        seeds = np.array([[10.0, 20.0], [20.0, 20.0]])
        config = FormalTrackerConfig(Path("."), Path("."), views=3,
                                     rows=40, cols=40, gate_radius_px=10.0)
        _, observed, merged, _ = _track_direction(raw, 0, seeds, 1, config)
        np.testing.assert_array_equal(observed, np.ones((3, 2), dtype=bool))
        np.testing.assert_array_equal(merged, [[False, False], [True, True],
                                              [False, False]])

    def test_missing_target_outside_gate_is_not_marked_merged(self):
        image = np.zeros((40, 40), dtype=np.float32)
        image[19:22, 9:12] = 20.0
        _, valid, stats = _extract_from_predictions(
            image, np.array([[10., 20.], [35., 20.]]), 5., 5., 4)
        np.testing.assert_array_equal(valid, [True, False])
        self.assertEqual(stats["merged_ids"], [])


if __name__ == "__main__":
    unittest.main()
