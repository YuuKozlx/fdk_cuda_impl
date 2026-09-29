import unittest

import numpy as np

from calibrate import Geometry, calibrate, project
from detect import detect_and_track, detect_frame, track_candidates


def stack_with_spots(views=20, height=100, width=120):
    stack = np.zeros((views, height, width), dtype=float)
    centers = np.array([[25., 20.], [75., 48.], [50., 80.], [95., 75.]])
    for frame in range(views):
        for u, row in centers:
            uu, rr = np.meshgrid(np.arange(width), np.arange(height))
            stack[frame] += 20 * np.exp(-((uu - u - frame * .25) ** 2 +
                                          (rr - row) ** 2) / (2 * 2.2 ** 2))
    return stack


class DetectionTests(unittest.TestCase):
    def test_subpixel_centroid_and_v_conversion(self):
        image = stack_with_spots(1)[0]
        candidates = detect_frame(image, threshold=3, min_radius_px=1, max_radius_px=10)
        self.assertEqual(len(candidates), 4)
        centers = sorted((c.u_px, c.v_px) for c in candidates)
        self.assertAlmostEqual(centers[0][0], 25.0, delta=.03)
        self.assertAlmostEqual(centers[0][1], 79.0, delta=.03)

    def test_frame_to_frame_ids(self):
        tracks, visible, candidates = detect_and_track(
            stack_with_spots(), threshold=3, min_radius_px=1, max_radius_px=10,
            expected_count=4, max_jump_px=5)
        self.assertEqual(tracks.shape, (20, 4, 2))
        self.assertTrue(np.all(visible))
        self.assertTrue(np.all(np.diff(tracks[:, 0, 0]) > 0))

    def test_out_of_panel_track_is_excluded(self):
        stack = stack_with_spots()
        stack[8:, :, :45] = 0  # first target disappears after frame 7
        tracks, visible, _ = detect_and_track(
            stack, threshold=3, min_radius_px=1, max_radius_px=10,
            expected_count=4, max_jump_px=5)
        self.assertEqual(tracks.shape[1], 3)
        self.assertEqual(visible.tolist(), [True, True, True, False])

    def test_all_invalid_is_explicit_error(self):
        stack = stack_with_spots()
        stack[1:] = 0
        with self.assertRaisesRegex(ValueError, "invalid"):
            detect_and_track(stack, threshold=3, min_radius_px=1, max_radius_px=10,
                             expected_count=3, max_jump_px=5)

    def test_full_pipeline_image_to_geometry(self):
        """Synthetic FP images -> detection -> fixed IDs -> ellipse calibration."""
        truth = Geometry(250.0, 150.0, 180.0, 500.0, 1.2)
        beads = np.column_stack([
            np.full(5, 18.0), np.array([-30., -15., 0., 15., 30.]),
            np.array([-40., -20., 5., 25., 45.])])
        angles = np.arange(180) * 2 * np.pi / 180.0
        pitch = np.array([0.5, 0.5])
        expected = project(truth, beads, angles, pitch)
        height, width = 1000, 360
        rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        stack = np.zeros((len(angles), height, width), dtype=float)
        for frame in range(len(angles)):
            # Detector image has row-down coordinates; detector calibration uses v-up.
            for u, v in expected[frame]:
                row = height - 1 - v
                stack[frame] += 100 * np.exp(-((cols - u) ** 2 + (rows - row) ** 2) / (2 * 0.8 ** 2))
        tracks, visible, candidates = detect_and_track(
            stack, expected_count=5, threshold=10, min_radius_px=1,
            max_radius_px=5, max_jump_px=10)
        self.assertEqual(tracks.shape, (180, 5, 2))
        self.assertTrue(np.all(visible))
        self.assertTrue(all(len(frame) == 5 for frame in candidates))
        distances = np.linalg.norm(beads[:, None] - beads[None, :], axis=-1)
        result, prediction = calibrate(tracks, angles, pitch, distances)
        self.assertAlmostEqual(result["geometry"]["sdd_mm"], truth.sdd_mm, delta=1.0)
        self.assertAlmostEqual(result["geometry"]["sod_mm"], truth.sod_mm, delta=1.0)
        self.assertAlmostEqual(result["geometry"]["eta_deg"], truth.eta_deg, delta=.2)
        self.assertLess(result["diagnostics"]["rmse_px"], .3)
        self.assertLess(np.sqrt(np.mean((tracks - expected) ** 2)), .05)
        self.assertLess(np.sqrt(np.mean((prediction - tracks) ** 2)), .3)


if __name__ == "__main__":
    unittest.main()
