"""Focused contracts for the retained single-row image-to-ellipse workflow."""
import unittest

import numpy as np

from .calibrate import demo_data, calibrate, project
from .detect import detect_and_track


class SingleRowWorkflowTests(unittest.TestCase):
    def test_359_views_do_not_require_opposite_pairs(self):
        data, truth = demo_data(0.0)
        angles = np.deg2rad(np.linspace(0.0, 359.0, 359))
        beads = np.column_stack([np.full(5, 24.0), np.full(5, 8.0),
                                 np.arange(5) * 6.0 - 10.0])
        data["angles_rad"] = angles
        data["tracks_px"] = project(truth, beads, angles, data["pixel_size_mm"])
        result, _ = calibrate(**data, min_axis_ratio=0.0,
                              min_pair_index_gap=1)
        self.assertAlmostEqual(result["geometry"]["sdd_mm"], truth.sdd_mm,
                               delta=1.0)
        self.assertNotIn("antipodal_pairing", result["diagnostics"])

    def test_calibration_reports_explicit_stages(self):
        data, _ = demo_data(0.02)
        result, predicted = calibrate(**data, refine=True,
                                      min_axis_ratio=0.0,
                                      min_pair_index_gap=1)
        self.assertEqual(list(result["stages"]), [
            "stage1_ellipse_and_inplane_angle",
            "stage2_sdd_and_principal_point",
            "stage3_sod_and_bead_positions",
            "stage4_joint_refinement",
        ])
        self.assertTrue(result["stages"]["stage4_joint_refinement"]["enabled"])
        self.assertEqual(predicted.shape[0], len(data["angles_rad"]))
        self.assertLess(result["diagnostics"]["rmse_px"], 0.2)

    def test_tracking_permanently_disables_a_lost_target(self):
        views, rows, cols = 12, 80, 100
        yy, xx = np.indices((rows, cols))
        stack = np.zeros((views, rows, cols), dtype=float)
        # Four initial tracks, one terminal loss: three complete ellipses remain,
        # which is the minimum accepted by the calibration stage.
        starts = [(15.0, 15.0), (35.0, 30.0), (55.0, 45.0), (75.0, 60.0)]
        for frame in range(views):
            for target, (u, row) in enumerate(starts):
                if target == 0 and frame >= 6:
                    continue
                stack[frame] += 20.0 * np.exp(-((xx-u-.2*frame)**2 + (yy-row)**2)/(2*2.0**2))
        tracks, visible, _ = detect_and_track(
            stack, threshold=3.0, min_radius_px=1.0, max_radius_px=8.0,
            expected_count=4, max_jump_px=5.0)
        self.assertEqual(tracks.shape[1], 3)
        self.assertEqual(int(visible.sum()), 3)
        self.assertEqual(int((~visible).sum()), 1)
        self.assertTrue(np.isfinite(tracks).all())


if __name__ == "__main__":
    unittest.main()
