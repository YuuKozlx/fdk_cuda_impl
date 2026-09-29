"""Numerical recovery, independent ray geometry, and input-contract tests."""

import unittest

import numpy as np

from calibrate import Geometry, fit_ellipse, calibrate, demo_data, project


def ray_plane_projection(g, beads, angles, pitch):
    """Independent 3-D ray/plane oracle for the coordinate convention."""
    eta = np.deg2rad(g.eta_deg)
    source = np.array([-g.sod_mm, 0, 0])
    detector_origin = np.array([g.sdd_mm - g.sod_mm, 0, 0])
    u_axis = np.array([0, -np.cos(eta), np.sin(eta)])
    v_axis = np.array([0, np.sin(eta), np.cos(eta)])
    output = []
    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rotated = beads @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        rays = rotated - source
        hits = source + rays * (g.sdd_mm / rays[:, :1])
        relative = hits - detector_origin
        output.append(np.column_stack([relative @ u_axis, relative @ v_axis]) / pitch
                      + [g.u0_px, g.v0_px])
    return np.array(output)


class CalibrationTests(unittest.TestCase):
    def assert_geometry(self, actual, expected, atol=1e-6):
        for key, value in vars(expected).items():
            self.assertAlmostEqual(actual[key], value, delta=atol, msg=key)

    def test_noiseless_rod(self):
        data, truth = demo_data(0)
        result, prediction = calibrate(**data)
        self.assert_geometry(result["geometry"], truth)
        np.testing.assert_allclose(prediction, data["tracks_px"], atol=1e-7)

    def test_independent_geometry_non_square_pixels_reverse_scan_and_phases(self):
        truth = Geometry(800, 430, 703.4, 521.2, -3.2)
        beads = np.array([[35, 4, -40], [-20, -15, -11], [15, -31, 20], [25, 30, 47]])
        pitch = np.array([0.22, 0.35])
        angles = 0.31 - np.arange(240) * 2 * np.pi / 240
        tracks = ray_plane_projection(truth, beads, angles, pitch)
        np.testing.assert_allclose(project(truth, beads, angles, pitch), tracks, atol=1e-10)
        distances = np.full((4, 4), np.nan)
        np.fill_diagonal(distances, 0)
        distances[0, 1] = distances[1, 0] = np.linalg.norm(beads[0] - beads[1])
        result, prediction = calibrate(tracks, angles, pitch, distances)
        self.assert_geometry(result["geometry"], truth)
        np.testing.assert_allclose(prediction, tracks, atol=1e-7)
        np.testing.assert_allclose(result["beads_mm"], beads, atol=1e-7)

    def test_noisy_refinement(self):
        data, truth = demo_data(0.1)
        result, _ = calibrate(**data, refine=True)
        self.assertEqual(list(result["stages"]), [
            "stage1_ellipse_and_inplane_angle",
            "stage2_sdd_and_principal_point",
            "stage3_sod_and_bead_positions",
            "stage4_joint_refinement",
        ])
        self.assertLess(result["diagnostics"]["rmse_px"], 0.11)
        self.assertLess(result["diagnostics"]["rmse_px"],
                        result["diagnostics"]["analytic_rmse_px"])
        self.assertAlmostEqual(result["geometry"]["sdd_mm"], truth.sdd_mm, delta=0.3)
        self.assertAlmostEqual(result["geometry"]["sod_mm"], truth.sod_mm, delta=0.2)

    def test_analytic_stages_stop_before_refinement(self):
        data, _ = demo_data(0)
        result, _ = calibrate(**data)
        self.assertEqual(list(result["stages"]), [
            "stage1_ellipse_and_inplane_angle",
            "stage2_sdd_and_principal_point",
            "stage3_sod_and_bead_positions",
        ])

    def test_exact_three_beads(self):
        data, truth = demo_data(0)
        data["tracks_px"] = data["tracks_px"][:, [0, 2, 4]]
        data["distances_mm"] = data["distances_mm"][np.ix_([0, 2, 4], [0, 2, 4])]
        result, _ = calibrate(**data)
        self.assert_geometry(result["geometry"], truth)

    def test_invalid_sampling(self):
        for mode in ("duplicate", "reverse", "multiple_turns"):
            with self.subTest(mode=mode):
                data, _ = demo_data(0)
                if mode == "duplicate":
                    data["angles_rad"][1] = data["angles_rad"][0]
                elif mode == "multiple_turns":
                    data["angles_rad"] *= 2
                else:
                    data["angles_rad"][2] = data["angles_rad"][0]
                with self.assertRaises(ValueError):
                    calibrate(**data)

    def test_odd_frames_missing_last_view(self):
        data, truth = demo_data(0)
        data["angles_rad"] = data["angles_rad"][:-1]
        data["tracks_px"] = data["tracks_px"][:-1]
        result, _ = calibrate(**data)
        self.assert_geometry(result["geometry"], truth)
        self.assertNotIn("antipodal_pairing", result["diagnostics"])

    def test_359_degree_scan_without_exact_opposites(self):
        data, truth = demo_data(0)
        beads = np.column_stack([np.full(5, 24.), np.full(5, 8.), np.arange(5) * 6. - 10.])
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                angles = np.deg2rad(np.linspace(0, 359, 359))
                if reverse:
                    angles = 0.4 - angles
                data["angles_rad"] = angles
                data["tracks_px"] = ray_plane_projection(truth, beads, angles, data["pixel_size_mm"])
                result, _ = calibrate(**data)
                self.assert_geometry(result["geometry"], truth, atol=1e-5)
                self.assertLess(result["diagnostics"]["rmse_px"], 1e-8)

    def test_irregular_noisy_angles_and_wrapped_input(self):
        data, truth = demo_data(0)
        beads = np.column_stack([np.full(5, 24.), np.full(5, 8.), np.arange(5) * 6. - 10.])
        rng = np.random.default_rng(47)
        angles = np.deg2rad(np.arange(359) + rng.uniform(-0.2, 0.2, 359)) + 0.8
        data["tracks_px"] = ray_plane_projection(truth, beads, angles, data["pixel_size_mm"])
        data["tracks_px"] += rng.normal(0, 0.1, data["tracks_px"].shape)
        data["angles_rad"] = angles % (2 * np.pi)
        result, _ = calibrate(**data, refine=True)
        self.assertIsNone(result["delta_beta_deg"])
        self.assertLess(result["diagnostics"]["rmse_px"], 0.11)
        self.assertAlmostEqual(result["geometry"]["sdd_mm"], truth.sdd_mm, delta=0.4)

    def test_short_arc_without_any_opposite_views(self):
        data, truth = demo_data(0)
        beads = np.column_stack([np.full(5, 24.), np.full(5, 8.), np.arange(5) * 6. - 10.])
        angles = np.deg2rad(np.linspace(13, 163, 151))
        data["angles_rad"] = angles
        data["tracks_px"] = ray_plane_projection(truth, beads, angles, data["pixel_size_mm"])
        result, _ = calibrate(**data)
        self.assert_geometry(result["geometry"], truth, atol=1e-5)
        self.assertLess(result["diagnostics"]["rmse_px"], 1e-7)
        self.assertTrue(result["diagnostics"]["warnings"])

    def test_general_ellipse_center_and_quadratic_form(self):
        theta = np.linspace(0.2, 4.9, 183)
        angle = 0.37
        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        center = np.array([850., -350.])
        points = np.column_stack([20 * np.cos(theta), 3 * np.sin(theta)]) @ rot.T + center
        fitted_center, q, _ = fit_ellipse(points)
        np.testing.assert_allclose(fitted_center, center, atol=1e-8)
        np.testing.assert_allclose(q, rot @ np.diag([1 / 400, 1 / 9]) @ rot.T, atol=1e-10)

    def test_missing_tracks_and_scale(self):
        data, _ = demo_data(0)
        data["tracks_px"][1, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing"):
            calibrate(**data)
        data, _ = demo_data(0)
        data["distances_mm"][:] = np.nan
        np.fill_diagonal(data["distances_mm"], 0)
        with self.assertRaisesRegex(ValueError, "known positive"):
            calibrate(**data)

    def test_degenerate_axis_midplane_and_equal_heights(self):
        for mode in ("axis", "midplane", "height"):
            with self.subTest(mode=mode):
                data, truth = demo_data(0)
                beads = np.array([[20., 10., -10.], [30., 8., 2.], [25., 6., 15.]])
                if mode == "axis":
                    beads[:, :2] = 0
                elif mode == "midplane":
                    beads[:, 2] = 0
                else:
                    beads[:, 2] = 5
                data["tracks_px"] = project(truth, beads, data["angles_rad"], data["pixel_size_mm"])
                data["distances_mm"] = np.linalg.norm(beads[:, None] - beads[None, :], axis=-1)
                with self.assertRaises(ValueError):
                    calibrate(**data)


if __name__ == "__main__":
    unittest.main()
