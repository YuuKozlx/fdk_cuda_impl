import unittest

import numpy as np

from .pic import (YangConfig, calibrate_stack, calibrate_view,
                       estimate_o, extended_calibrate_stack, fit_ellipse,
                       geometry_from_pose, project_with_geometry)


def independent_projection(theta, phi, eta, t, pitch=(.254,.317),
                           beads_per_ring=6):
    """World rays intersect a detector plane; no production projection code."""
    source = np.array([0.,30.,1500.])
    center = np.array([0.,10.,500.])
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    eu = np.array([cp, st*sp, -ct*sp])
    ev = np.array([0.,ct,st])
    normal = np.cross(eu,ev)
    u = np.cos(eta)*eu + np.sin(eta)*ev
    v = -np.sin(eta)*eu + np.cos(eta)*ev
    a = -t - np.arange(beads_per_ring) * 2.0 * np.pi / beads_per_ring
    beads = np.array([[30*np.cos(x), center[1]+dy, center[2]+30*np.sin(x)]
                      for dy in (-25.,25.) for x in a])
    rays = beads-source
    hits = source - (source@normal)/(rays@normal)[:,None]*rays
    origin = np.array([511.5,487.25])
    obs = np.column_stack([hits@u,hits@v])/pitch+origin
    principal = origin + np.array([source@u,source@v])/pitch
    return obs, source, center, abs(source@normal), principal


class YangPicPrimitiveTests(unittest.TestCase):
    def test_full_staged_recovery_independent_rays(self):
        for degrees in [(0,0,0), (0,-12,8), (.001,3,-4),
                        (9,-7,5), (-30,25,5), (40,-35,10)]:
            with self.subTest(degrees=degrees):
                theta,phi,eta = np.deg2rad(degrees)
                obs,s,w,sdd,principal = independent_projection(theta,phi,eta,.4)
                q = calibrate_view(obs,YangConfig(30,25,(.254,.317)))
                np.testing.assert_allclose([q.roll_rad,q.pitch_rad,q.yaw_rad],
                                           [theta,phi,eta],atol=1e-6)
                np.testing.assert_allclose(q.source_i_mm,s,atol=.002)
                np.testing.assert_allclose(q.phantom_center_i_mm,w,atol=.002)
                self.assertAlmostEqual(q.source_detector_distance_mm,sdd,delta=.002)
                np.testing.assert_allclose(q.principal_point_px,principal,atol=.01)
                self.assertLess(q.reprojection_rmse_px,.001)

    def test_projection_matrix_matches_calibrated_view(self):
        theta, phi, eta = np.deg2rad((9., -7., 5.))
        obs, _, _, _, _ = independent_projection(theta, phi, eta, .4)
        q = calibrate_view(obs, YangConfig(30, 25, (.254, .317)))
        g = geometry_from_pose(q)
        # Recreate bead coordinates in the object-fixed phantom frame.
        beta = np.arange(6)*np.pi/3
        points_p = np.array([[30*np.cos(x), dy, 30*np.sin(x)]
                             for dy in (-25.,25.) for x in beta])
        predicted = project_with_geometry(points_p, g)
        np.testing.assert_allclose(predicted, obs, atol=1e-5)

    def test_missing_bead_rejected(self):
        obs,*_ = independent_projection(.1,.1,.1,.4)
        obs[3] = np.nan
        with self.assertRaises(ValueError):
            calibrate_view(obs,YangConfig(30,25))

    def test_noisy_centroids(self):
        obs,*_ = independent_projection(.15,-.12,.08,.4)
        obs += np.random.default_rng(42).normal(0,.01,obs.shape)
        q = calibrate_view(obs,YangConfig(30,25,(.254,.317)))
        self.assertLess(q.reprojection_rmse_px,.05)

    def test_twelve_beads_per_ring(self):
        angles = np.deg2rad((9.0, -7.0, 5.0))
        obs, source, center, sdd, principal = independent_projection(
            *angles, 0.4, beads_per_ring=12)
        pose = calibrate_view(
            obs, YangConfig(30, 25, (.254, .317), beads_per_ring=12))
        np.testing.assert_allclose(
            [pose.roll_rad, pose.pitch_rad, pose.yaw_rad], angles, atol=1e-6)
        np.testing.assert_allclose(pose.source_i_mm, source, atol=.002)
        np.testing.assert_allclose(pose.phantom_center_i_mm, center, atol=.002)
        self.assertAlmostEqual(pose.source_detector_distance_mm, sdd, delta=.002)
        np.testing.assert_allclose(pose.principal_point_px, principal, atol=.01)
        self.assertLess(pose.reprojection_rmse_px, .001)

    def test_zero_roll_strict_json(self):
        import json
        from .pic import pose_to_dict
        obs,*_ = independent_projection(0,0,0,.4)
        pose = calibrate_view(obs,YangConfig(30,25,(.254,.317)))
        encoded = json.dumps(pose_to_dict(pose),allow_nan=False)
        self.assertEqual(json.loads(encoded)['d_px'],[None,None])
    def test_general_ellipse_fit(self):
        angle = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        points = np.column_stack([
            3.0 + 20.0 * np.cos(angle) * np.cos(0.3)
            - 8.0 * np.sin(angle) * np.sin(0.3),
            -2.0 + 20.0 * np.cos(angle) * np.sin(0.3)
            + 8.0 * np.sin(angle) * np.cos(0.3),
        ])
        ellipse = fit_ellipse(points)
        np.testing.assert_allclose(ellipse.center_mm, [3.0, -2.0], atol=1e-9)
        np.testing.assert_allclose(ellipse.value(points), 0.0, atol=1e-9)

    def test_crossed_lines_find_projection_of_phantom_center(self):
        radius, half_spacing = 30.0, 25.0
        source = np.array([0.0, 0.0, 900.0])
        angles = np.arange(6) * (2.0 * np.pi / 6.0)
        points = []
        for y in (-half_spacing, half_spacing):
            for angle in angles:
                bead = np.array([radius * np.cos(angle), y,
                                 radius * np.sin(angle)])
                distance = -source[2] / (bead[2] - source[2])
                points.append((source + distance * (bead - source))[:2])
        o, diagnostics = estimate_o(np.asarray(points))
        np.testing.assert_allclose(o, [0.0, 0.0], atol=1e-9)
        self.assertLess(np.max(diagnostics["cross_intersection_scatter_px"]), 1e-9)

    def test_paper_linear_gantry_solver(self):
        obs, *_ = independent_projection(np.deg2rad(9), np.deg2rad(-7),
                                         np.deg2rad(5), .4)
        pose = calibrate_view(obs, YangConfig(30, 25, (.254, .317)))
        self.assertEqual(pose.diagnostics["gantry"]["method"],
                         "yang_eq26_linear_sin_cos")
        self.assertLess(pose.reprojection_rmse_px, 1.0e-3)

    def test_extended_fixed_source_constraint(self):
        rng = np.random.default_rng(7)
        frames = []
        for t in np.linspace(-1.0, 1.0, 6):
            obs, *_ = independent_projection(np.deg2rad(4), np.deg2rad(-3),
                                              np.deg2rad(2), t)
            frames.append(obs + rng.normal(0.0, 0.01, obs.shape))
        frames = np.asarray(frames)
        config = YangConfig(30, 25, (.254, .317))
        initial = calibrate_stack(frames, config)
        result = extended_calibrate_stack(
            frames, config, initial_poses=initial, max_nfev=300)
        self.assertLess(result.reprojection_rmse_after_px,
                        result.reprojection_rmse_before_px)
        self.assertLess(result.reprojection_rmse_after_px, 0.02)
        self.assertLess(np.linalg.norm(result.source_grid_std_after_mm),
                        np.linalg.norm(result.source_grid_std_before_mm))


if __name__ == "__main__":
    unittest.main()
