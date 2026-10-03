import unittest

import numpy as np

from .dlt_to_conevec import (conevec_normal, conevec_to_dlt, dlt_to_conevec,
                             ray_from_conevec, reindex_conevec)


class DltConeVectorTests(unittest.TestCase):
    def test_v_reindex_updates_complete_right_handed_plane(self):
        p = np.array([[1000., 0., 511.5, 0.], [0., 1000., 511.5, 0.],
                      [0., 0., 1., 440.]])
        original = dlt_to_conevec(p, sdd_mm=770.0,
                                  point_toward=np.zeros(3))
        converted = reindex_conevec(original, (1024, 1024), flip_v=True)
        np.testing.assert_allclose(converted.det_s,
                                   original.det_s + 1023 * original.r_v)
        np.testing.assert_allclose(converted.r_v, -original.r_v)
        np.testing.assert_allclose(conevec_normal(converted),
                                   -conevec_normal(original))
        for u, v in ((0., 0.), (511.5, 400.), (1023., 1023.)):
            _, a = ray_from_conevec(original, u, 1023.0 - v)
            _, b = ray_from_conevec(converted, u, v)
            np.testing.assert_allclose(a, b, atol=1e-12)
    def test_round_trip_preserves_projection(self):
        p = np.array([[900., 12., 510., -1200.],
                      [5., 880., 500., 700.],
                      [.01, -.02, 1., -430.]])
        cone = dlt_to_conevec(p, sdd_mm=770.0, point_toward=np.zeros(3))
        recovered = conevec_to_dlt(cone)
        points = np.array([[0., 0., 0., 1.], [20., -10., 30., 1.],
                           [-15., 25., -5., 1.]])
        a, b = points @ p.T, points @ recovered.T
        np.testing.assert_allclose(a[:, :2] / a[:, 2, None],
                                   b[:, :2] / b[:, 2, None], atol=1e-10)

    def test_each_pixel_ray_is_collinear_with_dlt_backprojection(self):
        p = np.array([[1000., 0., 511.5, 0.], [0., 1000., 511.5, 0.],
                      [0., 0., 1., 440.]])
        cone = dlt_to_conevec(p, sdd_mm=770.0, point_toward=np.zeros(3))
        inverse = np.linalg.inv(p[:, :3])
        for u, v in ((0., 0.), (511.5, 511.5), (1023., 700.)):
            _, direction = ray_from_conevec(cone, u, v)
            expected = inverse @ np.array([u, v, 1.])
            expected /= np.linalg.norm(expected)
            self.assertAlmostEqual(abs(float(np.dot(direction, expected))), 1.0, 12)


if __name__ == "__main__":
    unittest.main()
