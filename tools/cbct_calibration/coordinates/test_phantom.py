import unittest

import numpy as np

from ..math.rigid import RigidTransform
from .phantom import (PhantomFrameTransform, PointCorrespondence,
                      transform_indexed_model_points)


class PhantomCoordinateTests(unittest.TestCase):
    def test_base_rigid_homogeneous_matrix_matches_apply(self):
        transform = RigidTransform(np.eye(3), np.array([2.0, 3.0, -1.0]))
        points = np.array([[1.0, 4.0, 2.0]])
        homogeneous = np.column_stack([points, np.ones(len(points))])
        actual = (transform.homogeneous_matrix() @ homogeneous.T).T[:, :3]
        np.testing.assert_allclose(actual, transform.apply(points))

    def test_rigid_transform_inverse_round_trip(self):
        angle = 0.41
        rotation = np.array([
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ])
        transform = PhantomFrameTransform(
            rotation, np.array([4.0, -7.0, 2.0]), "paper", "scanner")
        points = np.array([[1.0, 2.0, 3.0], [-5.0, 0.5, 8.0]])
        np.testing.assert_allclose(
            transform.inverse().apply(transform.apply(points)), points,
            atol=1.0e-12)

    def test_correspondence_changes_identity_not_coordinate_frame(self):
        points = np.array([[1.0, 0.0, 0.0],
                           [0.0, 2.0, 0.0],
                           [0.0, 0.0, 3.0]])
        rotation = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        transform = PhantomFrameTransform(rotation)
        correspondence = PointCorrespondence((2, 0, 1), "test permutation")
        actual = transform_indexed_model_points(
            points, transform, correspondence)
        expected = transform.apply(points)[[2, 0, 1]]
        np.testing.assert_allclose(actual, expected)

    def test_reflection_is_not_a_rigid_right_handed_frame_change(self):
        with self.assertRaises(ValueError):
            PhantomFrameTransform(np.diag([1.0, 1.0, -1.0]))


if __name__ == "__main__":
    unittest.main()
