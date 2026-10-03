import unittest

import numpy as np

from .detector import (
    DetectorConvention,
    FDK_TEST_CONVENTION,
    convert_detector_points,
    convert_projection_matrix,
    convert_pose_pixel_fields,
    detector_homography,
    detector_frame_basis,
    detector_frame_transform,
    convert_detector_frame_axes,
    phantom_frame_homography,
    convert_projection_world_and_detector,
)
from ..math.rigid import RigidTransform
from ..yang.coordinate_adapter import (YANG_PAPER_CONVENTION,
                                       YANG_PHANTOM_TO_FDK,
                                       yang_projection_to_fdk)


class DetectorConventionTests(unittest.TestCase):
    def test_detector_conventions_are_right_handed(self):
        for convention in (FDK_TEST_CONVENTION, YANG_PAPER_CONVENTION):
            basis = detector_frame_basis(convention)
            self.assertAlmostEqual(np.linalg.det(basis), 1.0, places=7)
            np.testing.assert_allclose(np.cross(basis[:, 0], basis[:, 1]), basis[:, 2])

    def test_yang_to_fdk_is_proper_3d_rotation(self):
        transform = detector_frame_transform(YANG_PAPER_CONVENTION,
                                             FDK_TEST_CONVENTION)
        self.assertAlmostEqual(np.linalg.det(transform), 1.0, places=7)
        np.testing.assert_allclose(transform.T @ transform, np.eye(3), atol=1e-12)
        axes = np.asarray([YANG_PAPER_CONVENTION.u_axis,
                           YANG_PAPER_CONVENTION.v_axis,
                           YANG_PAPER_CONVENTION.n_axis])
        converted = convert_detector_frame_axes(axes,
                                                YANG_PAPER_CONVENTION,
                                                FDK_TEST_CONVENTION)
        expected = np.asarray([FDK_TEST_CONVENTION.u_axis,
                               FDK_TEST_CONVENTION.v_axis,
                               FDK_TEST_CONVENTION.n_axis])
        np.testing.assert_allclose(converted, expected)

    def test_yang_phantom_transform_is_proper_rotation(self):
        self.assertAlmostEqual(np.linalg.det(YANG_PHANTOM_TO_FDK), 1.0, places=7)
        np.testing.assert_allclose(
            YANG_PHANTOM_TO_FDK.T @ YANG_PHANTOM_TO_FDK, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            RigidTransform(YANG_PHANTOM_TO_FDK).apply(
                np.array([[1., 2., 3.]])),
            [[1., 3., -2.]])
    def test_yang_v_up_to_fdk_row_down_and_back(self):
        image_points = np.array([[0.0, 0.0], [7.25, 3.5], [9.0, 5.0]])
        yang_points = convert_detector_points(
            image_points, (6, 10), FDK_TEST_CONVENTION, YANG_PAPER_CONVENTION)
        np.testing.assert_allclose(yang_points[:, 0], image_points[:, 0])
        np.testing.assert_allclose(yang_points[:, 1], 5.0 - image_points[:, 1])
        restored = convert_detector_points(
            yang_points, (6, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        np.testing.assert_allclose(restored, image_points)

    def test_same_convention_is_identity_copy(self):
        points = np.array([[1.5, 2.5]])
        converted = convert_detector_points(
            points, (8, 12), FDK_TEST_CONVENTION, FDK_TEST_CONVENTION)
        np.testing.assert_array_equal(converted, points)
        self.assertIsNot(converted, points)

    def test_axis_reflection_uses_column_count(self):
        points = np.array([[1.0, 2.0]])
        flipped = convert_detector_points(
            points, (8, 12), FDK_TEST_CONVENTION,
            DetectorConvention("u-left, v-down", u_sign=-1))
        np.testing.assert_allclose(flipped, [[10.0, 2.0]])

    def test_homography_matches_point_conversion(self):
        points = np.array([[3.2, 1.5], [0.0, 5.0]])
        h = detector_homography((6, 10), YANG_PAPER_CONVENTION,
                                FDK_TEST_CONVENTION)
        homogeneous = np.column_stack((points, np.ones(len(points))))
        mapped = (h @ homogeneous.T).T[:, :2]
        expected = convert_detector_points(
            points, (6, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        np.testing.assert_allclose(mapped, expected)

    def test_projection_matrix_reflection_preserves_rays(self):
        p = np.array([[2.0, 0.0, 1.0, 3.0],
                      [0.0, 4.0, 2.0, 5.0],
                      [0.0, 0.0, 1.0, 7.0]])
        point = np.array([0.4, -0.2, 1.3, 1.0])
        q_paper = p @ point
        q_paper = q_paper[:2] / q_paper[2]
        converted = convert_projection_matrix(
            p, (8, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        q_fdk = converted @ point
        q_fdk = q_fdk[:2] / q_fdk[2]
        expected = convert_detector_points(
            q_paper, (8, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        np.testing.assert_allclose(q_fdk, expected)

    def test_yang_projection_conversion_transforms_world_and_image_once(self):
        p = np.array([[2.0, 0.0, 1.0, 3.0],
                      [0.0, 4.0, 2.0, 5.0],
                      [0.0, 0.0, 1.0, 7.0]])
        yang_point = np.array([0.4, -0.2, 1.3, 1.0])
        q_yang = p @ yang_point
        q_yang = q_yang[:2] / q_yang[2]
        fdk_point = np.r_[RigidTransform(YANG_PHANTOM_TO_FDK).apply(
            yang_point[:3][None, :])[0], 1.0]
        converted = yang_projection_to_fdk(p, (8, 10))
        q_fdk = converted @ fdk_point
        q_fdk = q_fdk[:2] / q_fdk[2]
        expected = convert_detector_points(
            q_yang, (8, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        np.testing.assert_allclose(q_fdk, expected)
        np.testing.assert_allclose(
            converted,
            detector_homography((8, 10), YANG_PAPER_CONVENTION,
                                FDK_TEST_CONVENTION) @ p @
            phantom_frame_homography(YANG_PHANTOM_TO_FDK),
            atol=1e-10)

    def test_arbitrary_right_handed_rigid_frame_preserves_projection(self):
        angle = 0.37
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        translation = np.array([12.0, -7.0, 3.5])
        projection = np.array([
            [850.0, 0.0, 511.5, 1300.0],
            [0.0, 840.0, 511.5, -400.0],
            [0.0, 0.0, 1.0, 4.0],
        ])
        source_points = np.array([
            [1.0, 2.0, 5.0, 1.0],
            [-3.0, 1.5, 8.0, 1.0],
        ])
        target_points = source_points.copy()
        target_points[:, :3] = (
            source_points[:, :3] @ rotation.T + translation)
        converted = convert_projection_world_and_detector(
            projection, (1024, 1024),
            phantom_source_to_target=rotation,
            phantom_translation=translation,
            source_detector=FDK_TEST_CONVENTION,
            target_detector=FDK_TEST_CONVENTION)
        source_pixels = (projection @ source_points.T).T
        target_pixels = (converted @ target_points.T).T
        source_pixels = source_pixels[:, :2] / source_pixels[:, 2:3]
        target_pixels = target_pixels[:, :2] / target_pixels[:, 2:3]
        np.testing.assert_allclose(target_pixels, source_pixels, atol=1e-10)

    def test_pose_converter_only_changes_pixel_fields(self):
        source = {"o_px": [2.0, 1.0], "principal_point_px": [4.0, 3.0],
                  "source_i_mm": [1.0, 2.0, 3.0]}
        converted = convert_pose_pixel_fields(
            source, (8, 10), YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        self.assertEqual(converted["o_px"], [2.0, 6.0])
        self.assertEqual(converted["principal_point_px"], [4.0, 4.0])
        self.assertEqual(converted["source_i_mm"], source["source_i_mm"])


if __name__ == "__main__":
    unittest.main()
