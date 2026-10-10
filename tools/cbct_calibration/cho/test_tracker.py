import unittest

import numpy as np

from .tracker import detect_frame_with_status


def component(center, area=20):
    x, y = np.rint(center).astype(int)
    xs = np.arange(x, x + area)
    ys = np.full(area, y)
    return xs, ys, np.ones(area), np.asarray(center, dtype=float)


class ChoOverlapStatusTests(unittest.TestCase):
    def test_multiple_merged_blobs_are_not_usable(self):
        predicted = np.column_stack([np.arange(12) * 10.0,
                                     np.full(12, 50.0)])
        components = [component(p) for p in predicted]
        components[2] = component((predicted[2] + predicted[3]) / 2, 40)
        del components[3]
        components[7] = component((predicted[8] + predicted[9]) / 2, 40)
        del components[8]
        points, observed, merged, usable = detect_frame_with_status(
            np.zeros((128, 128)), predicted, components, 5.0, 12)
        self.assertEqual(np.count_nonzero(merged), 4)
        self.assertEqual(np.count_nonzero(usable), 8)
        self.assertTrue(np.all(observed))
        np.testing.assert_allclose(points[merged], predicted[merged])


if __name__ == "__main__":
    unittest.main()
