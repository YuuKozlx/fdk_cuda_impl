import unittest

import numpy as np

from .tracker import Component, _assign_components


class YangOverlapStatusTests(unittest.TestCase):
    def test_multiple_merged_blobs_are_not_usable(self):
        predicted = np.column_stack([np.arange(12) * 10.0,
                                     np.full(12, 50.0)])
        centers = [p.copy() for p in predicted]
        centers[2] = (predicted[2] + predicted[3]) / 2
        del centers[3]
        centers[7] = (predicted[8] + predicted[9]) / 2
        del centers[8]
        components = [Component(np.asarray([p]), np.ones(1),
                                np.asarray(p), 40) for p in centers]
        points, usable, merged = _assign_components(components, predicted)
        self.assertEqual(len(merged), 4)
        self.assertEqual(np.count_nonzero(usable), 8)
        np.testing.assert_allclose(points[~usable], predicted[~usable])


if __name__ == "__main__":
    unittest.main()
