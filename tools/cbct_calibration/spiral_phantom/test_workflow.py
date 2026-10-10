import unittest
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from ..identifiable_phantom.joint_fit import project
from ..identifiable_phantom.workflow import calibrate_indexed
from .phantom import spiral_points
from .tracker import (SpiralTrackerConfig, initialize_indices,
                      _extract_from_predictions, _track_direction)


def run_repository_overlap_test():
    """Read RAW afresh, compare both fits, and export this run for C++ tests."""
    from .api_example import run_repository_example
    report = run_repository_example()
    output = Path(report["input"]["output_directory"])
    # These arrays were written by the RAW run directly above, not a reused cache.
    points = np.load(output / "points.npy")
    merged = np.load(output / "merged.npy")
    usable = np.load(output / "usable.npy")
    local = np.asarray(report["phantom"]["points_mm"])
    baseline = calibrate_indexed(points, local_points=local)
    with (output / "indexed_cpp_test.bin").open("wb") as stream:
        np.asarray(points.shape[:2], dtype="<i4").tofile(stream)
        np.asarray(local, dtype="<f8").tofile(stream)
        np.asarray(points, dtype="<f8").tofile(stream)
        np.asarray(usable, dtype="u1").tofile(stream)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].imshow(merged.T, origin="lower", aspect="auto", cmap="Reds",
                      interpolation="nearest", vmin=0, vmax=1)
    axes[0, 0].set(title="Excluded merged observations", xlabel="view", ylabel="bead ID (24 = marker)")
    axes[0, 1].plot(usable.sum(axis=1), label="usable", color="tab:green")
    axes[0, 1].plot(np.isfinite(points).all(axis=2).sum(axis=1), label="tracked", ls="--")
    axes[0, 1].set(title="Tracks remain alive through merging", xlabel="view", ylabel="target count")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=.2)
    for result, label, color in [(baseline, "all centroids", "tab:orange"),
                                  (report, "merged excluded", "tab:blue")]:
        cameras = result["dlt_cameras"]
        axes[1, 0].plot([c["reprojection_rmse_px"] for c in cameras], label=label, color=color)
        axes[1, 1].plot([c["sdd_mm"] for c in cameras], label=label, color=color, alpha=.8)
    axes[1, 0].set(title="Physical-camera DLT residual", xlabel="view", ylabel="RMSE / px")
    axes[1, 1].set(title="Per-view SDD (no smoothing)", xlabel="view", ylabel="mm")
    for axis in axes[1]: axis.legend(); axis.grid(alpha=.2)
    before = baseline["joint_fit"]["diagnostics"]["rmse_px"]
    after = report["joint_fit"]["diagnostics"]["rmse_px"]
    figure.suptitle(f"Joint RMSE: {before:.4f} -> {after:.4f} px; {merged.sum()} observations excluded")
    figure.savefig(output / "overlap_comparison.png", dpi=170); plt.close(figure)
    comparison = {"all_centroids": baseline["joint_fit"], "merged_excluded": report["joint_fit"]}
    (output / "overlap_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


class SpiralNumberingTest(unittest.TestCase):
    def test_retired_seed_does_not_steal_foreground(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[9:12, 9:12] = 20.0
        config = SpiralTrackerConfig(Path("."), Path("."), views=1)
        predicted = np.array([[np.nan, np.nan], [10.0, 10.0]])
        measured, valid, _ = _extract_from_predictions(image, predicted, config)
        np.testing.assert_array_equal(valid, [False, True])
        np.testing.assert_allclose(measured[1], [10.0, 10.0])
        self.assertTrue(np.isnan(measured[0]).all())

    def test_touching_targets_are_marked_merged(self):
        yy, xx = np.indices((40, 40))
        image = np.zeros((40, 40), dtype=np.float32)
        image[(xx - 19) ** 2 + (yy - 20) ** 2 <= 4 ** 2] = 20.0
        image[(xx - 23) ** 2 + (yy - 20) ** 2 <= 4 ** 2] = 20.0
        config = SpiralTrackerConfig(Path("."), Path("."), views=1,
                                     gate_radius_px=10.0)
        _, valid, stats = _extract_from_predictions(
            image, np.array([[19.0, 20.0], [23.0, 20.0]]), config)
        np.testing.assert_array_equal(valid, [True, True])
        self.assertEqual(stats["merged_ids"], [0, 1])

    def test_merged_frame_does_not_retire_tracks(self):
        raw = np.zeros((3, 40, 40), dtype=np.float32)
        raw[:, 19:22, 9:12] = 20.0
        raw[:, 19:22, 19:22] = 20.0
        raw[1, 20, 12:19] = 20.0
        seeds = np.array([[10.0, 20.0], [20.0, 20.0]])
        config = SpiralTrackerConfig(Path("."), Path("."), views=3,
                                     rows=40, cols=40, gate_radius_px=10.0)
        _, observed, merged, _ = _track_direction(raw, 0, seeds, 1, config)
        np.testing.assert_array_equal(observed, np.ones((3, 2), dtype=bool))
        np.testing.assert_array_equal(merged, [[False, False], [True, True],
                                              [False, False]])
        self.assertTrue((observed & ~merged)[2].all())

    def test_known_helix_layout(self):
        points = spiral_points()
        self.assertEqual(points.shape, (25, 3))
        np.testing.assert_allclose(points[0], [50.0, 0.0, -57.5])
        np.testing.assert_allclose(points[-1], [50.0, 0.0, -67.5])

    def test_joint_model_projects_non_coplanar_helix(self):
        state = np.array([440., 770., 2.085, 4.17,
                          np.deg2rad(1), np.deg2rad(2), np.deg2rad(3),
                          0., 0., 0., 10., 15., 20.])
        result = project(state, spiral_points(), np.linspace(0, 2*np.pi, 8, endpoint=False),
                         np.array([.417, .417]), np.array([511.5, 511.5]))
        self.assertEqual(result.shape, (8, 25, 2))
        self.assertTrue(np.isfinite(result).all())

    def test_anchor_numbering_uses_marker_and_helix_geometry(self):
        state = np.array([440., 770., 2.085, 4.17,
                          np.deg2rad(1), np.deg2rad(2), np.deg2rad(3),
                          np.deg2rad(1), np.deg2rad(2), np.deg2rad(3),
                          10., 15., 20.])
        expected = project(state, spiral_points(), np.array([0.]),
                           np.array([.417, .417]),
                           np.array([511.5, 511.5]))[0]
        image = np.zeros((1024, 1024), dtype=np.float32)
        yy, xx = np.ogrid[:1024, :1024]
        for index, (u, v) in enumerate(expected):
            radius = 6.0 if index == 24 else 3.5
            image[(xx - u) ** 2 + (yy - v) ** 2 <= radius ** 2] = 20.0
        config = SpiralTrackerConfig(Path("."), Path("."), views=1,
                                     max_dlt_rmse_px=.5)
        measured, report = initialize_indices(image, config)
        self.assertLess(report["selected_physical_camera_rmse_px"], .5)
        np.testing.assert_allclose(measured, expected, atol=.5)


if __name__ == "__main__":
    unittest.main()
