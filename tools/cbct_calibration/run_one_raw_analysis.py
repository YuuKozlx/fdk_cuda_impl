from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from cho_calibration import ChoConfig, calibrate_view
from cho_preprocess import detect_frame, initialize_indices
from cho_system_geometry import analyze_poses, _jsonable


def main(raw_name: str, points_name: str, report_name: str):
    raw_path = Path(raw_name)
    shape = (1024, 1024)
    views = 360
    frame_bytes = shape[0] * shape[1] * 4
    config = ChoConfig(
        ring_radius_mm=50.0, ring_half_spacing_mm=50.0, beads_per_ring=12,
        pixel_size_mm=(0.417, 0.417), robust_scale_px=0.15, max_nfev=800,
    )

    def read_frame(stream, index):
        stream.seek(index * frame_bytes)
        data = np.fromfile(stream, dtype=np.float32, count=shape[0] * shape[1])
        return data.reshape(shape)

    points = []
    frame_ids = []
    rejected = []
    with raw_path.open("rb") as stream:
        # Use unsplit components when all 24 beads are already separable. Fall
        # back to standalone peak/Voronoi splitting only for merged first views.
        first_image = read_frame(stream, 0)
        try:
            first = detect_frame(first_image, 5.0, split_merged=False)
        except ValueError:
            first = detect_frame(first_image, 5.0, split_merged=True)
        points.append(initialize_indices(first, config))
        frame_ids.append(0)
        for index in range(1, views):
            if len(points) < 2:
                predicted = points[-1]
            else:
                gap = max(frame_ids[-1] - frame_ids[-2], 1)
                predicted = points[-1] + (points[-1] - points[-2]) * (index - frame_ids[-1]) / gap
            try:
                assigned = detect_frame(read_frame(stream, index), 5.0,
                                        split_merged=True, predicted=predicted)
                error = float(np.max(np.linalg.norm(assigned - predicted, axis=1)))
            except Exception as exc:
                rejected.append({"frame": index, "reason": str(exc)})
                continue
            if error > 40.0:
                rejected.append({"frame": index, "reason": f"prediction error {error}"})
                continue
            points.append(assigned)
            frame_ids.append(index)

    points_array = np.asarray(points)
    np.save(points_name, points_array)
    poses = [calibrate_view(frame, config) for frame in points_array]
    result = analyze_poses(poses, shape, config.pixel_size_mm, nominal_sid_mm=440.0)
    report = _jsonable(result)
    report["frame_ids"] = frame_ids
    report["rejected"] = rejected
    Path(report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = dict(report)
    summary.pop("per_view", None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
