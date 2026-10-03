"""配置式 Yang 2017 PIC/extended 运行入口。

在 IDE 中只修改顶部 CONFIG，然后运行本文件。输入必须是已经完成
亚像素检测和编号的 ``[views, 12, 2]`` 数组，顺序为
``E1..E6,F1..F6``。本文件不把普通单排钢珠 RAW 当作 Yang 模体。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pic import (YangConfig, calibrate_stack_geometries,
                       extended_calibrate_stack, extended_result_to_dict,
                       geometry_to_dict, pose_to_dict)
from .workflow import _canonical_geometry_dict
from ..coordinates import (FDK_TEST_CONVENTION, convert_detector_points,
                           convert_pose_pixel_fields)
from .coordinate_adapter import YANG_PAPER_CONVENTION


CONFIG = {
    "points_path": Path("out/cbct_calibration/yang/points_12beads.npy"),
    "output_json": Path("out/cbct_calibration/yang/report.json"),
    "ring_radius_mm": 30.0,
    "ring_half_spacing_mm": 25.0,
    "pixel_size_mm": (0.254, 0.317),
    "image_shape": (1024, 1024),
    "input_coordinate_convention": "fdk-test",
    "run_extended": False,
    # Yang Section II.D assumes fixed source and fixed detector.  Keep False
    # for a moving gantry unless that physical constraint is intentionally
    # being imposed.
    "source_constraint_weight": 12.0,
    "extended_max_nfev": 200,
}


def load_points(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        key = "points_px" if "points_px" in loaded else loaded.files[0]
        loaded = loaded[key]
    points = np.asarray(loaded, dtype=float)
    if points.ndim == 2:
        points = points[None, ...]
    if points.ndim != 3 or points.shape[1:] != (12, 2):
        raise ValueError("points_path must contain [views,12,2] detector centroids")
    return points


def run(config: dict = CONFIG) -> dict:
    points = load_points(Path(config["points_path"]))
    image_shape = tuple(config["image_shape"])
    if config.get("input_coordinate_convention", "fdk-test") == "fdk-test":
        points_paper = convert_detector_points(
            points, image_shape, FDK_TEST_CONVENTION, YANG_PAPER_CONVENTION)
    elif config["input_coordinate_convention"] == "yang-paper":
        points_paper = points
    else:
        raise ValueError("input_coordinate_convention must be fdk-test or yang-paper")
    yang = YangConfig(
        ring_radius_mm=float(config["ring_radius_mm"]),
        ring_half_spacing_mm=float(config["ring_half_spacing_mm"]),
        pixel_size_mm=tuple(config["pixel_size_mm"]),
        gantry_solver="paper_linear",
    )
    poses, geometries = calibrate_stack_geometries(points_paper, yang)
    paper_geometries = [geometry_to_dict(g) for g in geometries]
    fdk_geometries = [
        _canonical_geometry_dict(geometry, pose, image_shape)
        for pose, geometry in zip(poses, geometries)
    ]
    paper_poses = [pose_to_dict(pose) for pose in poses]
    fdk_poses = [convert_pose_pixel_fields(
        item, image_shape, YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
                 for item in paper_poses]
    report = {
        "method": "Yang_2017_PIC",
        "views": int(len(poses)),
        "point_order": "E1..E6,F1..F6",
        "image_coordinate_convention": FDK_TEST_CONVENTION.name,
        "paper_coordinate_convention": YANG_PAPER_CONVENTION.name,
        "phantom_coordinate_convention": "fdk-test canonical for reconstruction_geometries",
        "pic_poses_paper_coordinates": paper_poses,
        "pic_poses": fdk_poses,
        "reconstruction_geometries_paper_coordinates": paper_geometries,
        "reconstruction_geometries": fdk_geometries,
    }
    if bool(config["run_extended"]):
        extended = extended_calibrate_stack(
            points_paper, yang,
            source_constraint_weight=float(config["source_constraint_weight"]),
            max_nfev=int(config["extended_max_nfev"]),
            initial_poses=poses,
        )
        report["extended"] = extended_result_to_dict(extended)
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(f"Yang calibration complete: {result['views']} views")
