"""Run the standard Yang 6+6 calibration directly from a projection RAW.

Edit CONFIG in an IDE and run this file.  No command-line arguments are
required.  Image preprocessing and geometric calibration remain separate.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
for directory in (HERE, PARENT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from yang_pic import (YangConfig, calibrate_stack_geometries, calibrate_view,
                      geometry_to_dict, pose_to_dict)
from yang_standard_preprocess import (detect_components,
                                      initial_index_candidates,
                                      track_standard_stack)


CONFIG = {
    "raw_path": Path(
        r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\calibration"
        r"\double-ring-100mm-spacing100-bead3-cylinder6-6beads-spp10-projection-1024x1024x360-f32.raw"),
    "output_directory": Path("out/cbct_calibration/yang_standard_raw"),
    "views": 360,
    "rows": 1024,
    "cols": 1024,
    "threshold": 5.0,
    "pixel_size_mm": (0.417, 0.417),
    "ring_radius_mm": 50.0,
    "ring_half_spacing_mm": 50.0,
    "image_v_points_down": True,
    # Simulation truth is report-only.  It is never passed to Yang's solver.
    "ground_truth": {
        "offset_u_mm": 2.085,
        "offset_v_mm": 4.170,
        "offset_n_mm": 0.0,
        "tilt_u_deg": 1.0,
        "tilt_v_deg": 2.0,
        "tilt_n_deg": 3.0,
        "phantom_offset_x_mm": 10.0,
        "phantom_offset_y_mm": 15.0,
        "phantom_offset_z_mm": 20.0,
        "phantom_rotation_x_deg": 2.0,
        "phantom_rotation_y_deg": 0.0,
        "phantom_rotation_z_deg": 0.0,
    },
}


def load_raw(config: dict) -> np.memmap:
    path = Path(config["raw_path"])
    expected = (int(config["views"]) * int(config["rows"])
                * int(config["cols"]) * np.dtype("<f4").itemsize)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected:
        raise ValueError(f"RAW size {path.stat().st_size} != expected {expected}")
    return np.memmap(path, dtype="<f4", mode="r",
                     shape=(config["views"], config["rows"], config["cols"]))


def to_yang_coordinates(points: np.ndarray, rows: int,
                        image_v_points_down: bool) -> np.ndarray:
    output = np.asarray(points, dtype=float).copy()
    if image_v_points_down:
        output[..., 1] = (rows - 1) - output[..., 1]
    return output


def choose_initial_indices(image: np.ndarray, config: dict,
                           yang: YangConfig) -> tuple[np.ndarray, dict]:
    components = detect_components(image, config["threshold"])
    scored = []
    failures = []
    for candidate_id, candidate_image in enumerate(initial_index_candidates(components)):
        candidate_yang = to_yang_coordinates(
            candidate_image, config["rows"], config["image_v_points_down"])
        try:
            pose = calibrate_view(candidate_yang, yang)
            scored.append((pose.reprojection_rmse_px, candidate_id,
                           candidate_image, pose))
        except (ValueError, FloatingPointError) as exc:
            failures.append({"candidate": candidate_id, "reason": str(exc)})
    if not scored:
        raise RuntimeError(f"all initial index candidates failed: {failures}")
    score, candidate_id, points, pose = min(scored, key=lambda item: item[0])
    return points, {
        "selected_candidate": int(candidate_id),
        "candidate_rmse_px": float(score),
        "valid_candidate_count": len(scored),
        "failed_candidates": failures,
        "initial_pose": pose_to_dict(pose),
    }


def _save_detection_figure(path: Path, raw: np.ndarray, points: np.ndarray,
                           diagnostics: list[dict]) -> None:
    merged_frames = [item["frame"] for item in diagnostics if item["interpolated"]]
    frame = 118 if 118 in merged_frames else merged_frames[len(merged_frames) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, index, title in (
            (axes[0], 0, "Frame 0: twelve indexed targets"),
            (axes[1], frame, f"Frame {frame}: complete overlap recovered from trajectory")):
        ax.imshow(np.asarray(raw[index]), cmap="gray", vmin=0, vmax=21)
        ax.scatter(points[index, :, 0], points[index, :, 1], s=65,
                   facecolors="none", edgecolors="cyan", linewidths=1.3)
        for target, (u, v) in enumerate(points[index]):
            ax.text(u + 5, v, str(target), color="yellow", fontsize=8)
        ax.set_xlim(300, 800)
        ax.set_ylim(930, 260)
        ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_diagnostic_figure(path: Path, poses: list, diagnostics: list[dict]) -> None:
    frames = np.arange(len(poses))
    merged = np.asarray([item["interpolated"] for item in diagnostics])
    rmse = np.asarray([pose.reprojection_rmse_px for pose in poses])
    sdd = np.asarray([pose.source_detector_distance_mm for pose in poses])
    angles = np.rad2deg(np.asarray([
        [pose.roll_rad, pose.pitch_rad, pose.yaw_rad] for pose in poses]))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             constrained_layout=True)
    axes[0].plot(frames, rmse, color="#0072B2", lw=1.2)
    axes[0].scatter(frames[merged], rmse[merged], color="#D55E00", s=14,
                    label="overlap filled from trajectory")
    axes[0].set_ylabel("reprojection RMSE (px)")
    axes[0].legend(loc="upper right")
    axes[1].plot(frames, sdd, color="#009E73", lw=1.2)
    axes[1].set_ylabel("SDD (mm)")
    for column, label, color in zip(range(3), ("roll", "pitch", "yaw"),
                                    ("#0072B2", "#D55E00", "#CC79A7")):
        axes[2].plot(frames, angles[:, column], label=label, color=color, lw=1.0)
    axes[2].set_ylabel("angle (deg)")
    axes[2].set_xlabel("frame")
    axes[2].legend(ncol=3, loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(config: dict = CONFIG) -> dict:
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw(config)
    yang = YangConfig(
        ring_radius_mm=config["ring_radius_mm"],
        ring_half_spacing_mm=config["ring_half_spacing_mm"],
        pixel_size_mm=tuple(config["pixel_size_mm"]),
        gantry_solver="paper_linear",
    )
    initial, index_info = choose_initial_indices(np.asarray(raw[0]), config, yang)
    image_points, diagnostics = track_standard_stack(
        raw, config["threshold"], initial)
    yang_points = to_yang_coordinates(
        image_points, config["rows"], config["image_v_points_down"])
    np.save(output / "image_points_6plus6.npy", image_points)
    np.save(output / "yang_points_6plus6.npy", yang_points)
    (output / "preprocess_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    poses, geometries = calibrate_stack_geometries(yang_points, yang)
    merged_frames = [item["frame"] for item in diagnostics if item["interpolated"]]
    report = {
        "method": "Yang_2017_standard_6plus6_cylinder_markers",
        "raw_path": str(config["raw_path"]),
        "views": len(poses),
        "threshold": config["threshold"],
        "index_initialization": index_info,
        "complete_overlap_frame_count": len(merged_frames),
        "complete_overlap_frames": merged_frames,
        "overlap_policy": "identify hidden IDs, then periodic cubic trajectory interpolation",
        "ground_truth_report_only": config.get("ground_truth", {}),
        "reprojection_rmse_mean_px": float(np.mean([p.reprojection_rmse_px for p in poses])),
        "reprojection_rmse_max_px": float(np.max([p.reprojection_rmse_px for p in poses])),
        "sdd_mean_mm": float(np.mean([p.source_detector_distance_mm for p in poses])),
        "sdd_std_mm": float(np.std([p.source_detector_distance_mm for p in poses])),
        "poses": [pose_to_dict(pose) for pose in poses],
        "reconstruction_geometries": [geometry_to_dict(g) for g in geometries],
    }
    np.asarray([item.projection_matrix for item in geometries], dtype="<f8").tofile(
        output / "projection_matrices_360x3x4.f64")
    (output / "yang_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_detection_figure(output / "target_detection_and_overlap.png",
                           raw, image_points, diagnostics)
    _save_diagnostic_figure(output / "yang_geometry_diagnostics.png",
                            poses, diagnostics)
    summary = {key: report[key] for key in (
        "views", "complete_overlap_frame_count", "reprojection_rmse_mean_px",
        "reprojection_rmse_max_px", "sdd_mean_mm", "sdd_std_mm")}
    print(json.dumps(summary, indent=2))
    return report


if __name__ == "__main__":
    run()
