"""Cho 方法的 RAW 直接运行示例。

顶部 CONFIG 是唯一需要修改的地方。流程严格使用 Cho 的 24 点双环模型：

RAW -> 亚像素质心与编号 -> 每帧 DLT/物理分解 -> 共享 SDD/主点联合拟合
    -> 源点轨迹和探测器姿态分析。

这里的像素归属允许粘连区域和额外小连通域存在；它不使用仿真器的真实
offset、tilt 或模体姿态作为分割种子。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

COMMON_DIRECTORY = Path(__file__).resolve().parent.parent
if str(COMMON_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(COMMON_DIRECTORY))

from cho_calibration import ChoConfig, calibrate_stack_joint, calibrate_view
from cho_preprocess import initialize_indices
from cho_system_geometry import _jsonable, analyze_poses


@dataclass(frozen=True)
class Config:
    raw_path: Path
    output_directory: Path
    views: int = 360
    rows: int = 1024
    cols: int = 1024
    threshold: float = 5.0
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    ring_radius_mm: float = 50.0
    ring_half_spacing_mm: float = 50.0
    beads_per_ring: int = 12
    assignment_radius_px: float = 24.0
    max_frame_rmse_px: float = 2.0


CONFIG = Config(
    raw_path=Path(r"G:\Code\fanproj\fdk-test\example\multispectrum_sim\outputs\quality\double-ring-px10-py15-pz20-tu1-tv2-tn3-prx2-projection-1024x1024x360-f32.raw"),
    output_directory=Path("out/cbct_calibration/cho_reference/double_ring_px10_py15_pz20"),
)


def _weighted_centres_from_predictions(image: np.ndarray,
                                       seeds: np.ndarray,
                                       threshold: float,
                                       radius_px: float) -> np.ndarray:
    ys, xs = np.nonzero(image > threshold)
    if len(xs) == 0:
        raise ValueError("no bright pixels")
    pixels = np.column_stack([xs, ys]).astype(float)
    values = np.maximum(image[ys, xs].astype(float) - threshold, 1.0e-6)
    distance = np.sum((pixels[:, None, :] - seeds[None, :, :]) ** 2, axis=2)
    owner = np.argmin(distance, axis=1)
    nearest = np.min(distance, axis=1)
    output = np.empty_like(seeds, dtype=float)
    for bead in range(len(seeds)):
        selected = (owner == bead) & (nearest <= radius_px ** 2)
        if np.count_nonzero(selected) < 4:
            raise ValueError(f"bead {bead} has insufficient assigned pixels")
        weights = values[selected]
        output[bead] = np.sum(pixels[selected] * weights[:, None], axis=0) / np.sum(weights)
    return output


def _load_raw(config: Config) -> np.memmap:
    expected = config.views * config.rows * config.cols * 4
    if config.raw_path.stat().st_size != expected:
        raise ValueError("RAW size does not match views/rows/cols/float32 configuration")
    return np.memmap(config.raw_path, dtype="<f4", mode="r",
                     shape=(config.views, config.rows, config.cols))


def track_indexed_points(config: Config, cho: ChoConfig):
    raw = _load_raw(config)
    frame0 = np.asarray(raw[0])
    # 仅第一帧使用环形状和三维模体尺寸搜索初始编号；后续不重新排序。
    from cho_preprocess import detect_frame
    initial_candidates = detect_frame(frame0, config.threshold, split_merged=False)
    indexed = initialize_indices(initial_candidates, cho)
    tracks = [indexed]
    frame_ids = [0]
    frame_rmse = []
    rejected = []
    poses = []

    for frame in range(config.views):
        if frame == 0:
            measured = indexed
        else:
            if len(tracks) >= 2:
                previous_gap = max(frame_ids[-1] - frame_ids[-2], 1)
                current_gap = frame - frame_ids[-1]
                predicted = tracks[-1] + (tracks[-1] - tracks[-2]) / previous_gap * current_gap
            else:
                predicted = tracks[-1]
            try:
                measured = _weighted_centres_from_predictions(
                    np.asarray(raw[frame]), predicted, config.threshold,
                    config.assignment_radius_px)
            except ValueError as exc:
                rejected.append({"frame": frame, "reason": str(exc)})
                continue
        try:
            pose = calibrate_view(measured, cho)
        except ValueError as exc:
            rejected.append({"frame": frame, "reason": str(exc)})
            continue
        error = float(pose.reprojection_rmse_px)
        if not np.isfinite(error) or error > config.max_frame_rmse_px:
            rejected.append({"frame": frame, "reason": f"DLT RMSE {error:.6g} px"})
            continue
        if frame > 0:
            tracks.append(measured)
        poses.append(pose)
        frame_ids.append(frame) if frame > 0 else None
        frame_rmse.append(error)

    return (np.asarray(tracks), frame_ids, poses, frame_rmse, rejected)


def run(config: Config) -> dict:
    cho = ChoConfig(
        ring_radius_mm=config.ring_radius_mm,
        ring_half_spacing_mm=config.ring_half_spacing_mm,
        beads_per_ring=config.beads_per_ring,
        pixel_size_mm=config.pixel_size_mm,
        robust_loss="soft_l1",
        robust_scale_px=0.15,
        max_nfev=500,
    )
    tracks, frame_ids, poses, frame_rmse, rejected = track_indexed_points(config, cho)
    if len(poses) < 10:
        raise RuntimeError(f"only {len(poses)} usable Cho poses remain")

    joint = calibrate_stack_joint(tracks, cho)
    system = analyze_poses(poses, (config.rows, config.cols), config.pixel_size_mm,
                           nominal_sid_mm=440.0)
    geometry = _jsonable(system)
    report = {
        "method": "Cho_2005_complete_geometry",
        "raw_path": str(config.raw_path),
        "phantom": {
            "ring_radius_mm": config.ring_radius_mm,
            "ring_half_spacing_mm": config.ring_half_spacing_mm,
            "beads_per_ring": config.beads_per_ring,
        },
        "preprocessing": {
            "tracking": "prediction_guided_pixel_ownership",
            "input_views": config.views,
            "usable_views": len(poses),
            "rejected_views": rejected,
            "frame_ids": frame_ids,
            "frame_rmse_mean_px": float(np.mean(frame_rmse)),
            "frame_rmse_max_px": float(np.max(frame_rmse)),
        },
        "cho_shared_intrinsics": {
            "sdd_mm": float(joint.shared_sdd_mm),
            "principal_point_px": joint.shared_principal_point_px.tolist(),
            "offset_from_image_center_px": (
                joint.shared_principal_point_px - np.array([(config.cols - 1) / 2,
                                                             (config.rows - 1) / 2])).tolist(),
            "offset_from_image_center_mm": joint.shared_offset_mm.tolist(),
            "joint_reprojection_rmse_px": float(joint.reprojection_rmse_px),
            "optimizer_success": bool(joint.optimizer_success),
            "optimizer_message": joint.optimizer_message,
        },
        "cho_per_view": {
            "count": len(poses),
            "source_points_world_mm": [pose.source_world_mm.tolist() for pose in poses],
            "projection_matrices": [pose.projection_matrix.tolist() for pose in poses],
            "reprojection_rmse_px": [pose.reprojection_rmse_px for pose in poses],
        },
        "cho_system_geometry": geometry,
    }
    config.output_directory.mkdir(parents=True, exist_ok=True)
    np.save(config.output_directory / "points_indexed.npy", tracks)
    (config.output_directory / "cho_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _save_summary_figure(config.output_directory, poses, system)
    return report


def _save_summary_figure(output_directory: Path, poses, system) -> None:
    """Visualize only quantities directly obtained from Cho camera poses."""
    sources = np.asarray([pose.source_world_mm for pose in poses])
    errors = np.asarray([pose.reprojection_rmse_px for pose in poses])
    sdd = np.asarray([pose.source_detector_distance_mm for pose in poses])
    principal = np.asarray([pose.principal_point_px for pose in poses])
    circle = system.source_circle
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    fitted_circle = (circle.center_world_mm[None, :]
                     + circle.radius_mm * np.cos(angle)[:, None] * circle.basis_x_world[None, :]
                     + circle.radius_mm * np.sin(angle)[:, None] * circle.basis_y_world[None, :])

    figure = plt.figure(figsize=(13, 9))
    source_axis = figure.add_subplot(2, 2, 1, projection="3d")
    source_axis.plot(fitted_circle[:, 0], fitted_circle[:, 1], fitted_circle[:, 2],
                     color="tab:orange", label="fitted source circle")
    source_axis.scatter(sources[:, 0], sources[:, 1], sources[:, 2],
                        s=8, color="tab:blue", label="Cho source points")
    source_axis.set_title("Source trajectory in phantom coordinates")
    source_axis.set_xlabel("X / mm")
    source_axis.set_ylabel("Y / mm")
    source_axis.set_zlabel("Z / mm")
    source_axis.legend(fontsize=8)

    error_axis = figure.add_subplot(2, 2, 2)
    error_axis.plot(errors, linewidth=1.0)
    error_axis.set(title="Per-view reprojection RMSE", xlabel="view", ylabel="RMSE / px")
    error_axis.grid(alpha=0.25)

    sdd_axis = figure.add_subplot(2, 2, 3)
    sdd_axis.plot(sdd, linewidth=1.0)
    sdd_axis.axhline(np.median(sdd), color="tab:red", linestyle="--",
                     label=f"median {np.median(sdd):.3f} mm")
    sdd_axis.set(title="Per-view SDD", xlabel="view", ylabel="SDD / mm")
    sdd_axis.grid(alpha=0.25)
    sdd_axis.legend(fontsize=8)

    principal_axis = figure.add_subplot(2, 2, 4)
    principal_axis.scatter(principal[:, 0], principal[:, 1], s=9, alpha=0.65)
    principal_axis.scatter([principal[:, 0].mean()], [principal[:, 1].mean()],
                           marker="x", s=80, color="tab:red", label="mean")
    principal_axis.set(title="Per-view principal point", xlabel="u0 / px", ylabel="v0 / px")
    principal_axis.grid(alpha=0.25)
    principal_axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_directory / "cho_summary.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    report = run(CONFIG)
    shared = report["cho_shared_intrinsics"]
    system = report["cho_system_geometry"]
    print("Cho 双环完整几何标定完成")
    print(f"有效帧: {report['preprocessing']['usable_views']} / {report['preprocessing']['input_views']}")
    print(f"共享 SDD = {shared['sdd_mm']:.3f} mm")
    print(f"共享主点 = ({shared['principal_point_px'][0]:.3f}, {shared['principal_point_px'][1]:.3f}) px")
    print(f"共享 offset = ({shared['offset_from_image_center_px'][0]:.3f}, {shared['offset_from_image_center_px'][1]:.3f}) px")
    print(f"联合重投影 RMSE = {shared['joint_reprojection_rmse_px']:.3f} px")
    print(f"等效源轨迹半径 = {system['source_circle']['radius_mm']:.3f} mm")
    print("机械 offset/tilt 未从 Cho DLT 坐标规范中独立分解")
    print(f"结果目录: {CONFIG.output_directory.resolve()}")
