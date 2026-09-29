"""检查编号可辨识的单排钢珠能否逐帧恢复完整投影矩阵。

直接在 IDE 中运行本文件即可，不需要命令行参数。只需修改 CONFIG。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Config:
    tracks_path: Path
    output_directory: Path
    spacing_mm: float = 10.0
    example_view: int = 0


CONFIG = Config(
    tracks_path=Path("out/cbct_calibration/_api_single_row_smoke/tracks.npz"),
    output_directory=Path("out/cbct_calibration/single_row_dlt_observability"),
    spacing_mm=10.0,
    example_view=0,
)


def dlt_design_matrix(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """建立 q ~ P X 的标准 2N x 12 DLT 设计矩阵。"""
    rows = []
    for point, (u, v) in zip(points_3d, points_2d):
        x = np.r_[point, 1.0]
        rows.append(np.r_[x, np.zeros(4), -u * x])
        rows.append(np.r_[np.zeros(4), x, -v * x])
    return np.asarray(rows, dtype=float)


def fit_projection_on_line(z_mm: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """仅拟合单排直线上可被观测的投影列 P[:, 2:4]。"""
    z = np.asarray(z_mm, dtype=float)
    measured = np.asarray(points_2d, dtype=float)

    def residual(p):
        denominator = p[4] * z + 1.0
        predicted = np.column_stack([
            (p[0] * z + p[1]) / denominator,
            (p[2] * z + p[3]) / denominator,
        ])
        return (predicted - measured).ravel()

    slope_u, intercept_u = np.polyfit(z, measured[:, 0], 1)
    slope_v, intercept_v = np.polyfit(z, measured[:, 1], 1)
    initial = np.array([slope_u, intercept_u, slope_v, intercept_v, 0.0])
    result = least_squares(residual, initial, loss="soft_l1", f_scale=0.1)
    a, b, d, e, c = result.x
    return np.array([[a, b], [d, e], [c, 1.0]])


def make_projection(line_columns: np.ndarray,
                    free_xy_columns: np.ndarray) -> np.ndarray:
    """补入任意 X/Y 列，形成一个完整 3x4 投影矩阵。"""
    projection = np.empty((3, 4), dtype=float)
    projection[:, :2] = np.asarray(free_xy_columns, dtype=float)
    projection[:, 2:] = line_columns
    return projection


def project(points_3d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points_3d, np.ones(len(points_3d))]) @ projection.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def camera_center(projection: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(projection)
    centre = vt[-1]
    return centre[:3] / centre[3]


def run(config: Config) -> dict:
    data = np.load(config.tracks_path)
    tracks = np.asarray(data["tracks_px"], dtype=float)
    bead_ids = np.asarray(data["kept_bead_ids"], dtype=float)
    z_mm = bead_ids * config.spacing_mm
    points_3d = np.column_stack([np.zeros(len(z_mm)), np.zeros(len(z_mm)), z_mm])

    ranks = []
    singular_values = []
    for points_2d in tracks:
        design = dlt_design_matrix(points_3d, points_2d)
        singular = np.linalg.svd(design, compute_uv=False)
        ranks.append(int(np.sum(singular > singular[0] * 1.0e-10)))
        singular_values.append(singular)

    view = int(np.clip(config.example_view, 0, len(tracks) - 1))
    line_columns = fit_projection_on_line(z_mm, tracks[view])

    # X=Y=0，所以这两组明显不同的自由列不影响单排点的二维投影。
    free_a = np.array([[1.0, 0.0], [0.0, 1.0], [0.001, -0.002]])
    free_b = np.array([[0.2, 1.3], [-1.1, 0.4], [0.02, 0.01]])
    projection_a = make_projection(line_columns, free_a)
    projection_b = make_projection(line_columns, free_b)
    predicted_a = project(points_3d, projection_a)
    predicted_b = project(points_3d, projection_b)
    centre_a = camera_center(projection_a)
    centre_b = camera_center(projection_b)

    point_error = np.linalg.norm(predicted_a - tracks[view], axis=1)
    prediction_difference = np.linalg.norm(predicted_a - predicted_b, axis=1)
    nullities = 12 - np.asarray(ranks)
    report = {
        "conclusion": "single_row_is_not_sufficient_for_per_view_full_DLT",
        "reason": "point IDs are known, but all 3-D points are collinear",
        "views": int(len(tracks)),
        "points_per_view": int(tracks.shape[1]),
        "rank_min": int(min(ranks)),
        "rank_max": int(max(ranks)),
        "required_rank": 11,
        "nullity_min": int(nullities.min()),
        "nullity_max": int(nullities.max()),
        "example_view": view,
        "camera_center_a_mm": centre_a.tolist(),
        "camera_center_b_mm": centre_b.tolist(),
        "camera_center_distance_mm": float(np.linalg.norm(centre_a - centre_b)),
        "max_prediction_difference_px": float(prediction_difference.max()),
        "line_model_reprojection_rmse_px": float(np.sqrt(np.mean(point_error ** 2))),
    }

    config.output_directory.mkdir(parents=True, exist_ok=True)
    (config.output_directory / "observability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    singular_values = np.asarray(singular_values)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ranks, label="actual DLT rank")
    axes[0].axhline(11, color="tab:red", linestyle="--", label="required rank = 11")
    axes[0].set(xlabel="view", ylabel="matrix rank",
                title="Single-row per-view DLT observability")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].semilogy(singular_values[view], "o-")
    axes[1].set(xlabel="singular-value index", ylabel="singular value",
                title=f"View {view}: DLT singular values")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(config.output_directory / "dlt_observability.png", dpi=180)
    plt.close(figure)
    return report


if __name__ == "__main__":
    result = run(CONFIG)
    print("单排钢珠逐帧完整 DLT 可观测性检查完成")
    print(f"实际秩: {result['rank_min']} ~ {result['rank_max']}；唯一解需要 11")
    print(f"未约束维数: {result['nullity_min']} ~ {result['nullity_max']}")
    print("结论：编号正确仍不足以恢复完整投影矩阵、来源点和探测器姿态。")
    print(f"结果目录: {CONFIG.output_directory.resolve()}")
