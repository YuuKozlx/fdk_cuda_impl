"""用一帧 DLT 投影矩阵做真正的射线驱动正投影。

这个示例刻意不把 ``P @ X`` 称为正投影。``P @ X`` 只计算物点的
二维重投影位置；本文件对每个探测器像素建立一条射线，并使用 Siddon
算法累加射线穿过体素的长度：

    g(u, v) = sum_j volume[j] * L_j(u, v)

其中 L_j 是该射线在第 j 个体素中的实际穿透长度，单位为 mm。

使用方式：在 IDE 中打开本文件，只修改顶部 CONFIG，然后直接运行。
默认读取已经完成 Cho/DLT 标定的第 0 帧投影矩阵，并在一个小型球珠
模体上生成 256 x 256 的射线积分投影。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cho_calibration import ChoConfig, phantom_points, project_points


CONFIG = {
    # 已由 cho_raw_example.py 产生的报告；矩阵的世界坐标是 Cho 模体坐标。
    "report_json": Path(
        "out/cbct_calibration/cho_reference/"
        "double_ring_px10_py15_pz20/cho_report.json"
    ),
    "indexed_points_npy": Path(
        "out/cbct_calibration/cho_reference/"
        "double_ring_px10_py15_pz20/points_indexed.npy"
    ),
    "view_index": 0,
    # 为了让 Python 示例运行得快，只生成探测器的一个 ROI。
    # 这一块覆盖第 2、3 号钢珠的投影；改成 (300,320,500,520)
    # 可查看两环全部投影，但纯 Python Siddon 会相应变慢。
    "roi_u0": 300,
    "roi_v0": 320,
    "roi_width": 96,
    "roi_height": 96,
    "pixel_size_mm": 0.417,
    # 160^3 个 1 mm 体素覆盖双环钢珠模体（z 范围约 +/-50 mm）。
    "volume_shape": (112, 112, 112),  # nx, ny, nz；覆盖 +/-56 mm
    "volume_voxel_mm": 1.0,
    "bead_radius_mm": 1.5,
    "bead_attenuation": 1.0,
    "output_directory": Path("out/cbct_calibration/cho_reference/dlt_siddon_forward"),
}


def load_projection_matrix(report_json: Path, view_index: int) -> np.ndarray:
    report = json.loads(report_json.read_text(encoding="utf-8"))
    matrices = np.asarray(report["cho_per_view"]["projection_matrices"], dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 4):
        raise ValueError("report 中没有形状为 (views, 3, 4) 的 projection_matrices")
    if not 0 <= view_index < len(matrices):
        raise IndexError("view_index 超出 report 中的帧数")
    return matrices[view_index]


def source_from_projection(p: np.ndarray) -> np.ndarray:
    """由 P=[M|p4] 求射线源点 S，使 P [S,1]^T = 0。"""
    m = p[:, :3]
    return -np.linalg.solve(m, p[:, 3])


def ray_from_pixel(p: np.ndarray, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """返回世界坐标中的射线 (source, unit_direction)。

    对图像点 q=[u,v,1]^T，所有满足 P X ~ q 的点组成一条直线。
    直线方向可以取 M^{-1}q；这一步只由 DLT 矩阵完成，不需要假定
    source/detector 的机械 offset 或 tilt。
    """
    source = source_from_projection(p)
    direction = np.linalg.solve(p[:, :3], np.array([u, v, 1.0]))
    direction /= np.linalg.norm(direction)
    # direction 的正负不影响直线，但要沿 source -> detector 方向积分。
    if np.dot(direction, -source) < 0:
        direction = -direction
    return source, direction


def make_bead_volume(points_mm: np.ndarray, shape: tuple[int, int, int],
                     voxel_mm: float, radius_mm: float,
                     attenuation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 24 个三维钢珠离散成一个简单的球体衰减体。"""
    nx, ny, nz = shape
    axes = [
        (np.arange(n, dtype=float) - (n - 1) / 2.0) * voxel_mm
        for n in shape
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    volume = np.zeros(shape, dtype=np.float32)
    # 逐珠更新，避免构造 (24, nx, ny, nz) 的大数组。
    for centre in points_mm:
        mask = ((x - centre[0]) ** 2 + (y - centre[1]) ** 2
                + (z - centre[2]) ** 2) <= radius_mm ** 2
        volume[mask] = attenuation
    return volume, np.asarray(axes[0]), np.asarray(axes[1])


def ray_box_interval(source: np.ndarray, direction: np.ndarray,
                     bounds_min: np.ndarray, bounds_max: np.ndarray):
    """返回射线与体积盒的 [t0,t1]，没有交点则返回 None。"""
    t0, t1 = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-12:
            if source[axis] < bounds_min[axis] or source[axis] > bounds_max[axis]:
                return None
            continue
        a = (bounds_min[axis] - source[axis]) / direction[axis]
        b = (bounds_max[axis] - source[axis]) / direction[axis]
        lo, hi = min(a, b), max(a, b)
        t0, t1 = max(t0, lo), min(t1, hi)
        if t0 >= t1:
            return None
    return max(t0, 0.0), t1


def siddon_ray_integral(volume: np.ndarray, source: np.ndarray,
                        direction: np.ndarray, voxel_mm: float) -> float:
    """单条射线的 Siddon 长度积分，体素值视为线性衰减系数。"""
    shape = np.asarray(volume.shape, dtype=int)
    bounds_min = -(shape - 1) * voxel_mm / 2.0 - voxel_mm / 2.0
    bounds_max = +(shape - 1) * voxel_mm / 2.0 + voxel_mm / 2.0
    interval = ray_box_interval(source, direction, bounds_min, bounds_max)
    if interval is None:
        return 0.0
    t0, t1 = interval
    cuts = [t0, t1]
    # 每一组体素面给出一个参数 t，合并后相邻区间各属于一个体素。
    for axis, n in enumerate(shape):
        planes = bounds_min[axis] + np.arange(1, n, dtype=float) * voxel_mm
        if abs(direction[axis]) > 1.0e-12:
            cuts.extend(((planes - source[axis]) / direction[axis]).tolist())
    cuts = np.unique(np.clip(np.asarray(cuts), t0, t1))
    total = 0.0
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right - left <= 1.0e-10:
            continue
        midpoint = source + (0.5 * (left + right)) * direction
        index = np.floor((midpoint - bounds_min) / voxel_mm).astype(int)
        if np.all((index >= 0) & (index < shape)):
            total += float(volume[tuple(index)]) * float(right - left)
    return total


def siddon_forward(p: np.ndarray, volume: np.ndarray, roi_u0: int, roi_v0: int,
                   width: int, height: int, voxel_mm: float) -> np.ndarray:
    """对 ROI 内每个像素生成一条 DLT 射线并求 Siddon 积分。"""
    projection = np.zeros((height, width), dtype=np.float32)
    for row in range(height):
        v = float(roi_v0 + row)
        for col in range(width):
            u = float(roi_u0 + col)
            source, direction = ray_from_pixel(p, u, v)
            projection[row, col] = siddon_ray_integral(
                volume, source, direction, voxel_mm)
    return projection


def save_figure(output: Path, projection: np.ndarray, points_px: np.ndarray,
                config: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    v_min = config["roi_v0"]
    v_max = config["roi_v0"] + projection.shape[0]
    u_min = config["roi_u0"]
    u_max = config["roi_u0"] + projection.shape[1]
    extent = [u_min, u_max, v_max, v_min]
    image = ax.imshow(projection, cmap="magma", extent=extent)
    visible = (
        (points_px[:, 0] >= u_min) & (points_px[:, 0] <= u_max)
        & (points_px[:, 1] >= v_min) & (points_px[:, 1] <= v_max)
    )
    ax.scatter(points_px[visible, 0], points_px[visible, 1],
               facecolors="none", edgecolors="cyan", s=50, label="DLT bead projection")
    ax.set(xlabel="detector u / pixel", ylabel="detector v / pixel",
           title="DLT geometry + Siddon ray-driven forward projection")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="line integral (attenuation x mm)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    c = CONFIG
    c["output_directory"].mkdir(parents=True, exist_ok=True)
    p = load_projection_matrix(c["report_json"], c["view_index"])
    phantom = phantom_points(ChoConfig(
        ring_radius_mm=50.0, ring_half_spacing_mm=50.0,
        beads_per_ring=12, pixel_size_mm=(c["pixel_size_mm"], c["pixel_size_mm"])))
    points_px = project_points(phantom, p)
    measured = np.load(c["indexed_points_npy"])[c["view_index"]]
    reprojection_error = np.linalg.norm(points_px - measured, axis=1)
    volume, _, _ = make_bead_volume(
        phantom, c["volume_shape"], c["volume_voxel_mm"],
        c["bead_radius_mm"], c["bead_attenuation"])
    projection = siddon_forward(
        p, volume, c["roi_u0"], c["roi_v0"], c["roi_width"],
        c["roi_height"], c["volume_voxel_mm"])
    np.save(c["output_directory"] / "siddon_projection.npy", projection)
    save_figure(c["output_directory"] / "siddon_projection.png", projection, points_px, c)
    result = {
        "view_index": c["view_index"],
        "source_world_mm": source_from_projection(p).tolist(),
        "bead_reprojection_rmse_px": float(np.sqrt(np.mean(reprojection_error ** 2))),
        "bead_reprojection_max_px": float(np.max(reprojection_error)),
        "projection_shape": list(projection.shape),
        "projection_min": float(projection.min()),
        "projection_max": float(projection.max()),
        "nonzero_pixels": int(np.count_nonzero(projection)),
        "projector": "DLT ray construction + Siddon voxel path-length integration",
    }
    (c["output_directory"] / "siddon_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
