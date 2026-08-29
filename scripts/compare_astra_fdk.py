"""使用 ASTRA 官方 FDK_CUDA 重建当前 C++ 大锥角测试的同一份投影。"""

from pathlib import Path
import argparse
import csv
import json
import time

import astra
import numpy as np


def write_pgm(path: Path, image: np.ndarray, lo: float, hi: float) -> None:
    value = np.clip((image - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    pixels = np.asarray(value * 255.0 + 0.5, dtype=np.uint8)
    with path.open("wb") as stream:
        stream.write(f"P5\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
        stream.write(np.flipud(pixels).tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTRA official cone-beam FDK comparison")
    parser.add_argument("--input", type=Path,
                        default=Path("output/cfdk/large_water/projection_f32.raw"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/cfdk/large_water/astra_fdk"))
    parser.add_argument("--nu", type=int, default=768)
    parser.add_argument("--nv", type=int, default=384)
    parser.add_argument("--views", type=int, default=720)
    parser.add_argument("--du", type=float, default=1.04)
    parser.add_argument("--dv", type=float, default=1.0)
    parser.add_argument("--sid", type=float, default=400.0)
    parser.add_argument("--sdd", type=float, default=800.0)
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=384)
    parser.add_argument("--nz", type=int, default=384)
    parser.add_argument("--voxel", type=float, default=0.4)
    args = parser.parse_args()

    expected = args.nu * args.nv * args.views
    raw = np.fromfile(args.input, dtype=np.float32)
    if raw.size != expected:
        raise RuntimeError(f"projection size {raw.size} != expected {expected}")

    # C++ raw 布局为 [view][detector_v][detector_u]；ASTRA 3-D sinogram
    # 布局为 [detector_row][view][detector_col]。
    projection = raw.reshape(args.views, args.nv, args.nu).transpose(1, 0, 2)
    angles = np.arange(args.views, dtype=np.float32) * (2.0 * np.pi / args.views)
    # ASTRA 的最后两个距离分别是 source->origin 和 origin->detector，
    # 而工程参数保存的是 SID 和 source->detector 的 SDD。
    origin_detector = args.sdd - args.sid
    proj_geom = astra.create_proj_geom(
        "cone", args.du, args.dv, args.nv, args.nu, angles,
        args.sid, origin_detector)
    extent = 0.5 * args.nx * args.voxel
    vol_geom = astra.create_vol_geom(
        args.nx, args.ny, args.nz,
        -extent, extent, -extent, extent, -extent, extent)

    args.output.mkdir(parents=True, exist_ok=True)
    sino_id = astra.data3d.create("-sino", proj_geom, projection)
    recon_id = astra.data3d.create("-vol", vol_geom)
    algorithm_id = astra.algorithm.create({
        "type": "FDK_CUDA",
        "ProjectionDataId": sino_id,
        "ReconstructionDataId": recon_id,
        "option": {"ShortScan": False},
    })
    start = time.perf_counter()
    try:
        astra.algorithm.run(algorithm_id)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        reconstruction = np.asarray(astra.data3d.get(recon_id), dtype=np.float32)
    finally:
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(recon_id)
        astra.data3d.delete(sino_id)

    reconstruction.tofile(args.output / "reconstruction_f32.raw")
    # 中心 100 mm 半径内逐层均值，和 C++ 大水模测试保持同一指标。
    yy, xx = np.indices((args.ny, args.nx), dtype=np.float32)
    x = (xx + 0.5 - 0.5 * args.nx) * args.voxel
    y = (yy + 0.5 - 0.5 * args.ny) * args.voxel
    mask = x * x + y * y <= 50.0 * 50.0
    z = (np.arange(args.nz, dtype=np.float32) + 0.5 - 0.5 * args.nz) * args.voxel
    profile = reconstruction[:, mask].mean(axis=1)
    center_index = int(np.argmin(np.abs(z)))
    end_index = int(np.argmin(np.abs(z - 60.0)))
    center = float(profile[center_index])
    end = float(profile[end_index])
    with (args.output / "z_profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["z_mm", "astra_fdk", "astra_fdk_over_center"])
        for z_mm, value in zip(z, profile):
            writer.writerow([float(z_mm), float(value), float(value / center if center else 0.0)])

    # 输出中心冠状面和矢状面，便于与已有 BMP 进行视觉检查。
    write_pgm(args.output / "coronal_center.pgm", reconstruction[:, args.ny // 2, :], 0.0, 0.025)
    write_pgm(args.output / "sagittal_center.pgm", reconstruction[:, :, args.nx // 2], 0.0, 0.025)
    metadata = {
        "astra_version": astra.__version__,
        "algorithm": "FDK_CUDA",
        "projection_layout_input": "[view][v][u] float32",
        "projection_layout_astra": "[v][view][u] float32",
        "detector_uv": [args.nu, args.nv],
        "detector_pixel_mm_uv": [args.du, args.dv],
        "sid_mm": args.sid,
        "sdd_mm": args.sdd,
        "origin_detector_mm": origin_detector,
        "views": args.views,
        "volume_xyz": [args.nx, args.ny, args.nz],
        "voxel_mm": args.voxel,
        "fdk_time_ms": elapsed_ms,
        "center_water_mean": center,
        "z60_water_mean": end,
        "z60_over_center": float(end / center if center else 0.0),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
