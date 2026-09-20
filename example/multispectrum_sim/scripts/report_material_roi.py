"""标签内部 ROI 回归报告；Python 3.11+ 与 numpy。

多能谱重建没有唯一单能真值，参考均值/绝对容差必须由用户明确提供。
整层脑组织均值随解剖结构改变，不用作 z 均匀性判据。
"""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np


def measure(volume, labels, erosion=1, z_margin=4):
    if volume.shape != labels.shape or volume.ndim != 3:
        raise ValueError("volume/labels must have matching 3D shapes")
    if erosion < 0 or z_margin < 0 or 2 * max(erosion, z_margin) >= volume.shape[0]:
        raise ValueError("invalid erosion/z margin")
    if 2 * erosion >= min(volume.shape):
        raise ValueError("erosion removes the whole volume")
    if not np.isfinite(volume).all():
        raise ValueError("volume contains NaN/Inf")
    result = {}
    for label in np.unique(labels):
        mask = labels == label
        # 立方邻域腐蚀；每轮剔除所有 26 邻域触及其他材料的体素。
        for _ in range(erosion):
            padded = np.pad(mask, 1, constant_values=False)
            mask = np.logical_and.reduce([
                padded[z:z+mask.shape[0], y:y+mask.shape[1], x:x+mask.shape[2]]
                for z in range(3) for y in range(3) for x in range(3)])
        if z_margin:
            mask[:z_margin] = False
            mask[-z_margin:] = False
        values = volume[mask].astype(np.float64)
        result[str(int(label))] = {"count": int(values.size),
            "mean_mm_inv": float(values.mean()) if values.size else None,
            "std_mm_inv": float(values.std()) if values.size else None}
    return result


def main():
    parser = argparse.ArgumentParser(description="Report material interior ROI means (mm^-1)")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", type=Path, help="explicit exported water/label RAW (relative to current directory)")
    parser.add_argument("--reference", type=Path, help="JSON: labels -> label -> mean_mm_inv, tolerance_mm_inv")
    parser.add_argument("--erosion", type=int, default=1)
    parser.add_argument("--z-margin", type=int, default=4)
    parser.add_argument("--contrast-labels", nargs=2, type=int, help="report first mean minus second mean")
    parser.add_argument("--fixed-radius-mm", type=float, help="fixed central circular ROI for homogeneous water phantom")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8-sig"))
    base = args.config.resolve().parent
    g = config["geometry_config"]
    for axis in "xyz":
        if g.get(f"phantom_offset_{axis}_mm", 0) != g.get(f"reconstruction_offset_{axis}_mm", 0):
            raise ValueError("ROI requires aligned phantom/reconstruction grids; offsets differ")
    shape = tuple(g[f"volume_{a}"] for a in "zyx")
    label_path = args.labels or base / config["simulation"]["label_volume"]
    labels = np.fromfile(label_path, dtype=np.uint8).reshape(shape)
    path = base / config["reconstruction"]["output_volume_file"]
    volume = np.fromfile(path, dtype="<f4").reshape(shape)
    stats = measure(volume, labels, args.erosion, args.z_margin)
    for material in config["materials"]:
        if str(material["label"]) in stats:
            stats[str(material["label"])]["name"] = material["name"]
    report = {"pipeline": config["reconstruction"]["pipeline"],
        "erosion_voxels": args.erosion, "z_margin": args.z_margin,
        "labels": stats, "reference_passed": None}
    if args.fixed_radius_mm is not None:
        radius = args.fixed_radius_mm
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("fixed radius must be finite and positive")
        yy, xx = np.indices(shape[1:])
        xx = (xx - (shape[2]-1)/2) * g["voxel_x_mm"]
        yy = (yy - (shape[1]-1)/2) * g["voxel_y_mm"]
        mask = xx**2 + yy**2 <= radius**2
        if not mask.any() or radius >= min(shape[1]*g["voxel_y_mm"], shape[2]*g["voxel_x_mm"])/2:
            raise ValueError("fixed ROI is empty or touches volume boundary")
        end = shape[0] - args.z_margin
        slab = slice(args.z_margin, end)
        if np.unique(labels[slab][:, mask]).size != 1:
            raise ValueError("fixed ROI must contain one homogeneous material across z")
        profile = volume[slab][:, mask].astype(np.float64).mean(axis=1)
        report["fixed_roi"] = {"radius_mm": radius, "mean_mm_inv": float(profile.mean()),
            "z_std_mm_inv": float(profile.std()), "z_range_mm_inv": float(np.ptp(profile)),
            "z_mm": [(z-(shape[0]-1)/2)*g["voxel_z_mm"]+g.get("reconstruction_offset_z_mm", 0)
                     for z in range(args.z_margin, end)], "means_mm_inv": profile.tolist()}
    if args.contrast_labels:
        a, b = (stats[str(label)]["mean_mm_inv"] for label in args.contrast_labels)
        if a is None or b is None:
            raise ValueError("contrast ROI is empty")
        report["contrast"] = {"labels": args.contrast_labels, "difference_mm_inv": a-b}
    if args.reference:
        baseline = json.loads(args.reference.read_text(encoding="utf-8"))
        for key in ("pipeline", "erosion_voxels", "z_margin"):
            if key in baseline and baseline[key] != report[key]:
                raise ValueError(f"reference metadata mismatch: {key}")
        references = baseline["labels"]
        if not references:
            raise ValueError("reference must contain at least one label")
        passed = True
        for label, reference in references.items():
            mean = stats.get(label, {}).get("mean_mm_inv")
            expected, tolerance = reference["mean_mm_inv"], reference["tolerance_mm_inv"]
            if not np.isfinite([expected, tolerance]).all() or tolerance < 0:
                raise ValueError("reference/tolerance must be finite and tolerance nonnegative")
            error = None if mean is None else mean - expected
            ok = error is not None and abs(error) <= tolerance
            stats.setdefault(label, {}).update(reference_mm_inv=expected, error_mm_inv=error, passed=ok)
            passed = passed and ok
        report["reference_passed"] = passed
    output = args.output or path.with_suffix(".roi.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"report={output} reference_passed={report['reference_passed']}")
    return 1 if report["reference_passed"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
