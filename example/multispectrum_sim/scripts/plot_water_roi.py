"""绘制 report_material_roi.py 输出的固定水模 ROI；不使用解剖整层均值。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Compare fixed water ROI z profiles")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        roi = report["fixed_roi"]
        values = np.asarray(roi["means_mm_inv"])
        label = f"{report['pipeline']} (r={roi['radius_mm']:g} mm)"
        axes[0].plot(roi["z_mm"], values, label=label)
        # 去均值仅用于观察纵向变化，不能用于掩盖算法之间的绝对偏差。
        axes[1].plot(roi["z_mm"], values-values.mean(), label=label)
    axes[0].set_ylabel("ROI mean (mm^-1)")
    axes[1].set_ylabel("ROI mean - own average (mm^-1)")
    axes[1].set_xlabel("z (mm)")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    fig.suptitle("Water cylinder: absolute value and longitudinal variation")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
