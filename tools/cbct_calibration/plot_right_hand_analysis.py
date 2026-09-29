from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_right_hand_joint import fit


def main():
    points = np.load("analysis_right_hand.points.npy")
    report, pred, err = fit(points)
    frame_rmse = np.sqrt(np.mean(err * err, axis=(1, 2)))
    point_norm = np.linalg.norm(err, axis=2)
    truth = {
        "SID": 440.0,
        "SDD": 770.0,
        "offset u": 2.085,
        "offset v": 4.170,
        "tilt n": 3.0,
    }
    estimated = {
        "SID": report["sid_mm"],
        "SDD": report["sdd_mm"],
        "offset u": report["offset_u_mm"],
        "offset v": report["offset_v_mm"],
        "tilt n": report["tilt_n_deg"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(np.sqrt(np.sum(err[0] ** 2, axis=1))[None, :], aspect="auto", cmap="viridis")
    ax.set_title("View 0 point error (px)")
    ax.set_xlabel("bead index")
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=.8)

    ax = axes[0, 1]
    ax.plot(frame_rmse, lw=1.0)
    ax.axhline(np.mean(frame_rmse), color="tab:red", ls="--", label=f"mean {np.mean(frame_rmse):.3f} px")
    ax.set_title("Reprojection RMSE by frame")
    ax.set_xlabel("frame")
    ax.set_ylabel("RMSE (px)")
    ax.grid(alpha=.25)
    ax.legend()

    ax = axes[1, 0]
    keys = list(truth)
    x = np.arange(len(keys))
    tv = np.array([truth[k] for k in keys])
    ev = np.array([estimated[k] for k in keys])
    scale = np.maximum(np.abs(tv), 1.0)
    ax.bar(x - .18, tv / scale, .36, label="truth")
    ax.bar(x + .18, ev / scale, .36, label="estimated")
    ax.set_xticks(x, keys, rotation=25, ha="right")
    ax.set_ylabel("normalized value")
    ax.set_title("Truth versus estimate")
    ax.grid(axis="y", alpha=.25)
    ax.legend()

    ax = axes[1, 1]
    ax.hist(err[..., 0].ravel(), bins=50, alpha=.65, label="u residual")
    ax.hist(err[..., 1].ravel(), bins=50, alpha=.65, label="v residual")
    ax.set_title(f"Residual distribution, joint RMSE {report['rmse_px']:.3f} px")
    ax.set_xlabel("residual (px)")
    ax.set_ylabel("count")
    ax.grid(alpha=.25)
    ax.legend()
    out = Path("analysis_right_hand_summary.png")
    fig.savefig(out, dpi=160)
    print(out.resolve())


if __name__ == "__main__":
    main()
