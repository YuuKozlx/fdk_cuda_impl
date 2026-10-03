"""Render complete per-view Cho/Yang PIC diagnostic curves.

This is a reporting helper only.  It does not change either calibration
method or repair invalid PIC branches.  Wrapped angles are unwrapped for
display, while the raw JSON values remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _series(frames, getter):
    return np.asarray([getter(frame) for frame in frames], dtype=float)


def _unwrap_deg(values):
    return np.degrees(np.unwrap(np.radians(values)))


def _line_angle_deg(values):
    """Display a line orientation in its physical 180-degree equivalence class."""
    values = np.asarray(values, dtype=float)
    return (values + 90.0) % 180.0 - 90.0


def _plot_pair(ax, xa, a, xb, b, labels, ylabel, title, truth=None):
    ax.plot(xa, a, lw=0.8, label=labels[0])
    ax.plot(xb, b, lw=0.8, label=labels[1])
    if truth is not None:
        ax.axhline(truth, color="0.35", ls="--", lw=0.8, label="truth/reference")
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _cho_curves(frames):
    geo = [frame["complete_geometry"] for frame in frames]
    return {
        "sdd": _series(frames, lambda f: f["complete_geometry"]["source_detector_distance_mm"]),
        "principal_u": _series(frames, lambda f: f["complete_geometry"]["principal_point_px"][0]),
        "principal_v": _series(frames, lambda f: f["complete_geometry"]["principal_point_px"][1]),
        "piercing_u": _series(frames, lambda f: f["piercing_point_px"][0]),
        "piercing_v": _series(frames, lambda f: f["piercing_point_px"][1]),
        "phi": _unwrap_deg(_series(frames, lambda f: f["complete_geometry"]["cho_phi_deg"])),
        "theta": _unwrap_deg(_series(frames, lambda f: f["complete_geometry"]["cho_theta_deg"])),
        "eta": _line_angle_deg(_series(frames, lambda f: f["complete_geometry"]["cho_eta_deg"])),
        "gantry": _unwrap_deg(_series(frames, lambda f: f["complete_geometry"]["gantry_angle_deg"])),
        "lower_a": _series(frames, lambda f: f["ellipse_lower"]["semi_axes_px"][0]),
        "lower_b": _series(frames, lambda f: f["ellipse_lower"]["semi_axes_px"][1]),
        "upper_a": _series(frames, lambda f: f["ellipse_upper"]["semi_axes_px"][0]),
        "upper_b": _series(frames, lambda f: f["ellipse_upper"]["semi_axes_px"][1]),
        "line_rms": _series(frames, lambda f: f["line_fit_rms_px"]),
        "rmse": _series(frames, lambda f: f["complete_geometry"]["reprojection_rmse_px"]),
    }


def _yang_curves(frames, image_height=1024):
    return {
        "sdd": _series(frames, lambda f: f["source_detector_distance_mm"]),
        "principal_u": _series(frames, lambda f: f["principal_point_px"][0]),
        # Workflow output is already converted to fdk-test v-down.
        "principal_v": _series(frames, lambda f: f["principal_point_px"][1]),
        "o_u": _series(frames, lambda f: f["o_px"][0]),
        "o_v": _series(frames, lambda f: f["o_px"][1]),
        "yaw": _unwrap_deg(np.degrees(_series(frames, lambda f: f["yaw_rad"]))),
        "roll": _unwrap_deg(np.degrees(_series(frames, lambda f: f["roll_rad"]))),
        "pitch": _unwrap_deg(np.degrees(_series(frames, lambda f: f["pitch_rad"]))),
        "gantry": _unwrap_deg(np.degrees(_series(frames, lambda f: f["gantry_angle_rad"]))),
        "source_phantom": _series(frames, lambda f: f["source_phantom_distance_mm"]),
        "source_axis": _series(frames, lambda f: f["source_axis_distance_mm"]),
        "ellipse_condition_0": _series(frames, lambda f: f["ring_ellipses"][0]["condition"]),
        "ellipse_condition_1": _series(frames, lambda f: f["ring_ellipses"][1]["condition"]),
        "rmse": _series(frames, lambda f: f["reprojection_rmse_px"]),
    }


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_pic_curves(cho_json: Path, yang_json: Path, output: Path) -> Path:
    """Render every reported PIC value against its original frame index."""
    cho_report = _read_json(cho_json)
    yang_report = _read_json(yang_json)
    cho = _cho_curves(cho_report["pic"])
    yang = _yang_curves(yang_report["pic_steps"],
                        image_height=int(yang_report["input"]["image_shape"][0]))
    cho_ids = np.asarray(cho_report["tracking"]["valid_frame_ids"], dtype=int)
    yang_ids = np.arange(len(yang["sdd"]), dtype=int)
    x0, x1 = cho_ids, yang_ids

    fig, axes = plt.subplots(5, 2, figsize=(16, 22), constrained_layout=True)
    fig.suptitle("Per-view PIC parameters: Cho 12+12 and Yang 6+6", fontsize=16)

    _plot_pair(axes[0, 0], x0, cho["sdd"], x1, yang["sdd"], ("Cho", "Yang"), "mm", "SDD")
    axes[0, 0].axhline(770.0, color="0.35", ls="--", lw=0.8, label="truth")
    axes[0, 0].legend(frameon=False)
    _plot_pair(axes[0, 1], x0, cho["rmse"], x1, yang["rmse"], ("Cho", "Yang"), "px", "PIC reprojection RMSE")
    axes[0, 1].set_yscale("log")

    _plot_pair(axes[1, 0], x0, cho["principal_u"], x1, yang["principal_u"], ("Cho", "Yang"), "px", "Principal point u")
    _plot_pair(axes[1, 1], x0, cho["principal_v"], x1, yang["principal_v"], ("Cho", "Yang -> v-down"), "px", "Principal point v (common image coordinates)")
    _plot_pair(axes[2, 0], x0, cho["piercing_u"], x1, yang["o_u"], ("Cho piercing u", "Yang O u"), "px", "Piercing point / O: u")
    _plot_pair(axes[2, 1], x0, cho["piercing_v"], x1, yang["o_v"], ("Cho piercing v", "Yang O v-down"), "px", "Piercing point / O: v (common image coordinates)")

    axes[3, 0].plot(x0, cho["phi"], label="Cho phi")
    axes[3, 0].plot(x0, cho["theta"], label="Cho theta")
    axes[3, 0].plot(x0, cho["eta"], label="Cho eta")
    axes[3, 0].plot(x0, cho["gantry"], label="Cho gantry")
    axes[3, 0].set_title("Cho angles (eta wrapped modulo 180 deg)")
    axes[3, 0].set_ylabel("deg"); axes[3, 0].set_xlabel("frame")
    axes[3, 0].legend(frameon=False, fontsize=8); axes[3, 0].grid(alpha=0.2)
    axes[3, 1].plot(x1, yang["yaw"], label="Yang yaw")
    axes[3, 1].plot(x1, yang["roll"], label="Yang roll")
    axes[3, 1].plot(x1, yang["pitch"], label="Yang pitch")
    axes[3, 1].plot(x1, yang["gantry"], label="Yang gantry")
    axes[3, 1].set_title("Yang angles (unwrapped for display)")
    axes[3, 1].set_ylabel("deg"); axes[3, 1].set_xlabel("frame")
    axes[3, 1].legend(frameon=False, fontsize=8); axes[3, 1].grid(alpha=0.2)

    axes[4, 0].plot(x0, cho["lower_a"], label="lower a")
    axes[4, 0].plot(x0, cho["lower_b"], label="lower b")
    axes[4, 0].plot(x0, cho["upper_a"], label="upper a")
    axes[4, 0].plot(x0, cho["upper_b"], label="upper b")
    axes[4, 0].set_title("Cho ellipse semi-axes")
    axes[4, 0].set_ylabel("px"); axes[4, 0].set_xlabel("frame")
    axes[4, 0].legend(frameon=False, fontsize=8); axes[4, 0].grid(alpha=0.2)
    axes[4, 1].plot(x1, yang["source_phantom"], label="source-phantom")
    axes[4, 1].plot(x1, yang["source_axis"], label="source-axis")
    axes[4, 1].plot(x1, yang["ellipse_condition_0"], label="ellipse condition 0")
    axes[4, 1].plot(x1, yang["ellipse_condition_1"], label="ellipse condition 1")
    axes[4, 1].set_title("Yang distance and ellipse diagnostics")
    axes[4, 1].set_ylabel("mm / condition"); axes[4, 1].set_xlabel("frame")
    axes[4, 1].legend(frameon=False, fontsize=8); axes[4, 1].grid(alpha=0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _joint_machine(report: dict) -> tuple[dict, dict]:
    joint = report["fixed_source_joint_fit"]
    return joint["machine"], joint["diagnostics"]


def render_joint_fit(cho_json: Path, yang_json: Path, output: Path) -> Path:
    """Compare fixed-source joint fits with truth, grouped by physical unit."""
    cho_report = _read_json(cho_json)
    yang_report = _read_json(yang_json)
    cho, cho_diag = _joint_machine(cho_report)
    yang, yang_diag = _joint_machine(yang_report)
    truth = {
        "sid_mm": 440.0,
        "sdd_mm": 770.0,
        "offset_u_px": 5.0,
        "offset_v_px": 10.0,
        "tilt_u_deg": 1.0,
        "tilt_v_deg": 2.0,
        "tilt_n_deg": 3.0,
    }

    groups = [
        (("sid_mm", "sdd_mm"), "Distance", "mm"),
        (("offset_u_px", "offset_v_px"), "Detector offset", "px"),
        (("tilt_u_deg", "tilt_v_deg", "tilt_n_deg"), "Detector tilt", "deg"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle("Fixed-source joint fit (source offsets = 0, offsetN = 0)", fontsize=15)
    width = 0.24
    for ax, (keys, title, unit) in zip(axes, groups):
        x = np.arange(len(keys), dtype=float)
        ax.bar(x - width, [cho[k] for k in keys], width, label="Cho")
        ax.bar(x, [yang[k] for k in keys], width, label="Yang")
        ax.bar(x + width, [truth[k] for k in keys], width, label="simulator truth")
        ax.set_xticks(x, [key.replace("_mm", "").replace("_px", "").replace("_deg", "")
                          for key in keys])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.2)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3g", fontsize=8, padding=2)
    axes[0].legend(frameon=False)
    fig.text(
        0.5, 0.005,
        f"all-view reprojection RMSE: Cho {cho_diag['rmse_px']:.4f} px; "
        f"Yang {yang_diag['rmse_px']:.4f} px. Truth is mechanical; fits use a fixed-source equivalent gauge.",
        ha="center", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def render(cho_json: Path, yang_json: Path, output_directory: Path) -> tuple[Path, Path]:
    output_directory = Path(output_directory)
    return (
        render_pic_curves(cho_json, yang_json,
                          output_directory / "pic_parameters_all_frames.png"),
        render_joint_fit(cho_json, yang_json,
                         output_directory / "joint_fit_comparison.png"),
    )


if __name__ == "__main__":
    root = Path("out/cbct_calibration/raw_recalibration")
    render(root / "cho/calibration.json", root / "yang/calibration.json", root)
