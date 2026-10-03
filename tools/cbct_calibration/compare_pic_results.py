"""Run and summarize Cho/Yang PIC comparisons on the synchronized marker phantoms.

This is a Python-callable integration entry point; it deliberately does not
feed simulator truth into either calibration method.  Truth fields may be
added to the returned table by the caller for report-only comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .coordinates import FDK_TEST_CONVENTION
from .cho.workflow import ChoWorkflowConfig, calibrate_raw as calibrate_cho
from .yang.workflow import YangWorkflowConfig, calibrate_raw as calibrate_yang


def _median(values):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return None if not len(a) else float(np.median(a))


def summarize(report: dict, method: str) -> dict:
    if method == "cho":
        frames = report["pic"]
        common = {
            "sdd_mm_median": _median([f["complete_geometry"]["source_detector_distance_mm"] for f in frames]),
            "principal_u_px_median": _median([f["complete_geometry"]["principal_point_px"][0] for f in frames]),
            "principal_v_px_median": _median([f["complete_geometry"]["principal_point_px"][1] for f in frames]),
            "piercing_u_px_median": _median([f["piercing_point_px"][0] for f in frames]),
            "piercing_v_px_median": _median([f["piercing_point_px"][1] for f in frames]),
            "reprojection_rmse_px_median": _median([f["complete_geometry"]["reprojection_rmse_px"] for f in frames]),
            "eta_deg_median": _median([f["detector_in_plane_angle_deg"] for f in frames]),
        }
    else:
        frames = report["pic_steps"]
        image_shape = tuple(report.get("input", {}).get("image_shape", [1024, 1024]))
        common = {
            "sdd_mm_median": _median([f["source_detector_distance_mm"] for f in frames]),
            "principal_u_px_median": _median([f["principal_point_px"][0] for f in frames]),
            "principal_v_px_median": _median([f["principal_point_px"][1]
                                               for f in frames]),
            "o_u_px_median": _median([f["o_px"][0] for f in frames]),
            "o_v_px_median": _median([f["o_px"][1] for f in frames]),
            "reprojection_rmse_px_median": _median([f["reprojection_rmse_px"] for f in frames]),
            "yaw_deg_median": _median([np.degrees(f["yaw_rad"]) for f in frames]),
            "roll_deg_median": _median([np.degrees(f["roll_rad"]) for f in frames]),
            "pitch_deg_median": _median([np.degrees(f["pitch_rad"]) for f in frames]),
        }
    return {
        "method": method,
        "frame_count": len(frames),
        "common_pic": common,
        "joint_fit": report.get("fixed_source_joint_fit", {}),
        "coordinate_convention": FDK_TEST_CONVENTION.name,
        "joint_fit_status": "report-only; validate virtual/mechanical convention before reconstruction",
    }


def run_comparison(base_directory: Path, output_directory: Path) -> dict:
    base = Path(base_directory)
    output = Path(output_directory)
    p12 = next(base.glob("synchronized-ring-12-marker-*-projection.raw"))
    p6 = next(base.glob("synchronized-ring-6-marker-*-projection.raw"))
    cho = calibrate_cho(ChoWorkflowConfig(p12, output / "cho"))
    yang = calibrate_yang(YangWorkflowConfig(p6, output / "yang", run_extended=False))
    result = {"cho": summarize(cho, "cho"), "yang": summarize(yang, "yang")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
