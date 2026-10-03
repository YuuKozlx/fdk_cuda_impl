"""Callable Yang workflow: target tracking -> six PIC steps -> extension."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..common import load_raw_float32, save_array, save_json
from ..coordinates import (DetectorConvention, FDK_TEST_CONVENTION,
                           PhantomFrameTransform, PointCorrespondence,
                           convert_detector_points,
                           convert_projection_world_and_detector,
                           convert_pose_pixel_fields,
                           )
from .coordinate_adapter import (YANG_PAPER_CONVENTION,
                                 YANG_PHANTOM_TO_FDK_FRAME,
                                 YANG_TRACKER_RING_ORDER)
from ..geometry.dlt_to_conevec import conevec_normal, dlt_to_conevec
from ..math.dlt import normalized_dlt
from .pic import (YangConfig, calibrate_stack_geometries, calibrate_view,
                  extended_calibrate_stack, extended_result_to_dict,
                  geometry_to_dict, pose_to_dict)
from .joint_fit import fit_fixed_source, phantom_points
from .tracker import (detect_components, external_marker_index_candidates,
                      track_standard_stack)


@dataclass(frozen=True)
class YangWorkflowConfig:
    raw_path: Path
    output_directory: Path
    views: int = 360
    image_shape: tuple[int, int] = (1024, 1024)
    threshold: float = 5.0
    pixel_size_mm: tuple[float, float] = (0.417, 0.417)
    ring_radius_mm: float = 50.0
    ring_half_spacing_mm: float = 50.0
    beads_per_ring: int = 6
    run_extended: bool = True
    source_constraint_weight: float = 12.0
    extended_max_nfev: int = 200
    # Explicit target correspondence; it is separate from the 3-D frame map.
    input_detector_convention: DetectorConvention = FDK_TEST_CONVENTION
    phantom_frame_transform: PhantomFrameTransform = YANG_PHANTOM_TO_FDK_FRAME
    point_correspondence: PointCorrespondence = YANG_TRACKER_RING_ORDER


def _canonical_geometry_dict(
        geometry, pose, image_shape: tuple[int, int],
        frame_transform: PhantomFrameTransform = YANG_PHANTOM_TO_FDK_FRAME,
        ) -> dict:
    """Serialize one Yang geometry entirely in the fdk-test 3-D frame."""
    p_fdk = convert_projection_world_and_detector(
        geometry.projection_matrix, image_shape,
        phantom_source_to_target=frame_transform.rotation,
        phantom_translation=frame_transform.translation,
        source_detector=YANG_PAPER_CONVENTION,
        target_detector=FDK_TEST_CONVENTION)
    cone = dlt_to_conevec(
        p_fdk, sdd_mm=pose.source_detector_distance_mm,
        point_toward=np.zeros(3))
    rows, columns = image_shape
    center_px = np.array([(columns - 1) / 2.0, (rows - 1) / 2.0])
    u_axis = cone.r_u / np.linalg.norm(cone.r_u)
    v_axis = cone.r_v / np.linalg.norm(cone.r_v)
    normal = conevec_normal(cone)
    detector_center = (cone.det_s + center_px[0] * cone.r_u
                       + center_px[1] * cone.r_v)
    return {
        "source_mm": cone.source.tolist(),
        "detector_origin_mm": cone.det_s.tolist(),
        "detector_center_mm": detector_center.tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
        "normal": normal.tolist(),
        "principal_point_px": convert_detector_points(
            geometry.principal_point_px, image_shape,
            YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION).tolist(),
        "projection_matrix": p_fdk.tolist(),
        "pixel_size_mm": np.asarray(geometry.pixel_size_mm).tolist(),
        "source_detector_distance_mm": float(pose.source_detector_distance_mm),
        "coordinate_frame": "fdk-test canonical phantom/world and detector",
    }


def _dlt_geometry_dict(projection: np.ndarray, pixel_size_mm: tuple[float, float],
                       image_shape: tuple[int, int]) -> dict:
    """Serialize an exact canonical DLT ray geometry."""
    cone = dlt_to_conevec(projection, sdd_mm=None, point_toward=np.zeros(3))
    rows, columns = image_shape
    center_px = np.array([(columns - 1) / 2.0, (rows - 1) / 2.0])
    u_axis = cone.r_u / np.linalg.norm(cone.r_u)
    v_axis = cone.r_v / np.linalg.norm(cone.r_v)
    return {
        "source_mm": cone.source.tolist(),
        "detector_origin_mm": cone.det_s.tolist(),
        "detector_center_mm": (cone.det_s + center_px[0] * cone.r_u
                                + center_px[1] * cone.r_v).tolist(),
        "u_axis": u_axis.tolist(),
        "v_axis": v_axis.tolist(),
        "normal": conevec_normal(cone).tolist(),
        "principal_point_px": center_px.tolist(),
        "projection_matrix": np.asarray(projection).tolist(),
        "pixel_size_mm": list(pixel_size_mm),
        "source_detector_distance_mm": None,
        "coordinate_frame": "fdk-test canonical phantom/world and detector",
    }


def calibrate_indexed_points(points_px: np.ndarray, config: YangWorkflowConfig) -> dict:
    """Run Yang PIC; retain its native results and expose fdk-test outputs."""
    image_points = np.asarray(points_px, dtype=float)
    expected = 2 * int(config.beads_per_ring)
    if image_points.ndim != 3 or image_points.shape[1:] != (expected, 2):
        raise ValueError(
            f"points_px must have shape (views,{expected},2)")
    points = convert_detector_points(
        image_points, config.image_shape, config.input_detector_convention,
        YANG_PAPER_CONVENTION)
    canonical_image_points = convert_detector_points(
        image_points, config.image_shape, config.input_detector_convention,
        FDK_TEST_CONVENTION)
    yang = YangConfig(ring_radius_mm=config.ring_radius_mm,
                      ring_half_spacing_mm=config.ring_half_spacing_mm,
                      beads_per_ring=config.beads_per_ring,
                      pixel_size_mm=config.pixel_size_mm,
                      gantry_solver="paper_linear")
    poses, geometries = calibrate_stack_geometries(points, yang)
    canonical_points = phantom_points(
        yang, config.point_correspondence, config.phantom_frame_transform)
    # The PIC phase is a diagnostic quantity; the complete ray matrix is
    # estimated directly from this frame's indexed 3-D targets and pixels.
    dlt_matrices = [normalized_dlt(canonical_points, frame)[0]
                    for frame in canonical_image_points]
    paper_poses = [pose_to_dict(pose) for pose in poses]
    paper_geometries = [geometry_to_dict(geometry) for geometry in geometries]
    fdk_poses = []
    fdk_geometries = []
    for pose, geometry, pose_dict, geometry_dict in zip(
            poses, geometries, paper_poses, paper_geometries):
        fdk_pose = convert_pose_pixel_fields(
            pose_dict, config.image_shape,
            YANG_PAPER_CONVENTION, FDK_TEST_CONVENTION)
        # The public geometry is rebuilt from the converted matrix, so its
        # source and detector axes cannot accidentally remain in Yang's local
        # virtual-detector frame.
        fdk_geometry = _dlt_geometry_dict(
            dlt_matrices[len(fdk_geometries)], config.pixel_size_mm,
            config.image_shape)
        fdk_poses.append(fdk_pose)
        fdk_geometries.append(fdk_geometry)
    report = {
        "method": "Yang_2017_PIC",
        "image_coordinate_convention": FDK_TEST_CONVENTION.name,
        "paper_coordinate_convention": YANG_PAPER_CONVENTION.name,
        "phantom_coordinate_convention": "fdk-test canonical for reconstruction_geometries",
        "pic_steps_paper_coordinates": paper_poses,
        "pic_steps": fdk_poses,
        "reconstruction_geometries_paper_coordinates": paper_geometries,
        "reconstruction_geometries": fdk_geometries,
        "dlt_projection_matrices": [p.tolist() for p in dlt_matrices],
    }
    report["fixed_source_joint_fit"] = fit_fixed_source(
        canonical_image_points, poses, yang, config.image_shape,
        max_nfev=config.extended_max_nfev,
        projection_matrices=dlt_matrices,
        correspondence=config.point_correspondence,
        frame_transform=config.phantom_frame_transform)
    if config.run_extended:
        result = extended_calibrate_stack(
            points, yang, source_constraint_weight=config.source_constraint_weight,
            max_nfev=config.extended_max_nfev, initial_poses=poses)
        extended_paper = extended_result_to_dict(result)
        extended_fdk = dict(extended_paper)
        extended_fdk["poses"] = [convert_pose_pixel_fields(
            pose, config.image_shape, YANG_PAPER_CONVENTION,
            FDK_TEST_CONVENTION) for pose in extended_paper["poses"]]
        report["extended_joint_fit_paper_coordinates"] = extended_paper
        report["extended_joint_fit"] = extended_fdk
    return report


def calibrate_raw(config: YangWorkflowConfig) -> dict:
    """Run the complete RAW workflow through Python-callable functions."""
    raw = load_raw_float32(config.raw_path,
                           (config.views, *config.image_shape))
    yang = YangConfig(ring_radius_mm=config.ring_radius_mm,
                      ring_half_spacing_mm=config.ring_half_spacing_mm,
                      beads_per_ring=config.beads_per_ring,
                      pixel_size_mm=config.pixel_size_mm,
                      gantry_solver="paper_linear")
    components = detect_components(np.asarray(raw[0]), config.threshold)
    initial, marker0, marker_diagnostics = external_marker_index_candidates(components)
    if not initial:
        raise RuntimeError("no valid Yang initial index candidate")
    # Ring swap and winding are not known from image coordinates.  Score every
    # candidate with the paper PIC reprojection error before tracking the stack.
    candidates = []
    for index, candidate in enumerate(initial):
        candidate_yang = convert_detector_points(
            candidate, config.image_shape, config.input_detector_convention,
            YANG_PAPER_CONVENTION)
        try:
            pose = calibrate_view(candidate_yang, yang)
        except (ValueError, FloatingPointError):
            continue
        candidates.append((pose.reprojection_rmse_px, index, candidate))
    if not candidates:
        raise RuntimeError("all Yang initial index candidates failed PIC")
    _, selected, initial_points = min(candidates, key=lambda item: item[0])
    image_points, diagnostics = track_standard_stack(
        raw, config.threshold, initial_points, initial_marker=marker0)
    points = np.asarray(image_points, dtype=float)
    report = calibrate_indexed_points(points, config)
    report["input"] = {"raw_path": str(config.raw_path), "views": config.views,
                        "image_shape": list(config.image_shape),
                        "pixel_size_mm": list(config.pixel_size_mm)}
    report["tracking"] = diagnostics
    report["index_initialization"] = {
        "selected_candidate": int(selected),
        "candidate_rmse_px": {str(i): float(score) for score, i, _ in candidates},
        "external_marker": marker_diagnostics,
    }
    save_array(config.output_directory / "points_indexed.npy", points)
    save_json(config.output_directory / "calibration.json", report)
    return report
