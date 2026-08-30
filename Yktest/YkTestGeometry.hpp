#pragma once

#include <vector>

#include "Heli/YkHeliCTParams.h"
#include "Heli/analytic/wfbp/YkWfbpTypes.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"

namespace YK::TestGeometry {

inline SFlatDetectorSpec flatDetector(const SHeliCTParam& p)
{
    SFlatDetectorSpec detector{};
    detector.channels = p.iPU; detector.rows = p.iPV;
    detector.channel_size_mm = p.du_mm; detector.row_size_mm = p.dv_mm;
    detector.pose.offset_unv_mm = make_float3(p.offsetU_mm, 0.f, p.offsetV_mm);
    detector.pose.tilt_u_rad = p.tiltu_angle_rad;
    detector.pose.tilt_v_rad = p.tiltv_angle_rad;
    detector.pose.tilt_n_rad = p.tiltn_angle_rad;
    return detector;
}

inline SCylDetectorSpec cylDetector(const SHeliCTParam& p, float radius_mm,
    float channel_angle_step_rad = 0.f)
{
    SCylDetectorSpec detector{};
    detector.channels = p.iPU; detector.rows = p.iPV;
    detector.channel_arc_mm = channel_angle_step_rad > 0.f
        ? radius_mm * channel_angle_step_rad : p.du_mm;
    detector.row_size_mm = p.dv_mm;
    detector.curvature_radius_mm = radius_mm;
    detector.pose.offset_unv_mm = make_float3(p.offsetU_mm, 0.f, p.offsetV_mm);
    detector.pose.tilt_u_rad = p.tiltu_angle_rad;
    detector.pose.tilt_v_rad = p.tiltv_angle_rad;
    detector.pose.tilt_n_rad = p.tiltn_angle_rad;
    return detector;
}

inline SCircularTrajectorySpec circularTrajectory(const SHeliCTParam& p)
{
    SCircularTrajectorySpec trajectory{};
    trajectory.angles_rad = p.angle_list;
    trajectory.sid_mm = p.SID; trajectory.sdd_mm = p.SDD;
    trajectory.z_mm = p.start_z_mm;
    trajectory.source_offset_mm = make_float3(
        p.sourceOffsetX_mm, p.sourceOffsetY_mm, p.sourceOffsetZ_mm);
    return trajectory;
}

inline SHelicalTrajectorySpec helicalTrajectory(const SHeliCTParam& p)
{
    SHelicalTrajectorySpec trajectory{};
    trajectory.angles_rad = p.angle_list;
    trajectory.sid_mm = p.SID; trajectory.sdd_mm = p.SDD;
    trajectory.start_z_mm = p.start_z_mm;
    trajectory.pitch_mm_per_turn = p.pitch_mm;
    trajectory.source_offset_mm = make_float3(
        p.sourceOffsetX_mm, p.sourceOffsetY_mm, p.sourceOffsetZ_mm);
    return trajectory;
}

inline SRegularHelicalScanSpec regularHelicalScan(const SHeliCTParam& p)
{
    SRegularHelicalScanSpec scan{};
    scan.total_views = static_cast<int>(p.angle_list.size());
    scan.views_per_turn = p.views_per_rot;
    scan.start_angle_rad = p.angle_list.empty() ? 0.f : p.angle_list.front();
    if (p.angle_list.size() > 1 && p.angle_list[1] < p.angle_list[0])
        scan.rotation_direction = -1;
    scan.sid_mm = p.SID;
    scan.sdd_mm = p.SDD;
    scan.start_z_mm = p.start_z_mm;
    scan.pitch_mm_per_turn = p.pitch_mm;
    scan.source_offset_mm = make_float3(
        p.sourceOffsetX_mm, p.sourceOffsetY_mm, p.sourceOffsetZ_mm);
    return scan;
}

inline SRegularCircularScanSpec regularCircularScan(const SHeliCTParam& p)
{
    SRegularCircularScanSpec scan{};
    scan.total_views = static_cast<int>(p.angle_list.size());
    scan.views_per_turn = p.views_per_rot;
    scan.start_angle_rad = p.angle_list.empty() ? 0.f : p.angle_list.front();
    if (p.angle_list.size() > 1 && p.angle_list[1] < p.angle_list[0])
        scan.rotation_direction = -1;
    scan.sid_mm = p.SID;
    scan.sdd_mm = p.SDD;
    scan.z_mm = p.start_z_mm;
    scan.source_offset_mm = make_float3(
        p.sourceOffsetX_mm, p.sourceOffsetY_mm, p.sourceOffsetZ_mm);
    return scan;
}

inline SVolumeGridSpec volumeGrid(const SHeliCTParam& p)
{
    return {p.iVX, p.iVY, p.iVZ, p.vox_x_mm, p.vox_y_mm, p.vox_z_mm,
        make_float3(p.vol_offset_x_mm, p.vol_offset_y_mm, p.vol_offset_z_mm)};
}

inline std::vector<SConeProjGeomVec> helicalFlat(const SHeliCTParam& p)
{
    SHelicalFlatSystemSpec system{};
    system.scan = regularHelicalScan(p);
    system.detector = flatDetector(p);
    system.volume = volumeGrid(p);
    std::vector<SConeProjGeomVec> geometry;
    buildHelicalFlatGeometry(system, geometry);
    return geometry;
}

inline std::vector<SConeProjGeomVec> staticFlat(const SHeliCTParam& p)
{
    SStaticFlatSystemSpec system{};
    system.scan = regularCircularScan(p);
    system.detector = flatDetector(p);
    system.volume = volumeGrid(p);
    std::vector<SConeProjGeomVec> geometry;
    buildStaticFlatGeometry(system, geometry);
    return geometry;
}

inline std::vector<SCylConeProjGeomVec> staticCyl(const SHeliCTParam& p,
    float radius_mm, float channel_angle_step_rad = 0.f)
{
    SStaticCylSystemSpec system{};
    system.scan = regularCircularScan(p);
    system.detector = cylDetector(p, radius_mm, channel_angle_step_rad);
    system.volume = volumeGrid(p);
    std::vector<SCylConeProjGeomVec> geometry;
    buildStaticCylGeometry(system, geometry);
    return geometry;
}

inline std::vector<SCylConeProjGeomVec> helicalCyl(const SHeliCTParam& p,
    float radius_mm, float channel_angle_step_rad = 0.f)
{
    SHelicalCylSystemSpec system{};
    system.scan = regularHelicalScan(p);
    system.detector = cylDetector(p, radius_mm, channel_angle_step_rad);
    system.volume = volumeGrid(p);
    std::vector<SCylConeProjGeomVec> geometry;
    buildHelicalCylGeometry(system, geometry);
    return geometry;
}

inline Helical::Wfbp::InputGeometry wfbpInput(const SHeliCTParam& p)
{
    Helical::Wfbp::InputGeometry input{};
    buildHelicalTrajectory(regularHelicalScan(p), input.trajectory);
    input.channels = p.iPU; input.rows = p.iPV;
    input.channel_spacing_mm = p.du_mm;
    input.row_spacing_mm = p.dv_mm;
    input.detector_pose = flatDetector(p).pose;
    input.views_per_turn = p.views_per_rot;
    return input;
}

inline SVolGeom volume(const SHeliCTParam& p)
{
    SVolGeom geometry{};
    buildVolumeGeometry(volumeGrid(p), geometry);
    return geometry;
}

} // namespace YK::TestGeometry
