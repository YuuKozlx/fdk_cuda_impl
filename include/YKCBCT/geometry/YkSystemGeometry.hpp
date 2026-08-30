#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"

namespace YK {

// 四种规则扫描系统均由 scan、detector 和 volume 三个正交部分组成。
// 调用方填写宏观系统参数，builder 负责生成角度和逐视图 vector geometry。
struct SStaticFlatSystemSpec {
    SRegularCircularScanSpec scan{};
    SFlatDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

struct SStaticCylSystemSpec {
    SRegularCircularScanSpec scan{};
    SCylDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

struct SHelicalFlatSystemSpec {
    SRegularHelicalScanSpec scan{};
    SFlatDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

// 曲率只存在于 detector 中，不会与 scan.sdd_mm 混为一个字段；
// 柱面迭代重建允许曲率半径与 SDD 不相等。
struct SHelicalCylSystemSpec {
    SRegularHelicalScanSpec scan{};
    SCylDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

namespace SystemGeometryDetail {

template <typename Scan>
inline bool validateRegularScan(const Scan& scan)
{
    return scan.total_views > 0 && scan.views_per_turn >= 2 &&
        (scan.rotation_direction == 1 || scan.rotation_direction == -1) &&
        scan.sid_mm > 0.f && scan.sdd_mm > scan.sid_mm &&
        std::isfinite(scan.start_angle_rad) &&
        std::isfinite(scan.source_offset_mm.x) &&
        std::isfinite(scan.source_offset_mm.y) &&
        std::isfinite(scan.source_offset_mm.z);
}

template <typename Scan>
inline void buildRegularAngles(const Scan& scan, std::vector<float>& angles)
{
    constexpr float kTwoPi = 6.28318530717958647692f;
    const float step = static_cast<float>(scan.rotation_direction) *
        kTwoPi / static_cast<float>(scan.views_per_turn);
    angles.resize(scan.total_views);
    for (int view = 0; view < scan.total_views; ++view)
        angles[view] = scan.start_angle_rad + view * step;
}

} // namespace SystemGeometryDetail

inline bool buildCircularTrajectory(const SRegularCircularScanSpec& scan,
    SCircularTrajectorySpec& trajectory)
{
    if (!SystemGeometryDetail::validateRegularScan(scan) ||
        !std::isfinite(scan.z_mm)) return false;
    trajectory = {};
    SystemGeometryDetail::buildRegularAngles(scan, trajectory.angles_rad);
    trajectory.sid_mm = scan.sid_mm;
    trajectory.sdd_mm = scan.sdd_mm;
    trajectory.z_mm = scan.z_mm;
    trajectory.source_offset_mm = scan.source_offset_mm;
    return true;
}

inline bool buildHelicalTrajectory(const SRegularHelicalScanSpec& scan,
    SHelicalTrajectorySpec& trajectory)
{
    if (!SystemGeometryDetail::validateRegularScan(scan) ||
        !std::isfinite(scan.start_z_mm) ||
        !std::isfinite(scan.pitch_mm_per_turn) ||
        scan.pitch_mm_per_turn == 0.f) return false;

    constexpr float kTwoPi = 6.28318530717958647692f;
    trajectory = {};
    SystemGeometryDetail::buildRegularAngles(scan, trajectory.angles_rad);
    trajectory.sid_mm = scan.sid_mm;
    trajectory.sdd_mm = scan.sdd_mm;
    // 底层显式轨迹按绝对角度计算 z。换算截距并吸收旋转方向，保证
    // pitch 的正负只表示物理进床方向，不随角度增减方向翻转。
    trajectory.pitch_mm_per_turn = scan.pitch_mm_per_turn *
        static_cast<float>(scan.rotation_direction);
    trajectory.start_z_mm = scan.start_z_mm -
        trajectory.pitch_mm_per_turn * scan.start_angle_rad / kTwoPi;
    trajectory.source_offset_mm = scan.source_offset_mm;
    return true;
}

inline bool buildVolumeGeometry(const SVolumeGridSpec& volume,
    SVolGeom& geometry)
{
    if (volume.nx <= 0 || volume.ny <= 0 || volume.nz <= 0 ||
        !(volume.voxel_x_mm > 0.f) || !(volume.voxel_y_mm > 0.f) ||
        !(volume.voxel_z_mm > 0.f) ||
        !std::isfinite(volume.center_mm.x) ||
        !std::isfinite(volume.center_mm.y) ||
        !std::isfinite(volume.center_mm.z)) return false;
    geometry = SVolGeom::make_centered(volume.nx, volume.ny, volume.nz,
        volume.voxel_x_mm, volume.voxel_y_mm, volume.voxel_z_mm);
    geometry.center = volume.center_mm;
    return true;
}

template <typename System, typename Trajectory, typename Geometry>
inline bool buildSystemGeometry(const System& system, Trajectory& trajectory,
    std::vector<Geometry>& geometry, SVolGeom& volume)
{
    return buildVolumeGeometry(system.volume, volume) &&
        buildProjectionGeometry(trajectory, system.detector, geometry);
}

inline bool buildStaticFlatGeometry(const SStaticFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SCircularTrajectorySpec trajectory{};
    return buildCircularTrajectory(system.scan, trajectory) &&
        buildSystemGeometry(system, trajectory, geometry, volume);
}
inline bool buildStaticFlatGeometry(const SStaticFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry)
{ SVolGeom volume{}; return buildStaticFlatGeometry(system, geometry, volume); }

inline bool buildStaticCylGeometry(const SStaticCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SCircularTrajectorySpec trajectory{};
    return buildCircularTrajectory(system.scan, trajectory) &&
        buildSystemGeometry(system, trajectory, geometry, volume);
}
inline bool buildStaticCylGeometry(const SStaticCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry)
{ SVolGeom volume{}; return buildStaticCylGeometry(system, geometry, volume); }

inline bool buildHelicalFlatGeometry(const SHelicalFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SHelicalTrajectorySpec trajectory{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildSystemGeometry(system, trajectory, geometry, volume);
}
inline bool buildHelicalFlatGeometry(const SHelicalFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry)
{ SVolGeom volume{}; return buildHelicalFlatGeometry(system, geometry, volume); }

inline bool buildHelicalCylGeometry(const SHelicalCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SHelicalTrajectorySpec trajectory{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildSystemGeometry(system, trajectory, geometry, volume);
}
inline bool buildHelicalCylGeometry(const SHelicalCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry)
{ SVolGeom volume{}; return buildHelicalCylGeometry(system, geometry, volume); }

} // namespace YK
