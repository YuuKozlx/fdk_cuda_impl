#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"

namespace YK {

// 面向调用方的平板螺旋系统参数。scan、detector 和 volume 分别描述
// 采集轨迹、探测器和输出网格；用户不需要构造任何逐视图 vector geometry。
struct SHelicalFlatSystemSpec {
    SRegularHelicalScanSpec scan{};
    SFlatDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

// 面向调用方的柱面螺旋系统参数。曲率只存在于 detector 中，不会与
// scan.sdd_mm 混为同一个字段；迭代重建允许二者不相等。
struct SHelicalCylSystemSpec {
    SRegularHelicalScanSpec scan{};
    SCylDetectorSpec detector{};
    SVolumeGridSpec volume{};
};

inline bool buildHelicalTrajectory(const SRegularHelicalScanSpec& scan,
    SHelicalTrajectorySpec& trajectory)
{
    if (scan.total_views <= 0 || scan.views_per_turn < 2 ||
        (scan.rotation_direction != 1 && scan.rotation_direction != -1) ||
        !(scan.sid_mm > 0.f) || !(scan.sdd_mm > scan.sid_mm) ||
        !std::isfinite(scan.start_angle_rad) ||
        !std::isfinite(scan.start_z_mm) ||
        !std::isfinite(scan.pitch_mm_per_turn) ||
        scan.pitch_mm_per_turn == 0.f) return false;

    constexpr float kTwoPi = 6.28318530717958647692f;
    const float step = static_cast<float>(scan.rotation_direction) *
        kTwoPi / static_cast<float>(scan.views_per_turn);
    trajectory = {};
    trajectory.angles_rad.resize(scan.total_views);
    for (int view = 0; view < scan.total_views; ++view)
        trajectory.angles_rad[view] = scan.start_angle_rad + view * step;
    trajectory.sid_mm = scan.sid_mm;
    trajectory.sdd_mm = scan.sdd_mm;
    // 底层显式轨迹按绝对角度计算 z。将宏观“第一帧 Z”换算成零角度
    // 截距，并吸收旋转方向，使物理进床方向不随角度增减方向翻转。
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

inline bool buildHelicalFlatGeometry(const SHelicalFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry)
{
    SHelicalTrajectorySpec trajectory{};
    SVolGeom volume{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildVolumeGeometry(system.volume, volume) &&
        buildProjectionGeometry(trajectory, system.detector, geometry);
}

// 同时导出投影几何和体积网格，适合重建 pipeline 的一次性初始化。
// 体积中心保留为世界坐标，不会被误当作旋转中心。
inline bool buildHelicalFlatGeometry(const SHelicalFlatSystemSpec& system,
    std::vector<SConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SHelicalTrajectorySpec trajectory{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildVolumeGeometry(system.volume, volume) &&
        buildProjectionGeometry(trajectory, system.detector, geometry);
}

inline bool buildHelicalCylGeometry(const SHelicalCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry)
{
    SHelicalTrajectorySpec trajectory{};
    SVolGeom volume{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildVolumeGeometry(system.volume, volume) &&
        buildProjectionGeometry(trajectory, system.detector, geometry);
}

// 柱面系统的完整初始化入口；曲率仍由 detector 描述，体积由 volume 返回。
inline bool buildHelicalCylGeometry(const SHelicalCylSystemSpec& system,
    std::vector<SCylConeProjGeomVec>& geometry, SVolGeom& volume)
{
    SHelicalTrajectorySpec trajectory{};
    return buildHelicalTrajectory(system.scan, trajectory) &&
        buildVolumeGeometry(system.volume, volume) &&
        buildProjectionGeometry(trajectory, system.detector, geometry);
}

} // namespace YK
