#pragma once

#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"
#include "YKCBCT/geometry/YkRigidTransform.hpp"

namespace YK {

enum class EDetectorKind : int32_t { Flat, Cylindrical };
enum class ETrajectoryKind : int32_t { Circular, Helical };

// 库对外唯一的宏观系统配置。探测器和轨迹通过两个枚举组合成 Flat/Cyl
// 与 Circular/Helical 四种系统；builder 负责只读取对应分支并生成规范几何。
// 该结构不包含算法、CUDA 资源或预计算结果。
struct SSystemConfig {
    EDetectorKind detector = EDetectorKind::Flat;
    ETrajectoryKind trajectory = ETrajectoryKind::Circular;
    SRegularCircularScanSpec circular{};
    SRegularHelicalScanSpec helical{};
    SFlatDetectorSpec flat_detector{};
    SCylDetectorSpec cylindrical_detector{};
    SVolumeGridSpec volume{};
    SForwardProjectionPose forward_projection_pose{};
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

// 统一入口：调用方只提交一份 SSystemConfig，所有后端共享这次构造结果。
inline bool buildSystemGeometry(const SSystemConfig& system,
    std::vector<SConeProjGeomVec>& flat_geometry,
    std::vector<SCylConeProjGeomVec>& cyl_geometry,
    SVolGeom& volume)
{
    flat_geometry.clear(); cyl_geometry.clear();
    if (system.detector != EDetectorKind::Flat &&
        system.detector != EDetectorKind::Cylindrical) return false;
    if (system.trajectory != ETrajectoryKind::Circular &&
        system.trajectory != ETrajectoryKind::Helical) return false;
    if (!buildVolumeGeometry(system.volume, volume)) return false;
    if (system.detector == EDetectorKind::Flat) {
        if (system.trajectory == ETrajectoryKind::Circular) {
            SCircularTrajectorySpec trajectory{};
            return buildCircularTrajectory(system.circular, trajectory) &&
                buildProjectionGeometry(trajectory, system.flat_detector, flat_geometry);
        }
        SHelicalTrajectorySpec trajectory{};
        return buildHelicalTrajectory(system.helical, trajectory) &&
            buildProjectionGeometry(trajectory, system.flat_detector, flat_geometry);
    }
    if (system.trajectory == ETrajectoryKind::Circular) {
        SCircularTrajectorySpec trajectory{};
        return buildCircularTrajectory(system.circular, trajectory) &&
            buildProjectionGeometry(trajectory, system.cylindrical_detector, cyl_geometry);
    }
    SHelicalTrajectorySpec trajectory{};
    return buildHelicalTrajectory(system.helical, trajectory) &&
        buildProjectionGeometry(trajectory, system.cylindrical_detector, cyl_geometry);
}

inline bool buildSystemGeometry(const SSystemConfig& system,
    std::vector<SConeProjGeomVec>& flat_geometry,
    std::vector<SCylConeProjGeomVec>& cyl_geometry)
{ SVolGeom volume{}; return buildSystemGeometry(system, flat_geometry, cyl_geometry, volume); }

} // namespace YK
