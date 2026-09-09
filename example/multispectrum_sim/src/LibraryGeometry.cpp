#include "LibraryGeometry.hpp"

#include <cmath>
#include <stdexcept>

namespace yk::spectral {
namespace {
void check(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
}

YK::SSystemSpec makeLibrarySystem(const SimulationConfig& config,
    YK::EPipeline pipeline, YK::ETask forward_projector, YK::EFdkFilter filter)
{
    const auto& g = config.geometry_config;
    check(g.views > 0 && g.detector_u > 0 && g.detector_v > 0 &&
        g.volume_x > 0 && g.volume_y > 0 && g.volume_z > 0,
        "geometry_config 尺寸必须为正");
    check(g.sid_mm > 0.0 && g.sdd_mm > g.sid_mm,
        "geometry_config 要求 sdd_mm > sid_mm > 0");

    YK::SSystemSpec system{};
    system.geometry.detector =
        config.geometry == GeometryKind::CylCbct ||
        config.geometry == GeometryKind::CylHelical
        ? YK::EDetectorKind::Cylindrical : YK::EDetectorKind::Flat;
    system.geometry.trajectory =
        config.geometry == GeometryKind::FlatHelical ||
        config.geometry == GeometryKind::CylHelical
        ? YK::ETrajectoryKind::Helical : YK::ETrajectoryKind::Circular;

    if (system.geometry.trajectory == YK::ETrajectoryKind::Circular) {
        auto& scan = system.geometry.circular;
        scan.total_views = g.views;
        scan.views_per_turn = g.views;
        scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
        scan.sid_mm = static_cast<float>(g.sid_mm);
        scan.sdd_mm = static_cast<float>(g.sdd_mm);
        scan.z_mm = static_cast<float>(g.start_z_mm);
    } else {
        auto& scan = system.geometry.helical;
        check(std::abs(g.pitch_mm_per_turn) > 0.0,
            "螺旋 geometry 必须提供非零 pitch_mm_per_turn");
        scan.total_views = g.views;
        scan.views_per_turn = g.views;
        scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
        scan.sid_mm = static_cast<float>(g.sid_mm);
        scan.sdd_mm = static_cast<float>(g.sdd_mm);
        scan.start_z_mm = static_cast<float>(g.start_z_mm);
        scan.pitch_mm_per_turn = static_cast<float>(g.pitch_mm_per_turn);
    }

    if (system.geometry.detector == YK::EDetectorKind::Flat) {
        auto& detector = system.geometry.flat_detector;
        detector.channels = g.detector_u;
        detector.rows = g.detector_v;
        detector.channel_size_mm = static_cast<float>(g.pixel_u_mm);
        detector.row_size_mm = static_cast<float>(g.pixel_v_mm);
        detector.pose.offset_unv_mm = make_float3(
            static_cast<float>(g.offset_u_mm), 0.f,
            static_cast<float>(g.offset_v_mm));
    } else {
        auto& detector = system.geometry.cylindrical_detector;
        detector.channels = g.detector_u;
        detector.rows = g.detector_v;
        detector.channel_arc_mm = static_cast<float>(g.pixel_u_mm);
        detector.row_size_mm = static_cast<float>(g.pixel_v_mm);
        detector.curvature_radius_mm = static_cast<float>(g.sdd_mm);
        detector.pose.offset_unv_mm = make_float3(
            static_cast<float>(g.offset_u_mm), 0.f,
            static_cast<float>(g.offset_v_mm));
    }

    system.geometry.volume = {g.volume_x, g.volume_y, g.volume_z,
        static_cast<float>(g.voxel_x_mm), static_cast<float>(g.voxel_y_mm),
        static_cast<float>(g.voxel_z_mm), make_float3(0.f, 0.f, 0.f)};
    system.reconstruction.pipeline = pipeline;
    system.reconstruction.forward_projector = forward_projector;
    system.reconstruction.fdk.filter = filter;
    return system;
}
}
