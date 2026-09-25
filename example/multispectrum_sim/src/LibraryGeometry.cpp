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
        YK::EPipeline pipeline, YK::EProjectionModel projection_model, YK::EFdkFilter filter,
        GeometryUse use)
    {
        const auto& g = config.geometry.parameters;
        check(g.views > 0 && g.views_per_turn >= 2 &&
            (g.rotation_direction == 1 || g.rotation_direction == -1) &&
            g.detector_u > 0 && g.detector_v > 0 &&
            g.volume_x > 0 && g.volume_y > 0 && g.volume_z > 0,
            "geometry_config 尺寸必须为正");
        check(g.sid_mm > 0.0 && g.sdd_mm > g.sid_mm,
            "geometry_config 要求 sdd_mm > sid_mm > 0");

        YK::SSystemSpec system{};
        system.geometry.detector =
            config.geometry.kind == GeometryKind::CylCbct ||
            config.geometry.kind == GeometryKind::CylHelical
            ? YK::EDetectorKind::Cylindrical : YK::EDetectorKind::Flat;
        system.geometry.trajectory =
            config.geometry.kind == GeometryKind::FlatHelical ||
            config.geometry.kind == GeometryKind::CylHelical
            ? YK::ETrajectoryKind::Helical : YK::ETrajectoryKind::Circular;

        if (system.geometry.trajectory == YK::ETrajectoryKind::Circular) {
            auto& scan = system.geometry.circular;
            scan.total_views = g.views;
            scan.views_per_turn = g.views_per_turn;
            scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
            scan.rotation_direction = g.rotation_direction;
            scan.sid_mm = static_cast<float>(g.sid_mm);
            scan.sdd_mm = static_cast<float>(g.sdd_mm);
            scan.z_mm = static_cast<float>(g.start_z_mm);
            scan.source_offset_mm = make_float3(
                static_cast<float>(g.source_offset_x_mm),
                static_cast<float>(g.source_offset_y_mm),
                static_cast<float>(g.source_offset_z_mm));
        }
        else {
            auto& scan = system.geometry.helical;
            check(std::abs(g.pitch_mm_per_turn) > 0.0,
                "螺旋 geometry 必须提供非零 pitch_mm_per_turn");
            scan.total_views = g.views;
            scan.views_per_turn = g.views_per_turn;
            scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
            scan.rotation_direction = g.rotation_direction;
            scan.sid_mm = static_cast<float>(g.sid_mm);
            scan.sdd_mm = static_cast<float>(g.sdd_mm);
            scan.start_z_mm = static_cast<float>(g.start_z_mm);
            scan.pitch_mm_per_turn = static_cast<float>(g.pitch_mm_per_turn);
            scan.source_offset_mm = make_float3(
                static_cast<float>(g.source_offset_x_mm),
                static_cast<float>(g.source_offset_y_mm),
                static_cast<float>(g.source_offset_z_mm));
        }

        if (system.geometry.detector == YK::EDetectorKind::Flat) {
            auto& detector = system.geometry.flat_detector;
            detector.channels = g.detector_u;
            detector.rows = g.detector_v;
            detector.channel_size_mm = static_cast<float>(g.pixel_u_mm);
            detector.row_size_mm = static_cast<float>(g.pixel_v_mm);
            detector.pose.offset_unv_mm = make_float3(
                static_cast<float>(g.offset_u_mm),
                static_cast<float>(g.offset_n_mm),
                static_cast<float>(g.offset_v_mm));
        }
        else {
            auto& detector = system.geometry.cylindrical_detector;
            detector.channels = g.detector_u;
            detector.rows = g.detector_v;
            detector.channel_arc_mm = static_cast<float>(g.pixel_u_mm);
            detector.row_size_mm = static_cast<float>(g.pixel_v_mm);
            detector.curvature_radius_mm = static_cast<float>(g.sdd_mm);
            detector.pose.offset_unv_mm = make_float3(
                static_cast<float>(g.offset_u_mm),
                static_cast<float>(g.offset_n_mm),
                static_cast<float>(g.offset_v_mm));
        }

        const auto& volume_offset = use == GeometryUse::Reconstruction
            ? make_float3(static_cast<float>(g.reconstruction_offset_x_mm),
                static_cast<float>(g.reconstruction_offset_y_mm),
                static_cast<float>(g.reconstruction_offset_z_mm))
            : make_float3(static_cast<float>(g.phantom_offset_x_mm),
                static_cast<float>(g.phantom_offset_y_mm),
                static_cast<float>(g.phantom_offset_z_mm));
        const int volume_x = use == GeometryUse::Reconstruction ?
            g.reconstruction_volume_x : g.volume_x;
        const int volume_y = use == GeometryUse::Reconstruction ?
            g.reconstruction_volume_y : g.volume_y;
        const int volume_z = use == GeometryUse::Reconstruction ?
            g.reconstruction_volume_z : g.volume_z;
        const double voxel_x = use == GeometryUse::Reconstruction ?
            g.reconstruction_voxel_x_mm : g.voxel_x_mm;
        const double voxel_y = use == GeometryUse::Reconstruction ?
            g.reconstruction_voxel_y_mm : g.voxel_y_mm;
        const double voxel_z = use == GeometryUse::Reconstruction ?
            g.reconstruction_voxel_z_mm : g.voxel_z_mm;
        check(volume_x > 0 && volume_y > 0 && volume_z > 0 &&
            voxel_x > 0 && voxel_y > 0 && voxel_z > 0,
            "投影或重建体网格尺寸必须为正");
        system.geometry.volume = { volume_x, volume_y, volume_z,
            static_cast<float>(voxel_x), static_cast<float>(voxel_y),
            static_cast<float>(voxel_z), volume_offset };
        system.reconstruction.pipeline = pipeline;
        system.reconstruction.projection_model = projection_model;
        const auto& iterative = config.reconstruction.iterative;
        if (use == GeometryUse::Reconstruction && config.reconstruction.type == "iterative") {
            const bool siddon = iterative.projection_model == "siddon";
            system.reconstruction.projection_model = siddon
                ? YK::EProjectionModel::Siddon : YK::EProjectionModel::Joseph;
        }
        system.reconstruction.iterative.iterations = iterative.iterations;
        system.reconstruction.iterative.relaxation = static_cast<float>(iterative.relaxation);
        system.reconstruction.iterative.subsets = iterative.subsets;
        if (iterative.algorithm == "tigre_sart")
            system.reconstruction.tigre.method = YK::ETigreGradientMethodSpec::Sart;
        else if (iterative.algorithm == "tigre_sirt")
            system.reconstruction.tigre.method = YK::ETigreGradientMethodSpec::Sirt;
        else if (iterative.algorithm == "tigre_os_sart")
            system.reconstruction.tigre.method = YK::ETigreGradientMethodSpec::OsSart;
        else if (iterative.algorithm == "tigre_sart_tv")
            system.reconstruction.tigre.method = YK::ETigreGradientMethodSpec::AsdPocs;
        else if (iterative.algorithm == "tigre_os_sart_tv")
            system.reconstruction.tigre.method = YK::ETigreGradientMethodSpec::OsAsdPocs;
        system.reconstruction.tigre.tv_iterations = iterative.tv_iterations;
        system.reconstruction.tigre.tv_alpha = static_cast<float>(iterative.tv_alpha);
        system.reconstruction.tigre.tv_alpha_reduction = static_cast<float>(iterative.tv_alpha_reduction);
        system.reconstruction.tigre.maximum_update_ratio = static_cast<float>(iterative.maximum_update_ratio);
        system.reconstruction.tigre.non_negative = iterative.non_negative;
        system.reconstruction.fdk.filter = filter;
        return system;
    }
}
