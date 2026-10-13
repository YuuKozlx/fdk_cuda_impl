#include "LibraryGeometry.hpp"
#include <limits>
#include "NonStandardFpBp/Geometry/YkTrajectorySampling.hpp"
#include "NonStandardFpBp/NikonTiltedCt/YkNikonTiltedCtGeometry.hpp"
#include "NonStandardFpBp/SourceOrbitTomography/YkSourceOrbitTomography.hpp"

#include <cmath>
#include <stdexcept>

namespace yk::spectral {
    namespace {
        void check(bool condition, const char* message)
        {
            if (!condition) throw std::runtime_error(message);
        }

        YK::EForwardProjector parseForwardProjector(const std::string& value)
        {
            if (value == "auto") return YK::EForwardProjector::Automatic;
            if (value == "siddon") return YK::EForwardProjector::Siddon;
            if (value == "joseph") return YK::EForwardProjector::Joseph;
            if (value == "footprint") return YK::EForwardProjector::SeparableFootprint;
            throw std::runtime_error("Unsupported forward_projector: " + value);
        }

        YK::EBackProjector parseBackProjector(const std::string& value)
        {
            if (value == "auto") return YK::EBackProjector::Automatic;
            if (value == "fdk_pseudo_matched") return YK::EBackProjector::FdkPseudoMatched;
            if (value == "siddon_ray") return YK::EBackProjector::SiddonRayDriven;
            if (value == "joseph") return YK::EBackProjector::Joseph;
            if (value == "joseph_v3") return YK::EBackProjector::JosephV3;
            if (value == "footprint") return YK::EBackProjector::SeparableFootprint;
            throw std::runtime_error("Unsupported back_projector: " + value);
        }
    }

    YK::SSystemSpec makeLibrarySystem(const SimulationConfig& config,
        YK::EPipeline pipeline, YK::EFdkFilter filter,
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
        system.geometry.trajectory = config.geometry.kind == GeometryKind::PlanarSourceOrbit
            ? YK::ETrajectoryKind::PlanarSourceOrbit
            : config.geometry.kind == GeometryKind::NikonTilted
            ? YK::ETrajectoryKind::NikonTilted
            :
            config.geometry.kind == GeometryKind::FlatHelical ||
            config.geometry.kind == GeometryKind::CylHelical
            ? YK::ETrajectoryKind::Helical : YK::ETrajectoryKind::Circular;

        if (system.geometry.trajectory == YK::ETrajectoryKind::PlanarSourceOrbit) {
            auto& scan = system.nonstandard.planar_source_orbit;
            scan.total_views = g.views;
            scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
            scan.scan_range_rad = static_cast<float>(g.scan_range_rad);
            scan.rotation_direction = g.rotation_direction;
            scan.sid_mm = static_cast<float>(g.sid_mm);
            scan.detector_y_mm = static_cast<float>(g.planar_detector_y_mm);
            scan.source_orbit_radius_mm = static_cast<float>(g.source_orbit_radius_mm);
            scan.source_lateral_offset_mm = static_cast<float>(g.source_lateral_offset_mm);
            scan.reconstruction_channels = g.reconstruction_detector_u;
            scan.reconstruction_rows = g.reconstruction_detector_v;
            scan.resample_mode = g.planar_resample_mode == "in_plane_rotated"
                ? YK::EPlanarResampleMode::InPlaneRotated
                : YK::EPlanarResampleMode::PerspectiveOrthogonal;
        }
        else if (system.geometry.trajectory == YK::ETrajectoryKind::NikonTilted) {
            auto& scan = system.nonstandard.nikon_tilted;
            scan.total_views = g.views;
            scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
            scan.scan_range_rad = static_cast<float>(g.scan_range_rad);
            scan.rotation_direction = g.rotation_direction;
            scan.sid_mm = static_cast<float>(g.sid_mm);
            scan.sdd_mm = static_cast<float>(g.sdd_mm);
            scan.tilt_rad = static_cast<float>(g.nikon_tilt_rad);
            scan.tilt_axis_azimuth_rad = static_cast<float>(g.nikon_tilt_axis_azimuth_rad);
            scan.source_offset_mm = make_float3(static_cast<float>(g.source_offset_x_mm),
                static_cast<float>(g.source_offset_y_mm),
                static_cast<float>(g.source_offset_z_mm));
            scan.rotation_axis_pivot_mm = make_float3(
                static_cast<float>(g.rotation_axis_pivot_x_mm),
                static_cast<float>(g.rotation_axis_pivot_y_mm),
                static_cast<float>(g.rotation_axis_pivot_z_mm));
        }
        else if (system.geometry.trajectory == YK::ETrajectoryKind::Circular) {
            auto& scan = system.geometry.circular;
            scan.total_views = g.views;
            scan.views_per_turn = g.views_per_turn;
            scan.start_angle_rad = static_cast<float>(g.start_angle_rad);
            scan.rotation_direction = g.rotation_direction;
            scan.sid_mm = static_cast<float>(g.sid_mm);
            scan.sdd_mm = static_cast<float>(g.sdd_mm);
            scan.z_mm = static_cast<float>(g.start_z_mm);
            scan.source_offset_mm = make_float3(
                static_cast<float>(g.source_offset_y_mm),
                static_cast<float>(-g.source_offset_x_mm),
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
            // Simulation pitch follows the source angle; SDK pitch is physical bed travel.
            scan.pitch_mm_per_turn = -static_cast<float>(g.pitch_mm_per_turn) * g.rotation_direction;
            scan.source_offset_mm = make_float3(
                static_cast<float>(g.source_offset_y_mm),
                static_cast<float>(-g.source_offset_x_mm),
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
            detector.pose.tilt_u_rad = static_cast<float>(g.tilt_u_rad);
            detector.pose.tilt_v_rad = static_cast<float>(g.tilt_v_rad);
            detector.pose.tilt_n_rad = static_cast<float>(g.tilt_n_rad);
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
            detector.pose.tilt_u_rad = static_cast<float>(g.tilt_u_rad);
            detector.pose.tilt_v_rad = static_cast<float>(g.tilt_v_rad);
            detector.pose.tilt_n_rad = static_cast<float>(g.tilt_n_rad);
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
        const auto& iterative = config.reconstruction.iterative;
        if (use == GeometryUse::Reconstruction && config.reconstruction.type == "iterative") {
            system.reconstruction.forward_projector =
                parseForwardProjector(iterative.forward_projector);
            system.reconstruction.back_projector =
                parseBackProjector(iterative.back_projector);
        }
        system.reconstruction.iterative.iterations = iterative.iterations;
        system.reconstruction.iterative.relaxation = static_cast<float>(iterative.relaxation);
        system.reconstruction.iterative.subsets = iterative.subsets;
        if(pipeline==YK::EPipeline::TigreGradient)
            system.reconstruction.tigre.block_size=std::max(1,(g.views+std::max(1,iterative.subsets)-1)/std::max(1,iterative.subsets));
        if(use==GeometryUse::Reconstruction&&config.reconstruction.type=="statistical"){
            const auto& s=config.reconstruction.statistical;auto& out=system.statistical;
            system.reconstruction.forward_projector=parseForwardProjector(s.forward_projector);
            system.reconstruction.back_projector=parseBackProjector(s.back_projector);
            out.data_fidelity.scale=float(s.data_fidelity.scale);
            out.data_fidelity.normalize_by_ray_count=s.data_fidelity.normalize_by_ray_count;
            out.data_fidelity.require_verified_adjoint=s.data_fidelity.require_verified_adjoint;
            out.data_fidelity.use_projection_weights=!s.data_fidelity.weights_file.empty();
            out.regularization.plugin_id=s.regularization.plugin.c_str();out.regularization.strength=float(s.regularization.strength);
            out.regularization.shape=float(s.regularization.shape);out.regularization.dimensions=s.regularization.dimensions;
            out.optimizer.kind=s.optimizer.type=="fista"?YK::EStatisticalOptimizer::Fista:(s.optimizer.type=="raw_gradient_descent"||s.optimizer.type=="gradient_descent")?YK::EStatisticalOptimizer::GradientDescent:YK::EStatisticalOptimizer::RegularizedAlgebraic;
            out.optimizer.iterations=s.optimizer.iterations;out.optimizer.step_size=float(s.optimizer.step_size);
            out.optimizer.backtracking=s.optimizer.backtracking;out.optimizer.max_backtracking=s.optimizer.max_backtracking;
            out.optimizer.step_shrink=float(s.optimizer.step_shrink);out.optimizer.relative_tolerance=float(s.optimizer.relative_tolerance);
            out.optimizer.minimum_iterations=s.optimizer.minimum_iterations;
            out.constraints.lower_bound=s.non_negative?0.f:-std::numeric_limits<float>::max();
        }
        if (use == GeometryUse::Reconstruction && config.reconstruction.type == "iterative") {
            const auto& tv=iterative.regularization;
            system.regularization.plugin_id=tv.plugin.c_str();
            system.regularization.strength=float(tv.strength);
            system.regularization.huber_delta=float(tv.huber_delta);
            system.regularization.step_size=float(tv.step_size);
            system.regularization.inner_iterations=tv.inner_iterations;
            system.regularization.dimensions=tv.dimensions;
            system.regularization.epsilon=float(tv.epsilon);
            system.regularization.strength_reduction=float(tv.strength_reduction);
            system.regularization.non_negative=iterative.non_negative;
        }
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
        system.reconstruction.fdk.filter.type = filter;
        auto& shared=system.reconstruction.fdk.filter;
        shared.cutoff=float(config.reconstruction.analytic.cutoff);
        shared.gain=float(config.reconstruction.analytic.gain);
        shared.butterworth_order=float(config.reconstruction.analytic.butterworth_order);
        shared.kaiser_beta=float(config.reconstruction.analytic.kaiser_beta);
        shared.tukey_alpha=float(config.reconstruction.analytic.tukey_alpha);
        system.reconstruction.wfbp.filter=shared;
        return system;
    }

    std::vector<YK::SConeProjGeomVec> buildNonStandardProjectionViews(
        const SimulationConfig& config)
    {
        if (config.geometry.kind != GeometryKind::PlanarSourceOrbit &&
            config.geometry.kind != GeometryKind::NikonTilted)
            return {};
        const auto system = makeLibrarySystem(config, YK::EPipeline::ForwardProjection,
            YK::EFdkFilter::RamLak,
            GeometryUse::PhantomProjection);
        YK::NonStandardFpBp::TrajectorySampling sampling{};
        if (config.geometry.kind == GeometryKind::PlanarSourceOrbit) {
            const auto& spec = system.nonstandard.planar_source_orbit;
            sampling = {spec.total_views, spec.start_angle_rad,
                spec.scan_range_rad, static_cast<float>(spec.rotation_direction)};
        } else {
            const auto& spec = system.nonstandard.nikon_tilted;
            sampling = {spec.total_views, spec.start_angle_rad,
                spec.scan_range_rad, static_cast<float>(spec.rotation_direction)};
        }
        std::vector<float> angles;
        check(YK::NonStandardFpBp::buildUniformTrajectoryParameters(sampling, angles),
            "invalid non-standard trajectory sampling");
        std::vector<YK::SConeProjGeomVec> geometry;
        if (config.geometry.kind == GeometryKind::PlanarSourceOrbit) {
            const auto& g = config.geometry.parameters;
            YK::NonStandardFpBp::SourceOrbitGeometryConfig c{};
            c.detector_u = g.detector_u; c.detector_v = g.detector_v;
            c.detector_pixel_u_mm = static_cast<float>(g.pixel_u_mm);
            c.detector_pixel_v_mm = static_cast<float>(g.pixel_v_mm);
            c.sid_mm = static_cast<float>(g.sid_mm);
            c.detector_y_mm = static_cast<float>(g.planar_detector_y_mm);
            c.source_orbit_radius_mm = static_cast<float>(g.source_orbit_radius_mm);
            c.source_lateral_offset_mm = static_cast<float>(g.source_lateral_offset_mm);
            c.detector_pose = system.geometry.flat_detector.pose;
            check(YK::NonStandardFpBp::buildSourceOrbitGeometry(c, angles, geometry),
                "failed to build planar source-orbit geometry");
        } else {
            const auto& spec = system.nonstandard.nikon_tilted;
            const auto& g = config.geometry.parameters;
            YK::NonStandardFpBp::NikonTiltedCt::GeometryConfig c{};
            c.detector_u = g.detector_u; c.detector_v = g.detector_v;
            c.detector_pixel_u_mm = static_cast<float>(g.pixel_u_mm);
            c.detector_pixel_v_mm = static_cast<float>(g.pixel_v_mm);
            c.sid_mm = spec.sid_mm; c.sdd_mm = spec.sdd_mm;
            c.tilt_rad = spec.tilt_rad;
            c.tilt_axis_azimuth_rad = spec.tilt_axis_azimuth_rad;
            c.source = make_float3(spec.source_offset_mm.x,
                -spec.sid_mm + spec.source_offset_mm.y,
                spec.source_offset_mm.z);
            c.rotation_axis_pivot = spec.rotation_axis_pivot_mm;
            c.detector_pose = system.geometry.flat_detector.pose;
            std::vector<YK::NonStandardFpBp::NonStandardViewPose> poses;
            check(YK::NonStandardFpBp::NikonTiltedCt::buildTiltedAxisGeometry(
                c, angles, poses, geometry), "failed to build Nikon tilted geometry");
        }
        return geometry;
    }
}
