#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"

int main_geometry_builder_four_modes()
{
    using namespace YK;
    std::vector<float> angles{0.f, 1.f, 2.f, 3.f};
    SCircularTrajectorySpec circular{};
    circular.angles_rad = angles; circular.sid_mm = 500.f;
    circular.sdd_mm = 1000.f; circular.z_mm = 7.f;
    SHelicalTrajectorySpec helical{};
    helical.angles_rad = angles; helical.sid_mm = 500.f;
    helical.sdd_mm = 1000.f; helical.start_z_mm = 7.f;
    helical.pitch_mm_per_turn = 40.f;
    SFlatDetectorSpec flat{}; flat.channels = 64; flat.rows = 32;
    flat.channel_size_mm = 1.f; flat.row_size_mm = 1.f;
    SCylDetectorSpec cyl{}; cyl.channels = 64; cyl.rows = 32;
    cyl.channel_arc_mm = 1.f; cyl.row_size_mm = 1.f;
    cyl.curvature_radius_mm = 900.f;
    std::vector<SScannerViewFrame> frames;
    std::vector<SConeProjGeomVec> static_flat, helical_flat;
    std::vector<SCylConeProjGeomVec> static_cyl, helical_cyl;
    if (!buildProjectionGeometry(circular, flat, static_flat) ||
        !buildProjectionGeometry(helical, flat, helical_flat) ||
        !buildProjectionGeometry(circular, cyl, static_cyl) ||
        !buildProjectionGeometry(helical, cyl, helical_cyl)) return 1;
    if (static_flat.size() != angles.size() || helical_flat.size() != angles.size() ||
        static_cyl.size() != angles.size() || helical_cyl.size() != angles.size()) return 1;
    if (std::fabs(static_flat.front().src.z - static_flat.back().src.z) > 1e-5f ||
        std::fabs(static_cyl.front().source.z - static_cyl.back().source.z) > 1e-5f)
        return 1;
    if (std::fabs(helical_cyl.back().source.z - helical_cyl.front().source.z -
            helical.pitch_mm_per_turn * (angles.back() - angles.front()) /
                (2.f * 3.14159265358979323846f)) > 1e-4f) return 1;
    if (std::fabs(cylDetectorRadius(static_cyl.front()) - 900.f) > 1e-5f)
        return 1;
    SCircularTrajectorySpec invalid = circular;
    invalid.sdd_mm = invalid.sid_mm;
    if (buildProjectionGeometry(invalid, flat, static_flat)) return 1;
    invalid = circular;
    invalid.angles_rad = {0.f, NAN};
    if (buildProjectionGeometry(invalid, flat, static_flat)) return 1;
    SFlatDetectorSpec invalid_detector = flat;
    invalid_detector.channel_size_mm = 0.f;
    if (buildProjectionGeometry(circular, invalid_detector, static_flat)) return 1;
    SHelicalTrajectorySpec invalid_heli = helical;
    invalid_heli.pitch_mm_per_turn = 0.f;
    if (buildProjectionGeometry(invalid_heli, flat, helical_flat)) return 1;
    SCylDetectorSpec invalid_cyl = cyl;
    invalid_cyl.curvature_radius_mm = -1.f;
    if (buildProjectionGeometry(circular, invalid_cyl, static_cyl)) return 1;

    // 四种宏观系统参数入口均自动生成规则角度；调用方无需逐帧填写
    // SConeProjGeomVec/SCylConeProjGeomVec。
    SStaticFlatSystemSpec static_flat_system{};
    static_flat_system.scan.total_views = 6;
    static_flat_system.scan.views_per_turn = 8;
    static_flat_system.scan.start_angle_rad = 0.25f;
    static_flat_system.scan.rotation_direction = -1;
    static_flat_system.scan.sid_mm = 500.f;
    static_flat_system.scan.sdd_mm = 1000.f;
    static_flat_system.scan.z_mm = 7.f;
    static_flat_system.detector = flat;
    static_flat_system.volume = {16, 16, 8, 1.f, 1.f, 1.f,
        make_float3(3.f, 0.f, 2.f)};
    SVolGeom static_flat_volume{};
    if (!buildStaticFlatGeometry(static_flat_system, static_flat,
            static_flat_volume) || static_flat.size() != 6 ||
        std::fabs(static_flat.front().src.z - 7.f) > 1e-5f ||
        !(static_flat[1].angle.x < static_flat[0].angle.x))
        return 1;

    SStaticCylSystemSpec static_cyl_system{};
    static_cyl_system.scan = static_flat_system.scan;
    static_cyl_system.detector = cyl;
    static_cyl_system.volume = static_flat_system.volume;
    SVolGeom static_cyl_volume{};
    if (!buildStaticCylGeometry(static_cyl_system, static_cyl,
            static_cyl_volume) || static_cyl.size() != 6 ||
        std::fabs(static_cyl.front().source.z - 7.f) > 1e-5f ||
        static_cyl_volume.Nx != static_flat_volume.Nx)
        return 1;

    SHelicalFlatSystemSpec flat_system{};
    flat_system.scan.total_views = 8;
    flat_system.scan.views_per_turn = 4;
    flat_system.scan.sid_mm = 500.f;
    flat_system.scan.sdd_mm = 1000.f;
    flat_system.scan.start_angle_rad = 0.5f;
    flat_system.scan.start_z_mm = -12.f;
    flat_system.scan.pitch_mm_per_turn = 20.f;
    flat_system.detector = flat;
    flat_system.volume = {16, 16, 8, 1.f, 1.f, 1.f, make_float3(3.f, 0.f, 2.f)};
    SVolGeom flat_system_volume{};
    if (!buildHelicalFlatGeometry(flat_system, helical_flat,
            flat_system_volume) ||
        helical_flat.size() != 8 ||
        std::fabs(helical_flat.front().src.z -
            (-12.f + flat_system.scan.source_offset_mm.z)) > 1e-5f ||
        std::fabs(flat_system_volume.center.x - 3.f) > 1e-5f)
        return 1;

    SVolGeom volume_geometry{};
    if (!buildVolumeGeometry(flat_system.volume, volume_geometry) ||
        std::fabs(volume_geometry.center.z - 2.f) > 1e-5f)
        return 1;

    SHelicalCylSystemSpec cyl_system{};
    cyl_system.scan = flat_system.scan;
    cyl_system.detector = cyl;
    cyl_system.volume = flat_system.volume;
    SVolGeom cyl_system_volume{};
    if (!buildHelicalCylGeometry(cyl_system, helical_cyl,
            cyl_system_volume) || helical_cyl.size() != 8 ||
        cyl_system_volume.Nz != flat_system.volume.nz)
        return 1;

    static_flat_system.scan.views_per_turn = 1;
    if (buildStaticFlatGeometry(static_flat_system, static_flat)) return 1;

    SStaticFlatReconstructionRequest request{};
    request.system = static_flat_system;
    request.system.scan.views_per_turn = 8;
    request.reconstruction.parker.mode = EParkerMode::Auto;
    if (!resolveParkerEnabled(request) ||
        request.reconstruction.pipeline != EPipeline::FDK)
        return 1;
    request.system.scan.total_views = request.system.scan.views_per_turn;
    if (resolveParkerEnabled(request)) return 1;
    return 0;
}
