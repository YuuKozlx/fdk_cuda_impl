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
    SSystemConfig unified{};
    unified.circular.total_views = 6;
    unified.circular.views_per_turn = 8;
    unified.circular.start_angle_rad = 0.25f;
    unified.circular.rotation_direction = -1;
    unified.circular.sid_mm = 500.f;
    unified.circular.sdd_mm = 1000.f;
    unified.circular.z_mm = 7.f;
    unified.helical.total_views = 8;
    unified.helical.views_per_turn = 4;
    unified.helical.sid_mm = 500.f;
    unified.helical.sdd_mm = 1000.f;
    unified.helical.start_angle_rad = 0.5f;
    unified.helical.start_z_mm = -12.f;
    unified.helical.pitch_mm_per_turn = 20.f;
    unified.flat_detector = flat;
    unified.cylindrical_detector = cyl;
    unified.volume = {16, 16, 8, 1.f, 1.f, 1.f,
        make_float3(3.f, 0.f, 2.f)};
    SVolGeom static_flat_volume{};
    unified.detector = EDetectorKind::Flat;
    unified.trajectory = ETrajectoryKind::Circular;
    if (!buildSystemGeometry(unified, static_flat, static_cyl,
            static_flat_volume) || static_flat.size() != 6 || !static_cyl.empty() ||
        std::fabs(static_flat.front().src.z - 7.f) > 1e-5f ||
        !(static_flat[1].angle.x < static_flat[0].angle.x))
        return 1;

    SVolGeom static_cyl_volume{};
    unified.detector = EDetectorKind::Cylindrical;
    if (!buildSystemGeometry(unified, static_flat, static_cyl,
            static_cyl_volume) || !static_flat.empty() || static_cyl.size() != 6 ||
        std::fabs(static_cyl.front().source.z - 7.f) > 1e-5f ||
        static_cyl_volume.Nx != static_flat_volume.Nx)
        return 1;

    SVolGeom flat_system_volume{};
    unified.detector = EDetectorKind::Flat;
    unified.trajectory = ETrajectoryKind::Helical;
    if (!buildSystemGeometry(unified, helical_flat, helical_cyl,
            flat_system_volume) || !helical_cyl.empty() ||
        helical_flat.size() != 8 ||
        std::fabs(helical_flat.front().src.z -
            (-12.f + unified.helical.source_offset_mm.z)) > 1e-5f ||
        std::fabs(flat_system_volume.center.x - 3.f) > 1e-5f)
        return 1;

    SVolGeom volume_geometry{};
    if (!buildVolumeGeometry(unified.volume, volume_geometry) ||
        std::fabs(volume_geometry.center.z - 2.f) > 1e-5f)
        return 1;

    SVolGeom cyl_system_volume{};
    unified.detector = EDetectorKind::Cylindrical;
    if (!buildSystemGeometry(unified, helical_flat, helical_cyl,
            cyl_system_volume) || !helical_flat.empty() || helical_cyl.size() != 8 ||
        cyl_system_volume.Nz != unified.volume.nz)
        return 1;

    // 新公开入口只使用一份 SSystemConfig。切换轨迹/探测器枚举时，
    // builder 必须清空另一种表面 geometry，不能留下上次构造结果。
    unified.detector = EDetectorKind::Flat;
    unified.trajectory = ETrajectoryKind::Circular;
    if (!buildSystemGeometry(unified, static_flat, static_cyl,
            volume_geometry) || static_flat.size() != 6 || !static_cyl.empty())
        return 1;
    unified.detector = EDetectorKind::Cylindrical;
    unified.trajectory = ETrajectoryKind::Helical;
    if (!buildSystemGeometry(unified, helical_flat, helical_cyl,
            volume_geometry) || !helical_flat.empty() || helical_cyl.size() != 8)
        return 1;

    unified.circular.views_per_turn = 1;
    unified.trajectory = ETrajectoryKind::Circular;
    if (buildSystemGeometry(unified, static_flat, static_cyl)) return 1;

    // DLL 边界可能收到来自其他语言的整数枚举。非法值必须被拒绝，不能
    // 落入 Cyl/Helical 的默认分支并构造一套含义错误但数值有效的 geometry。
    unified.circular.views_per_turn = 8;
    unified.detector = static_cast<EDetectorKind>(99);
    if (buildSystemGeometry(unified, static_flat, static_cyl)) return 1;
    unified.detector = EDetectorKind::Flat;
    unified.trajectory = static_cast<ETrajectoryKind>(99);
    if (buildSystemGeometry(unified, static_flat, static_cyl)) return 1;

    SReconstructionSpec reconstruction{};
    unified.detector = EDetectorKind::Flat;
    unified.trajectory = ETrajectoryKind::Circular;
    unified.circular.views_per_turn = 8;
    reconstruction.fdk.parker.mode = EParkerMode::Auto;
    if (!resolveParkerEnabled(unified, reconstruction) ||
        reconstruction.pipeline != EPipeline::FDK)
        return 1;
    unified.circular.total_views = unified.circular.views_per_turn;
    if (resolveParkerEnabled(unified, reconstruction)) return 1;
    return 0;
}
