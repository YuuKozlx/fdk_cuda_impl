#include <cmath>
#include <vector>

#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"

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
    return 0;
}
