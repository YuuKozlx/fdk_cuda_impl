#include "cbct_calibration/io.hpp"
#include "cbct_calibration/pipeline.hpp"

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>

using namespace cbct::calibration;

namespace {

// The Python RAW regression writes this test-only point/mask fixture afresh.
// This exercises C++ DLT, source-circle and joint fit, not a C++ spiral tracker.
int runIndexedFixture(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::int32_t shape[2]{};
    input.read(reinterpret_cast<char*>(shape), sizeof(shape));
    const int views = shape[0], targets = shape[1];
    if (!input || views != 360 || targets != 25)
        throw std::runtime_error("invalid spiral test fixture dimensions");
    std::vector<Point3> world(static_cast<std::size_t>(targets));
    std::vector<Point2> points(static_cast<std::size_t>(views * targets));
    std::vector<unsigned char> usable(points.size());
    for (auto& point : world) input.read(reinterpret_cast<char*>(point.data()), 3 * sizeof(double));
    for (auto& point : points) input.read(reinterpret_cast<char*>(point.data()), 2 * sizeof(double));
    input.read(reinterpret_cast<char*>(usable.data()), static_cast<std::streamsize>(usable.size()));
    if (!input) throw std::runtime_error("truncated spiral test fixture");
    PipelineConfig config;
    const auto result = calibrateIndexedPoints(points, world, usable, views, config);
    writeCalibrationJson(path.parent_path() / "cpp_calibration.json", result);
    std::cout << "C++ spiral (Python RAW points/mask): SID=" << result.machine.sid_mm
              << " SDD=" << result.machine.sdd_mm
              << " offsetU=" << result.machine.offset_u_mm
              << " offsetV=" << result.machine.offset_v_mm
              << " RMSE=" << result.joint_rmse_px << "\n";
    return result.joint_converged && result.joint_rmse_px < .2 &&
        std::abs(result.machine.sid_mm - 440.11) < .1 &&
        std::abs(result.machine.sdd_mm - 769.83) < .1 &&
        std::abs(result.machine.offset_u_mm - 9.31) < .1 &&
        std::abs(result.machine.offset_v_mm + 1.22) < .1 ? 0 : 1;
}

int runCase(const std::filesystem::path& repository,
            const std::string& label,
            const std::filesystem::path& raw_relative,
            int beads_per_ring,
            double upper_radius_mm,
            double lower_radius_mm,
            double lower_phase_deg,
            const std::vector<Point3>& markers) {
    PipelineConfig config;
    config.tracker.raw_path = repository / raw_relative;
    config.tracker.views = 360;
    config.tracker.image = {1024, 1024};
    config.tracker.threshold = 5.0F;
    config.phantom = PhantomSpec{};
    config.phantom.beads_per_ring = beads_per_ring;
    config.phantom.upper_radius_mm = upper_radius_mm;
    config.phantom.lower_radius_mm = lower_radius_mm;
    config.phantom.lower_phase_deg = lower_phase_deg;
    config.phantom.marker_points_mm = markers;
    config.pixel = {0.417, 0.417};
    config.joint_max_iterations = 120;
    const auto result = calibrateRaw(config);
    int merged_count = 0, recovered = 0, minimum_usable = result.tracking.targets;
    for (int view = 0; view < result.tracking.views; ++view) {
        int usable_count = 0;
        for (int target = 0; target < result.tracking.targets; ++target) {
            const auto index = static_cast<std::size_t>(view * result.tracking.targets + target);
            usable_count += result.tracking.isUsable(view, target) ? 1 : 0;
            if (result.tracking.merged[index]) {
                ++merged_count;
                if (!result.tracking.isObserved(view, target) || result.tracking.isUsable(view, target))
                    throw std::runtime_error("merged target has inconsistent usability");
                if (view + 1 < result.tracking.views && result.tracking.isUsable(view + 1, target))
                    ++recovered;
            }
        }
        minimum_usable = std::min(minimum_usable, usable_count);
    }
    double dlt_max = 0.0, dlt_sum = 0.0; int dlt_count = 0, first_bad = -1;
    for (int frame = 0; frame < static_cast<int>(result.tracking.dlt_rmse_px.size()); ++frame) {
        const double value = result.tracking.dlt_rmse_px[static_cast<std::size_t>(frame)];
        if (std::isfinite(value)) { dlt_max = std::max(dlt_max, value); dlt_sum += value; ++dlt_count; }
        if (first_bad < 0 && std::isfinite(value) && value > 1.0) first_bad = frame;
    }
    const auto output = repository / ("out/cbct-calibration-cpp-" + label + ".json");
    writeCalibrationJson(output, result);
    std::cout << label
              << " SID=" << result.machine.sid_mm
              << " SDD=" << result.machine.sdd_mm
              << " offsetU=" << result.machine.offset_u_mm
              << " offsetV=" << result.machine.offset_v_mm
              << " RMSE=" << result.joint_rmse_px
              << " anchor=" << result.tracking.anchor_frame
              << " anchorDLT=" << result.tracking.dlt_rmse_px.at(
                     static_cast<std::size_t>(result.tracking.anchor_frame))
              << " DLTmax=" << dlt_max
              << " firstBad=" << first_bad
              << " DLTmean=" << (dlt_count ? dlt_sum / dlt_count : 0.0)
              << " merged=" << merged_count << " recovered=" << recovered
              << " minUsable=" << minimum_usable << "\n"
              << output.string() << "\n";
    const bool geometry_ok =
        std::abs(result.machine.sid_mm - 440.0) < 2.0 &&
        std::abs(result.machine.sdd_mm - 770.0) < 3.0 &&
        std::abs(result.machine.offset_u_mm - 9.308) < 0.5 &&
        std::abs(result.machine.offset_v_mm + 1.23) < 0.5;
    return result.joint_rmse_px < 1.0 && geometry_ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
    if (argc == 2) return runIndexedFixture(argv[1]);
    const auto repository = std::filesystem::current_path();
    const double cho_marker_phase = 15.0 * 3.14159265358979323846 / 180.0;
    const double yang_marker_phase = 30.0 * 3.14159265358979323846 / 180.0;
    const int cho_single = runCase(
        repository, "cho-12-single-marker",
        "example/multispectrum_sim/outputs/calibration/synchronized-ring-12-marker-"
        "1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-"
        "prx1-pry2-prz3-projection.raw",
        12, 50.0, 50.0, 0.0, {{50.0, 0.0, 80.0}});
    const int cho = runCase(
        repository, "cho-12-dual-marker",
        "example/multispectrum_sim/outputs/calibration/synchronized-ring-12-dual-"
        "marker-1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-"
        "prx1-pry2-prz3-projection.raw",
        12, 50.0, 50.0, 0.0,
        {{50.0, 0.0, 80.0},
         {50.0 * std::cos(cho_marker_phase),
          50.0 * std::sin(cho_marker_phase), 70.0}});
    const int yang_single = runCase(
        repository, "yang-6-single-marker",
        "example/multispectrum_sim/outputs/calibration/synchronized-ring-6-marker-"
        "1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-"
        "prx1-pry2-prz3-projection.raw",
        6, 50.0, 50.0, 0.0, {{50.0, 0.0, 80.0}});
    const int yang = runCase(
        repository, "yang-6-dual-marker",
        "example/multispectrum_sim/outputs/calibration/synchronized-ring-6-dual-"
        "marker-1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-"
        "prx1-pry2-prz3-projection.raw",
        6, 50.0, 50.0, 0.0,
        {{50.0, 0.0, 80.0},
         {50.0 * std::cos(yang_marker_phase),
          50.0 * std::sin(yang_marker_phase), 70.0}});
    const int unequal = runCase(
        repository, "unequal-ring-single-marker",
        "example/multispectrum_sim/outputs/quality/double-ring-asymmetric-marker-"
        "1024x1024x360-ou5-ov10-tu1-tv2-tn3-soy10-soz5-px10-py15-pz20-"
        "prx1-pry2-prz3-projection.raw",
        12, 50.0, 40.0, 15.0, {{50.0, 0.0, 80.0}});
    return cho_single || cho || yang_single || yang || unequal;
    } catch (const std::exception& error) {
        std::cerr << "real-data calibration failed: " << error.what() << "\n";
        return 2;
    }
}
