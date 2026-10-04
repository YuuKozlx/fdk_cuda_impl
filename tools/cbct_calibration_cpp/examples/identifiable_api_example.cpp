#include "cbct_calibration/io.hpp"
#include "cbct_calibration/pipeline.hpp"

namespace cbct::calibration::examples {

CalibrationResult runIdentifiablePhantomCalibration(
        const std::filesystem::path& raw_path,
        const std::filesystem::path& report_path) {
    PipelineConfig config;
    config.tracker.raw_path = raw_path;
    config.tracker.views = 360;
    config.tracker.image = {1024, 1024};
    config.tracker.threshold = 5.0F;
    config.tracker.gate_radius_px = 28.0;
    config.phantom = PhantomSpec{};
    config.pixel = {0.417, 0.417};

    CalibrationResult result = calibrateRaw(config);
    writeCalibrationJson(report_path, result);
    return result;
}

}  // namespace cbct::calibration::examples
