#pragma once
#include "recon/postprocess/PostprocessPipeline.hpp"
namespace yk::spectral::postprocess {
struct HuLinearCoefficients {
    double scale = 1, offset = 0;
};
struct AirWaterTargets {
    double air_hu = -1000, water_hu = 0, minimum_measured_gap_hu = 1;
};
struct HuCalibrationResult {
    bool valid = false;
    HuLinearCoefficients coefficients;
    double measured_air_hu = 0, measured_water_hu = 0;
    double target_air_hu = -1000, target_water_hu = 0;
    size_t air_voxels = 0, water_voxels = 0;
};
// Half-open voxel-index box [begin,end), on the current output grid.
struct CalibrationRoi {
    std::array<int, 3> begin {}, end {};
};
HuCalibrationResult fitAirWaterHu(double measured_air_hu, double measured_water_hu,
    AirWaterTargets targets = {});
// HU -> HU: does not modify image values. Publishes a result for later correction.
// prepare invalidates an old result; successful results survive reset/end of run.
std::unique_ptr<IStage> makeAirWaterHuCalibrationStage(CalibrationRoi air, CalibrationRoi water,
    std::shared_ptr<HuCalibrationResult> output, AirWaterTargets targets = {}, Location = Location::Device);
} // namespace
