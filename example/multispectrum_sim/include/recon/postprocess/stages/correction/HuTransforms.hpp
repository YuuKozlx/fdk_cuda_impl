#pragma once
#include "recon/postprocess/stages/calibration/HuCalibration.hpp"
namespace yk::spectral::postprocess {
// Caller supplies the water attenuation at the chosen energy (e.g. BHC fit energy).
// Standard HU: air_mu=0, water_hu=0, air_hu=-1000. All anchors are configurable.
struct MuHuReference {
    double water_mu_mm_inv = 0, air_mu_mm_inv = 0;
    double water_hu = 0, air_hu = -1000;
};
std::unique_ptr<IStage> makeMuToHuStage(MuHuReference, Location = Location::Device);
std::unique_ptr<IStage> makeHuToMuStage(MuHuReference, Location = Location::Device);
std::unique_ptr<IStage> makeHuCorrectionStage(HuLinearCoefficients, Location = Location::Device);
// Reads the most recent valid calibration at process time, allowing calibration
// and correction in one pipeline. The result must not be mutated concurrently.
std::unique_ptr<IStage> makeHuCorrectionStage(std::shared_ptr<const HuCalibrationResult>, Location = Location::Device);
} // namespace
