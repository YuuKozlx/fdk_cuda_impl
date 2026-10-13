#pragma once
#include "recon/postprocess/stages/calibration/HuCalibration.hpp"
#include "recon/postprocess/stages/segmentation/HuThresholds.hpp"
namespace yk::spectral::postprocess::intensity_detail {
void validateLocation(Location);
void validateStorage(const Volume&, Location);
void validateCoefficients(HuLinearCoefficients);
void validateTargets(AirWaterTargets);
void validateRoi(const CalibrationRoi&, const Grid&);
size_t roiCount(const CalibrationRoi&);
void applyLinear(Volume&, Location, HuLinearCoefficients);
double meanRoi(const Volume&, Location, const CalibrationRoi&);
void classify(Volume&, Location, const HuThresholdSegmentation&);
#ifdef YK_INTENSITY_CUDA
void cudaLinear(float*, size_t, HuLinearCoefficients);
double cudaMeanRoi(const float*, const Grid&, const CalibrationRoi&);
void cudaClassify(float*, size_t, const HuThresholdSegmentation&);
#endif
} // namespace
