#pragma once
#include "recon/postprocess/PostprocessPipeline.hpp"
namespace yk::spectral::postprocess {
struct HuThresholdSegmentation {
    std::vector<double> thresholds_hu;
    // N thresholds require N+1 classes, including below/above the outer thresholds.
    // Equality belongs to the upper interval: [-inf,t0),[t0,t1),...,[tN,+inf).
    std::vector<LabelClass> classes;
};
// HU -> Labels. Storage stays float32 with exact nonnegative integer label IDs.
std::unique_ptr<IStage> makeHuThresholdStage(HuThresholdSegmentation, Location = Location::Device);
} // namespace
