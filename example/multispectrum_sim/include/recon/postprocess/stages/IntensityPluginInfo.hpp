#pragma once
#include "pipeline/PluginInfo.hpp"
namespace yk::spectral::postprocess {
inline const std::vector<pipeline::PluginInfo>& intensityPluginDescriptions()
{
    static const auto all = [] {
        using namespace pipeline;
        std::vector<PluginInfo> result;
        for (const bool device : { false, true })
            for (const std::string op : { "mu_to_hu", "hu_to_mu", "air_water_calibration", "hu_linear", "hu_thresholds" }) {
                PluginInfo p;
                p.id = "postprocess." + op + (device ? ".cuda" : ".cpu");
                p.name = (device ? "CUDA " : "CPU ") + op;
                p.version = "1.0";
                p.kind = PluginKind::PostprocessStage;
                p.category = op == "air_water_calibration" ? "calibration" : op == "hu_thresholds" ? "segmentation"
                                                                                                   : "correction";
                p.tags = { device ? "cuda" : "cpu", "volume", "HU" };
                p.documentation = "example/multispectrum_sim/include/recon/postprocess/stages/README.md";
                if (op == "mu_to_hu" || op == "hu_to_mu") {
                    p.description = "Convert attenuation and HU with configurable air/water reference anchors; preserves negative values.";
                    p.parameters = { { "water_mu_mm_inv", "Water attenuation", "Water reference at the selected energy.", ParameterType::Real, "mm^-1" }, { "air_mu_mm_inv", "Air attenuation", "Attenuation reference for air.", ParameterType::Real, "mm^-1" }, { "water_hu", "Water HU", "Target HU anchor.", ParameterType::Real, "HU" }, { "air_hu", "Air HU", "Target HU anchor.", ParameterType::Real, "HU" } };
                } else if (op == "air_water_calibration") {
                    p.description = "Measure mean HU in disjoint air/water ROIs and publish a linear calibration without changing image values.";
                    p.parameters = { { "air_roi", "Air ROI", "Half-open voxel-index box.", ParameterType::RealArray, "voxel" }, { "water_roi", "Water ROI", "Half-open voxel-index box.", ParameterType::RealArray, "voxel" }, { "air_hu", "Target air HU", "Default -1000.", ParameterType::Real, "HU" }, { "water_hu", "Target water HU", "Default 0.", ParameterType::Real, "HU" }, { "minimum_measured_gap_hu", "Minimum separation", "Reject degenerate or reversed anchors.", ParameterType::Real, "HU" } };
                } else if (op == "hu_linear") {
                    p.description = "Apply HU_out = scale * HU_in + offset, using supplied coefficients or a completed calibration.";
                    p.parameters = { { "scale", "Scale", "Positive linear gain.", ParameterType::Real, "1" }, { "offset", "Offset", "HU offset.", ParameterType::Real, "HU" } };
                } else {
                    p.description = "Classify HU into ordered intervals; equality belongs to the upper interval. Output is labeled data, not attenuation or HU.";
                    p.parameters = { { "thresholds_hu", "Thresholds", "Strictly increasing finite thresholds.", ParameterType::RealArray, "HU" }, { "label_ids", "Label IDs", "N+1 exact nonnegative float32 integers.", ParameterType::RealArray, "1" }, { "label_names", "Label names", "Optional human-readable names.", ParameterType::String, "" } };
                }
                result.push_back(std::move(p));
            }
        return result;
    }();
    return all;
}
} // namespace
