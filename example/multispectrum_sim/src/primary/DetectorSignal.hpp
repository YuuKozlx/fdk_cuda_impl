#pragma once

// CPU detector signal processing and projection metadata/thickness-map I/O.

#include "config/SimConfig.hpp"
#include <filesystem>
#include <random>
#include <vector>

namespace yk::spectral::primary_detail {

void writeMetadata(const std::filesystem::path& file, int u, int v,
    int views, const char* signal);
std::vector<float> loadLayerThickness(const DetectorLayerConfig& layer, int pixels);
void applyCrosstalkSignal(std::vector<float>& signal, int width, int height,
    const CrosstalkConfig& config);
void applyAfterglowSignal(std::vector<float>& signal,
    const DetectorPostprocessConfig::AfterglowConfig& config,
    std::vector<std::vector<float>>& state);
void applyDasSignal(std::vector<float>& signal,
    const DetectorPostprocessConfig::DasConfig& config, std::mt19937& random);

} // namespace yk::spectral::primary_detail
