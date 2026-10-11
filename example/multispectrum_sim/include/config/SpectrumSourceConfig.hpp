#pragma once
#include <filesystem>
#include <string>
#include <vector>
namespace yk::spectral {
struct SpectrumFilterConfig {
    std::string material;
    double thickness_mm = 0.0;
    int atomic_number = 0;
    double density_g_cm3 = 0.0;
};
// SDK-independent controls. Pipeline fluence is per mAs at 1 m.
struct SpectrumSourceConfig {
    std::string source = "file";
    std::string model = "spekpy-v1";
    double kvp = 120.0;
    double anode_angle_deg = 12.0;
    double bin_width_keV = 1.0;
    std::vector<SpectrumFilterConfig> filters;
    std::filesystem::path export_file;
};
}
