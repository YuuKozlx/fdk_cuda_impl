#pragma once
#include "input/config/SimConfig.hpp"
namespace yk::spectral::calibration {
// Energy-integrating negative-log input -> equivalent monoenergetic line integral.
// Independent of phantom size, reconstruction grid, and reconstruction algorithm.
struct WaterBhcTable {
    std::vector<double> coefficients;
    double clamp_min = 0, clamp_max = 20;
    double reference_energy_keV = 0, mu_water_mm_inv = 0;
    double calibrated_max_water_mm = 0;
    std::string provenance;
    void validate() const;
    void save(const std::filesystem::path&) const;
    static WaterBhcTable load(const std::filesystem::path&);
};
// Loads a prepared table or accepts explicit coefficients; never fits or generates.
BeamHardeningConfig resolveWaterBhc(const BeamHardeningConfig&);
}
