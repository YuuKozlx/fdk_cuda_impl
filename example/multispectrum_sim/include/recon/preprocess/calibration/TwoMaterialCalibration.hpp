#pragma once
#include <filesystem>
#include <string>
#include <vector>
namespace yk::spectral::preprocess {
// Rows: second-material thickness [mm]; columns: first-material thickness [mm].
// The water/bone field names also represent PMMA/Al in an empirical calibration;
// the provider must supply the same second-material basis as the table.
// Calibrated log transmission must increase strictly across each row.
struct TwoMaterialSamples {
    int bone_rows = 0, water_columns = 0;
    double bone_step_mm = 0, water_step_mm = 0;
    double target_water_mu_mm = 0, target_bone_mu_mm = 0, reference_energy_keV = 0;
    std::string provenance;
    std::vector<double> log_values;
};
struct TwoMaterialCalibration {
    // reference_energy_keV == 0 denotes an empirical tangent target without an assigned energy.
    int bone_rows = 0, projection_columns = 0;
    double bone_step_mm = 0, reference_energy_keV = 0;
    double target_water_mu_mm = 0, target_bone_mu_mm = 0;
    std::string provenance;
    std::vector<float> minimum_log, log_step, target_log;
    void validate() const;
    // Linear end extrapolation preserves noise; bone outside calibration is rejected.
    float correct(float measured_log, float bone_mm) const;
    void save(const std::filesystem::path&) const;
    static TwoMaterialCalibration load(const std::filesystem::path&);
};
TwoMaterialCalibration calibrateTwoMaterial(const TwoMaterialSamples&, int projection_columns = 1025);
}
