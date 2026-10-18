#include "input/calibration/WaterBhcTable.hpp"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <toml++/toml.hpp>
namespace yk::spectral::calibration {
void WaterBhcTable::validate() const
{
    if (coefficients.size() < 2 || coefficients.size() > 17 || !std::isfinite(clamp_min) || !std::isfinite(clamp_max) || clamp_min < 0 || clamp_max <= clamp_min || !std::isfinite(reference_energy_keV) || reference_energy_keV <= 0 || !std::isfinite(mu_water_mm_inv) || mu_water_mm_inv <= 0 || !std::isfinite(calibrated_max_water_mm) || calibrated_max_water_mm <= 0)
        throw std::invalid_argument("Invalid water BHC calibration table");
    for (double c : coefficients)
        if (!std::isfinite(c))
            throw std::invalid_argument("Non-finite water BHC coefficient");
}
void WaterBhcTable::save(const std::filesystem::path& path) const
{
    validate();
    toml::array values;
    for (double c : coefficients)
        values.push_back(c);
    toml::table table { { "format", "water_bhc" }, { "version", 1 },
        { "input_domain", "energy_integrating_log" }, { "output_domain", "mono_log" },
        { "length_unit", "mm" }, { "coefficients", std::move(values) },
        { "clamp_min", clamp_min }, { "clamp_max", clamp_max },
        { "reference_energy_keV", reference_energy_keV }, { "mu_water_mm_inv", mu_water_mm_inv },
        { "calibrated_max_water_mm", calibrated_max_water_mm }, { "provenance", provenance } };
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file << table;
    file.close();
}
WaterBhcTable WaterBhcTable::load(const std::filesystem::path& path)
{
    auto t = toml::parse_file(path.string());
    if (t["format"].value_or(std::string{}) != "water_bhc" || t["version"].value_or(0) != 1 || t["input_domain"].value_or(std::string{}) != "energy_integrating_log" || t["output_domain"].value_or(std::string{}) != "mono_log" || t["length_unit"].value_or(std::string{}) != "mm")
        throw std::runtime_error("Unsupported water BHC table format/domain/units");
    const auto number = [&](const char* key) {auto v=t[key].value<double>();if(!v)throw std::runtime_error(std::string("Missing water BHC field: ")+key);return *v; };
    WaterBhcTable result;
    if (auto values = t["coefficients"].as_array())
        for (const auto& value : *values) {
            auto v = value.value<double>();
            if (!v)
                throw std::runtime_error("Invalid water BHC coefficients");
            result.coefficients.push_back(*v);
        }
    result.clamp_min = number("clamp_min");
    result.clamp_max = number("clamp_max");
    result.reference_energy_keV = number("reference_energy_keV");
    result.mu_water_mm_inv = number("mu_water_mm_inv");
    result.calibrated_max_water_mm = number("calibrated_max_water_mm");
    result.provenance = t["provenance"].value_or(std::string{});
    result.validate();
    return result;
}
BeamHardeningConfig resolveWaterBhc(const BeamHardeningConfig& input)
{
    auto result = input;
    if (!input.calibration_file.empty()) {
        const auto table = WaterBhcTable::load(input.calibration_file);
        result.coefficients = table.coefficients;
        result.order = int(table.coefficients.size()) - 1;
        result.clamp_min = table.clamp_min;
        result.clamp_max = table.clamp_max;
        result.reference_energy_keV = table.reference_energy_keV;
        result.fit_from_water_plate = false;
    } else if (input.fit_from_water_plate) {
        throw std::runtime_error("Water BHC is not calibrated. Generate/select a water BHC table first (--calibrate-water-bhc or the UI calibration action).");
    }
    // Low-level pipelines accept explicit polynomials, including constants.
    // TOML configuration separately validates its declared fitting order.
    if (result.coefficients.empty() || result.coefficients.size() > 17)
        throw std::runtime_error("Invalid prepared water BHC polynomial");
    result.order = int(result.coefficients.size()) - 1;
    for (double c : result.coefficients)
        if (!std::isfinite(c))
            throw std::runtime_error("Non-finite water BHC coefficient");
    return result;
}
}
