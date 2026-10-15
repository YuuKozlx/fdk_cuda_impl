#include "recon/preprocess/calibration/TwoMaterialCalibration.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
namespace yk::spectral::preprocess {
namespace {
    void require(bool ok)
    {
        if (!ok)
            throw std::invalid_argument("invalid two-material calibration");
    }
    float row(const TwoMaterialCalibration& m, int b, float p)
    {
        const float x = (p - m.minimum_log[b]) / m.log_step[b];
        const int i = int(std::clamp(x, 0.f, float(m.projection_columns - 2)));
        const float f = x - i;
        const size_t base = size_t(b) * m.projection_columns + i;
        return m.target_log[base] + f * (m.target_log[base + 1] - m.target_log[base]);
    }
}
void TwoMaterialCalibration::validate() const
{
    require(bone_rows >= 2 && projection_columns >= 2 && std::isfinite(bone_step_mm) && bone_step_mm > 0);
    require(std::isfinite(reference_energy_keV) && reference_energy_keV >= 0);
    require(std::isfinite(target_water_mu_mm) && target_water_mu_mm > 0 && std::isfinite(target_bone_mu_mm) && target_bone_mu_mm > 0);
    require(minimum_log.size() == size_t(bone_rows) && log_step.size() == size_t(bone_rows) && target_log.size() == size_t(bone_rows) * projection_columns);
    for (int b = 0; b < bone_rows; ++b) {
        require(std::isfinite(minimum_log[b]) && std::isfinite(log_step[b]) && log_step[b] > 0);
        for (int j = 0; j < projection_columns; ++j) {
            auto v = target_log[size_t(b) * projection_columns + j];
            require(std::isfinite(v));
            if (j)
                require(v > target_log[size_t(b) * projection_columns + j - 1]);
        }
    }
}
float TwoMaterialCalibration::correct(float p, float bone) const
{
    require(std::isfinite(p) && std::isfinite(bone) && bone >= 0 && bone <= (bone_rows - 1) * bone_step_mm);
    const double x = bone / bone_step_mm;
    const int b = std::min(int(x), bone_rows - 2);
    const float t = float(x - b);
    return row(*this, b, p) * (1 - t) + row(*this, b + 1, p) * t;
}
TwoMaterialCalibration calibrateTwoMaterial(const TwoMaterialSamples& s, int columns)
{
    require(s.bone_rows >= 2 && s.water_columns >= 2 && columns >= 2 && s.bone_step_mm > 0 && s.water_step_mm > 0);
    require(std::isfinite(s.bone_step_mm) && std::isfinite(s.water_step_mm));
    require(s.log_values.size() == size_t(s.bone_rows) * s.water_columns);
    TwoMaterialCalibration m;
    m.bone_rows = s.bone_rows;
    m.projection_columns = columns;
    m.bone_step_mm = s.bone_step_mm;
    m.reference_energy_keV = s.reference_energy_keV;
    m.target_water_mu_mm = s.target_water_mu_mm;
    m.target_bone_mu_mm = s.target_bone_mu_mm;
    m.provenance = s.provenance;
    for (int b = 0; b < s.bone_rows; ++b) {
        const auto begin = s.log_values.begin() + size_t(b) * s.water_columns;
        for (int j = 0; j < s.water_columns; ++j) {
            require(std::isfinite(begin[j]));
            if (j)
                require(begin[j] > begin[j - 1]);
        }
        const double low = begin[0], step = (begin[s.water_columns - 1] - low) / (columns - 1);
        m.minimum_log.push_back(float(low));
        m.log_step.push_back(float(step));
        for (int j = 0; j < columns; ++j) {
            const double p = low + step * j;
            const int k = std::clamp(int(std::upper_bound(begin, begin + s.water_columns, p) - begin) - 1, 0, s.water_columns - 2);
            const double water = (k + (p - begin[k]) / (begin[k + 1] - begin[k])) * s.water_step_mm;
            m.target_log.push_back(float(water * s.target_water_mu_mm + b * s.bone_step_mm * s.target_bone_mu_mm));
        }
    }
    m.validate();
    return m;
}
void TwoMaterialCalibration::save(const std::filesystem::path& path) const
{
    validate();
    nlohmann::json j = { { "schema", "two-material-bhc-lut-v1" }, { "input_domain", "scatter-corrected-log-before-water-bhc" },
        { "method", reference_energy_keV > 0 ? "2DCalBH-style monotone inverse LUT; specified reference energy" : "Empirical two-exponential response; zero-thickness tangent target; no assigned energy" }, { "bone_rows", bone_rows }, { "projection_columns", projection_columns },
        { "bone_step_mm", bone_step_mm }, { "reference_energy_keV", reference_energy_keV }, { "target_water_mu_mm", target_water_mu_mm },
        { "target_bone_mu_mm", target_bone_mu_mm }, { "provenance", provenance }, { "minimum_log", minimum_log }, { "log_step", log_step }, { "target_log", target_log } };
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << j.dump(2);
    out.close();
}
TwoMaterialCalibration TwoMaterialCalibration::load(const std::filesystem::path& path)
{
    std::ifstream in(path);
    nlohmann::json j;
    in >> j;
    require(j.at("schema") == "two-material-bhc-lut-v1");
    require(j.at("input_domain") == "scatter-corrected-log-before-water-bhc");
    TwoMaterialCalibration m;
    m.bone_rows = j.at("bone_rows");
    m.projection_columns = j.at("projection_columns");
    m.bone_step_mm = j.at("bone_step_mm");
    m.reference_energy_keV = j.at("reference_energy_keV");
    m.target_water_mu_mm = j.at("target_water_mu_mm");
    m.target_bone_mu_mm = j.at("target_bone_mu_mm");
    m.provenance = j.at("provenance");
    m.minimum_log = j.at("minimum_log").get<std::vector<float>>();
    m.log_step = j.at("log_step").get<std::vector<float>>();
    m.target_log = j.at("target_log").get<std::vector<float>>();
    m.validate();
    return m;
}
}
