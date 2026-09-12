#include "XcomAttenuationProvider.hpp"

#include <cmath>
#include <iostream>

int main()
{
    yk::spectral::MaterialSpec water;
    water.label = 1;
    water.name = "water";
    water.formula = "H2O";
    water.density_g_cm3 = 1.0;

    yk::spectral::XcomAttenuationProvider provider(XCOM_TEST_DATA_DIRECTORY);
    std::vector<double> values;
    std::string error;
    if (!provider.query(water, {20.0, 60.0, 100.0}, values, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    const double expected[] = {0.809820711613, 0.205873382278, 0.170726725133};
    if (values.size() != 3) return 2;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - expected[i]) > 2.0e-6) {
            std::cerr << "XCOM water mismatch at " << i << ": " << values[i] << '\n';
            return 3;
        }
    }
    yk::spectral::MaterialSpec preset_water;
    preset_water.label = 2;
    preset_water.name = "preset-water";
    preset_water.preset = "water";
    preset_water.density_g_cm3 = 1.0;
    std::vector<double> preset_values;
    if (!provider.query(preset_water, {20.0, 60.0, 100.0}, preset_values, error)) {
        std::cerr << error << '\n';
        return 4;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - preset_values[i]) > 2.0e-6) {
            std::cerr << "builtin water mismatch at " << i << '\n';
            return 5;
        }
    }
    yk::spectral::MaterialSpec brain;
    brain.label = 3;
    brain.name = "brain";
    brain.preset = "ICRU_brain_adult";
    brain.density_g_cm3 = 1.04;
    if (!provider.query(brain, {20.0, 60.0, 100.0}, preset_values, error)) {
        std::cerr << error << '\n';
        return 6;
    }
    for (double value : preset_values) {
        if (!std::isfinite(value) || value <= 0.0) return 7;
    }
    std::size_t checked = 0;
    for (auto material = yk::spectral::builtinMaterialsBegin();
         material != yk::spectral::builtinMaterialsEnd(); ++material) {
        yk::spectral::MaterialSpec spec;
        spec.label = 1;
        spec.name = material->display_name;
        spec.preset = material->id;
        spec.density_g_cm3 = material->density_g_cm3;
        std::vector<double> value;
        if (!provider.query(spec, {60.0}, value, error) || value.size() != 1 ||
            !std::isfinite(value.front()) || value.front() < 0.0) {
            std::cerr << "builtin material query failed: " << material->id
                << " " << error << '\n';
            return 8;
        }
        ++checked;
    }
    if (checked < 250) return 9;
    return 0;
}
