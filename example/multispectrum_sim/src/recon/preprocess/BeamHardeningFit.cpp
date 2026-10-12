#include "BeamHardeningFit.hpp"
#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::preprocess {
void check(bool v, const std::string &message) {
    if (!v)
        throw std::runtime_error(message);
}
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> matrix,
                                      std::vector<double> rhs) {
    const int n = static_cast<int>(rhs.size());
    for (int column = 0; column < n; ++column) {
        int pivot = column;
        for (int row = column + 1; row < n; ++row)
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column]))
                pivot = row;
        check(std::abs(matrix[pivot][column]) > 1.0e-14,
              "water-plate beam hardening fit is singular");
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        const double scale = matrix[column][column];
        for (int j = column; j < n; ++j)
            matrix[column][j] /= scale;
        rhs[column] /= scale;
        for (int row = 0; row < n; ++row) {
            if (row == column)
                continue;
            const double factor = matrix[row][column];
            for (int j = column; j < n; ++j)
                matrix[row][j] -= factor * matrix[column][j];
            rhs[row] -= factor * rhs[column];
        }
    }
    return rhs;
}

BeamHardeningConfig fitWaterPlate(const SimulationConfig &config, const BeamHardeningConfig &input,
                                  const TaskProgressCallback &progress) {
    const auto spectrum = input.spectrum_file.empty() ? loadProjectionSpectrum(config.projection)
                                                      : loadSpectrum(input.spectrum_file);
    std::vector<double> energies;
    double photon_weight_sum = 0.0;
    double energy_weight_sum = 0.0;
    for (const auto &point : spectrum) {
        const double weight = point.fluence_per_keV_cm2_mAs_at_1m * point.bin_width_keV;
        energies.push_back(point.energy_keV);
        photon_weight_sum += weight;
        energy_weight_sum += weight * point.energy_keV;
    }
    check(photon_weight_sum > 0.0 && std::isfinite(energy_weight_sum) && energy_weight_sum > 0.0,
          "spectrum has no finite positive energy fluence");
    // The reference energy remains the photon-spectrum mean. The measured
    // polychromatic signal, however, must use energy-integrating weights.
    const auto reference = resolveWaterReference(config, input);
    MaterialSpec water;
    water.label = 1;
    water.name = "water";
    water.formula = "H2O";
    water.density_g_cm3 = 1.0;
    XcomAttenuationProvider attenuation({});
    std::vector<double> mass;
    std::string error;
    check(attenuation.query(water, energies, mass, error),
          "water attenuation query failed: " + error);
    const double mu_reference = reference.mu_water_mm_inv * 10.0;
    const int n = input.order + 1;
    std::vector<std::vector<double>> normal(n, std::vector<double>(n, 0.0));
    std::vector<double> rhs(n, 0.0);
    for (int sample = 0; sample < input.water_plate_samples; ++sample) {
        if (progress)
            progress({"Fitting BHC", sample, input.water_plate_samples});
        const double thickness_mm =
            input.water_plate_max_mm * sample / static_cast<double>(input.water_plate_samples - 1);
        double transmitted = 0.0;
        for (size_t e = 0; e < spectrum.size(); ++e)
            transmitted += spectrum[e].fluence_per_keV_cm2_mAs_at_1m * spectrum[e].bin_width_keV *
                           spectrum[e].energy_keV * std::exp(-mass[e] * thickness_mm / 10.0);
        const double multi = -std::log(
            std::max(transmitted / energy_weight_sum, std::numeric_limits<double>::min()));
        const double mono = mu_reference * thickness_mm / 10.0;
        std::vector<double> powers(static_cast<size_t>(n), 1.0);
        for (int j = 1; j < n; ++j)
            powers[j] = powers[j - 1] * multi;
        for (int row = 0; row < n; ++row) {
            rhs[row] += powers[row] * mono;
            for (int column = 0; column < n; ++column)
                normal[row][column] += powers[row] * powers[column];
        }
    }
    if (progress)
        progress({"Fitting BHC", input.water_plate_samples, input.water_plate_samples});
    BeamHardeningConfig result = input;
    result.coefficients = solveLinearSystem(std::move(normal), std::move(rhs));
    return result;
}
} // namespace yk::spectral::preprocess
