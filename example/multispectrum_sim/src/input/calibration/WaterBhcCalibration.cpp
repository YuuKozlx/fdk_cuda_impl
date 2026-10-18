#include "input/calibration/WaterBhcCalibration.hpp"
#include "input/calibration/WaterBhcTable.hpp"
#include "input/materials/WaterReference.hpp"
#include "input/materials/XcomAttenuationProvider.hpp"
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::calibration {
void check(bool v, const std::string& message)
{
    if (!v)
        throw std::runtime_error(message);
}
// Householder QR avoids squaring the Vandermonde condition number. Fit on
// [-1,1], then convert back to the raw line-integral polynomial used by stages.
static std::vector<double> fitPolynomial(const std::vector<double>& x,
    std::vector<double> y, int n)
{
    const int m = static_cast<int>(x.size());
    const double extent = *std::max_element(x.begin(), x.end());
    check(std::isfinite(extent) && extent > 1e-12,
        "water-plate beam hardening fit has insufficient dynamic range");
    Eigen::MatrixXd design(m, n);
    for (int i = 0; i < m; ++i) {
        design(i, 0) = 1;
        for (int j = 1; j < n; ++j)
            design(i, j) = design(i, j - 1) * (2 * x[i] / extent - 1);
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(design);
    qr.setThreshold(1e-14);
    check(qr.rank() == n, "water-plate beam hardening fit is singular");
    const Eigen::VectorXd c = qr.solve(Eigen::Map<const Eigen::VectorXd>(y.data(), m));
    check(c.allFinite(), "non-finite water-plate fit coefficient");
    // Horner composition q(2*p/extent-1).
    std::vector<double> result(1, c[n - 1]);
    for (int j = n - 2; j >= 0; --j) {
        std::vector<double> next(result.size() + 1, 0);
        for (size_t i = 0; i < result.size(); ++i) {
            next[i] -= result[i];
            next[i + 1] += 2 * result[i] / extent;
        }
        next[0] += c[j];
        result = std::move(next);
    }
    for (double c : result)
        check(std::isfinite(c), "non-finite water-plate fit coefficient");
    return result;
}

BeamHardeningConfig fitWaterPlate(const SimulationConfig& config, const BeamHardeningConfig& input,
    const TaskProgressCallback& progress)
{
    // Independent calibration API can be called without the TOML validator.
    check(input.order >= 1 && input.order <= 16 && input.water_plate_samples >= input.order + 2 && std::isfinite(input.water_plate_max_mm) && input.water_plate_max_mm > 0 && std::isfinite(input.reference_energy_keV) && input.reference_energy_keV >= 0,
        "invalid water-plate beam hardening fit parameters");
    const auto spectrum = input.spectrum_file.empty() ? loadProjectionSpectrum(config.projection)
                                                      : loadSpectrum(input.spectrum_file);
    std::vector<double> energies;
    double photon_weight_sum = 0.0;
    double energy_weight_sum = 0.0;
    for (const auto& point : spectrum) {
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
    std::vector<double> measured, target;
    for (int sample = 0; sample < input.water_plate_samples; ++sample) {
        if (progress)
            progress({ "Fitting BHC", sample, input.water_plate_samples });
        const double thickness_mm = input.water_plate_max_mm * sample / static_cast<double>(input.water_plate_samples - 1);
        double transmitted = 0.0;
        for (size_t e = 0; e < spectrum.size(); ++e)
            transmitted += spectrum[e].fluence_per_keV_cm2_mAs_at_1m * spectrum[e].bin_width_keV * spectrum[e].energy_keV * std::exp(-mass[e] * thickness_mm / 10.0);
        const double multi = -std::log(
            std::max(transmitted / energy_weight_sum, std::numeric_limits<double>::min()));
        const double mono = mu_reference * thickness_mm / 10.0;
        measured.push_back(multi);
        target.push_back(mono);
    }
    if (progress)
        progress({ "Fitting BHC", input.water_plate_samples, input.water_plate_samples });
    BeamHardeningConfig result = input;
    result.reference_energy_keV = reference.energy_keV;
    result.fit_from_water_plate = false;
    result.coefficients = fitPolynomial(measured, std::move(target), n);
    return result;
}
BeamHardeningConfig calibrateWaterBhc(const SimulationConfig& config, const std::filesystem::path& output, TaskProgressCallback progress)
{
    if (output.empty())
        throw std::invalid_argument("Water BHC output file is required");
    auto recipe = config.reconstruction.preprocessing.beam_hardening;
    recipe.calibration_file.clear();
    auto fitted = fitWaterPlate(config, recipe, progress);
    const auto reference = resolveWaterReference(config, fitted);
    WaterBhcTable table { fitted.coefficients, fitted.clamp_min, fitted.clamp_max, reference.energy_keV, reference.mu_water_mm_inv,
        recipe.water_plate_max_mm, "Simulated homogeneous water; energy-integrating detector; source=" + config.source_config_file.generic_string() };
    table.save(output);
    fitted.calibration_file = std::filesystem::absolute(output);
    return fitted;
}
} // namespace yk::spectral::calibration
