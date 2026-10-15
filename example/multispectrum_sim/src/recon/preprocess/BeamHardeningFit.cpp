#include "BeamHardeningFit.hpp"
#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace yk::spectral::preprocess {
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
    std::vector<std::vector<double>> a(m, std::vector<double>(n, 1));
    for (int i = 0; i < m; ++i)
        for (int j = 1; j < n; ++j)
            a[i][j] = a[i][j - 1] * (2 * x[i] / extent - 1);
    for (int k = 0; k < n; ++k) {
        double norm = 0;
        for (int i = k; i < m; ++i)
            norm = std::hypot(norm, a[i][k]);
        check(norm > 1e-14, "water-plate beam hardening fit is singular");
        const double alpha = -std::copysign(norm, a[k][k]);
        std::vector<double> v(m - k);
        for (int i = k; i < m; ++i)
            v[i - k] = a[i][k];
        v[0] -= alpha;
        double vn = 0;
        for (double z : v)
            vn = std::hypot(vn, z);
        for (double& z : v)
            z /= vn;
        for (int j = k; j < n; ++j) {
            double dot = 0;
            for (int i = k; i < m; ++i)
                dot += v[i - k] * a[i][j];
            for (int i = k; i < m; ++i)
                a[i][j] -= 2 * v[i - k] * dot;
        }
        double dot = 0;
        for (int i = k; i < m; ++i)
            dot += v[i - k] * y[i];
        for (int i = k; i < m; ++i)
            y[i] -= 2 * v[i - k] * dot;
    }
    std::vector<double> c(n);
    for (int i = n - 1; i >= 0; --i) {
        double rhs = y[i];
        for (int j = i + 1; j < n; ++j)
            rhs -= a[i][j] * c[j];
        c[i] = rhs / a[i][i];
    }
    // Horner composition q(2*p/extent-1).
    std::vector<double> result(1, c.back());
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
    // This helper is also called directly by stages, without the TOML validator.
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
    result.coefficients = fitPolynomial(measured, std::move(target), n);
    return result;
}
} // namespace yk::spectral::preprocess
