#include "ReconstructionPreprocessor.hpp"
#include "primary/AirReferenceReader.hpp"

#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"
#include "XcomAttenuationProvider.hpp"
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
#include "AdaptiveScatterSolver.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace yk::spectral {
namespace {

void check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

float harden(float value, const BeamHardeningConfig& config)
{
    double result = 0.0;
    for (auto it = config.coefficients.rbegin(); it != config.coefficients.rend(); ++it)
        result = result * static_cast<double>(value) + *it;
    check(std::isfinite(result), "beam hardening polynomial overflow");
    return static_cast<float>(std::clamp(result, config.clamp_min, config.clamp_max));
}

std::vector<double> solveLinearSystem(std::vector<std::vector<double>> matrix,
                                      std::vector<double> rhs)
{
    const int n = static_cast<int>(rhs.size());
    for (int column = 0; column < n; ++column) {
        int pivot = column;
        for (int row = column + 1; row < n; ++row)
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
        check(std::abs(matrix[pivot][column]) > 1.0e-14,
              "water-plate beam hardening fit is singular");
        std::swap(matrix[pivot], matrix[column]);
        std::swap(rhs[pivot], rhs[column]);
        const double scale = matrix[column][column];
        for (int j = column; j < n; ++j) matrix[column][j] /= scale;
        rhs[column] /= scale;
        for (int row = 0; row < n; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            for (int j = column; j < n; ++j) matrix[row][j] -= factor * matrix[column][j];
            rhs[row] -= factor * rhs[column];
        }
    }
    return rhs;
}

BeamHardeningConfig fitWaterPlate(const SimulationConfig& config,
                                  const BeamHardeningConfig& input)
{
    const auto spectrum = input.spectrum_file.empty()
        ? loadProjectionSpectrum(config.projection) : loadSpectrum(input.spectrum_file);
    std::vector<double> energies;
    double photon_weight_sum = 0.0;
    double energy_weight_sum = 0.0;
    double mean_energy = 0.0;
    for (const auto& point : spectrum) {
        const double weight = point.fluence_per_keV_cm2_mAs_at_1m * point.bin_width_keV;
        energies.push_back(point.energy_keV);
        photon_weight_sum += weight;
        energy_weight_sum += weight * point.energy_keV;
        mean_energy += weight * point.energy_keV;
    }
    check(photon_weight_sum > 0.0 && std::isfinite(energy_weight_sum) && energy_weight_sum > 0.0,
          "spectrum has no finite positive energy fluence");
    // The reference energy remains the photon-spectrum mean. The measured
    // polychromatic signal, however, must use energy-integrating weights.
    mean_energy /= photon_weight_sum;
    const double reference_energy = input.reference_energy_keV > 0.0
        ? input.reference_energy_keV : mean_energy;
    MaterialSpec water;
    water.label = 1;
    water.name = "water";
    water.formula = "H2O";
    water.density_g_cm3 = 1.0;
    XcomAttenuationProvider attenuation({});
    std::vector<double> mass;
    std::string error;
    check(attenuation.query(water, energies, mass, error), "water attenuation query failed: " + error);
    std::vector<double> reference_mass;
    check(attenuation.query(water, {reference_energy}, reference_mass, error),
          "reference water attenuation query failed: " + error);
    const double mu_reference = reference_mass.front();
    const int n = input.order + 1;
    std::vector<std::vector<double>> normal(n, std::vector<double>(n, 0.0));
    std::vector<double> rhs(n, 0.0);
    for (int sample = 0; sample < input.water_plate_samples; ++sample) {
        const double thickness_mm = input.water_plate_max_mm * sample /
            static_cast<double>(input.water_plate_samples - 1);
        double transmitted = 0.0;
        for (size_t e = 0; e < spectrum.size(); ++e)
            transmitted += spectrum[e].fluence_per_keV_cm2_mAs_at_1m *
                spectrum[e].bin_width_keV * spectrum[e].energy_keV *
                std::exp(-mass[e] * thickness_mm / 10.0);
        const double multi = -std::log(std::max(transmitted / energy_weight_sum,
                                                 std::numeric_limits<double>::min()));
        const double mono = mu_reference * thickness_mm / 10.0;
        std::vector<double> powers(static_cast<size_t>(n), 1.0);
        for (int j = 1; j < n; ++j) powers[j] = powers[j - 1] * multi;
        for (int row = 0; row < n; ++row) {
            rhs[row] += powers[row] * mono;
            for (int column = 0; column < n; ++column)
                normal[row][column] += powers[row] * powers[column];
        }
    }
    BeamHardeningConfig result = input;
    result.coefficients = solveLinearSystem(std::move(normal), std::move(rhs));
    return result;
}

}

bool prepareReconstructionProjection(const SimulationConfig& config,
                                     const std::filesystem::path& input_projection,
                                     const std::filesystem::path& output_projection,
                                     std::string& diagnostic)
{
    try {
        const auto& g = config.geometry.parameters;
        const auto& pp = config.reconstruction.preprocessing;
        BeamHardeningConfig beam_hardening = pp.beam_hardening;
        if (beam_hardening.enabled && beam_hardening.fit_from_water_plate)
            beam_hardening = fitWaterPlate(config, beam_hardening);
        const std::size_t pixels = static_cast<std::size_t>(g.detector_u) * g.detector_v;
        const std::size_t elements = static_cast<std::size_t>(g.views) * pixels;
        check(pp.enabled, "reconstruction preprocessing is disabled");
        check(g.views > 0 && g.detector_u > 0 && g.detector_v > 0,
              "invalid preprocessing detector dimensions");
        check(std::filesystem::absolute(input_projection).lexically_normal() !=
              std::filesystem::absolute(output_projection).lexically_normal(),
              "preprocessing output must not overwrite input");
        check(std::filesystem::is_regular_file(input_projection),
              "preprocessing input projection does not exist: " + input_projection.string());

        std::ifstream input(input_projection, std::ios::binary);
        check(static_cast<bool>(input), "cannot open preprocessing input projection");
        input.seekg(0, std::ios::end);
        check(static_cast<std::uint64_t>(input.tellg()) == elements * sizeof(float),
              "preprocessing input projection size does not match geometry");
        input.seekg(0, std::ios::beg);

        std::vector<float> air(pixels, 0.0f);
        std::unique_ptr<primary_detail::AirReferenceReader> air_reader;
        std::unique_ptr<FasksRuntimeModel> scatter_model;
        std::unique_ptr<CudaFasksScatter> scatter;
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
        std::unique_ptr<experimental::AdaptiveScatterSolver> adaptive_solver;
#endif
        if (pp.scatter.enabled) {
            air_reader = std::make_unique<primary_detail::AirReferenceReader>(
                pp.scatter.air_projection_file, pixels, g.views);

            scatter_model = std::make_unique<FasksRuntimeModel>(pp.scatter.model_file);
            scatter = std::make_unique<CudaFasksScatter>(
                g.detector_u, g.detector_v, g.pixel_u_mm, g.pixel_v_mm, *scatter_model,
                pp.scatter.kernel.mode == "segmented" ? pp.scatter.kernel.segments : 0);
            if (pp.scatter.solver.type == "adaptive_experimental") {
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
                adaptive_solver = std::make_unique<experimental::AdaptiveScatterSolver>(pixels);
#else
                throw std::runtime_error("adaptive scatter plugin is not built; enable MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER");
#endif
            }
        }

        const auto parent = output_projection.parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream output(output_projection, std::ios::binary);
        check(static_cast<bool>(output), "cannot create preprocessed projection");

        std::vector<float> log_projection(pixels);
        std::vector<float> primary(pixels);
        std::vector<float> measured(pixels);
        std::vector<float> corrected(pixels);
        std::vector<unsigned char> mask(pixels);
        std::vector<float> guide_measured, guide_air, guide_scatter;
        for (int view = 0; view < g.views; ++view) {
            input.read(reinterpret_cast<char*>(log_projection.data()),
                       static_cast<std::streamsize>(log_projection.size() * sizeof(float)));
            check(static_cast<bool>(input), "cannot read projection frame for preprocessing");
            for (float value : log_projection)
                check(std::isfinite(value), "projection contains a non-finite value");

            if (pp.scatter.enabled) {
                air_reader->read(view, air);
                for (std::size_t p = 0; p < pixels; ++p) {
                    const float value = log_projection[p];
                    check(std::isfinite(value), "projection contains a non-finite value");
                    primary[p] = air[p] * std::exp(-value);
                    check(std::isfinite(primary[p]), "restored energy signal overflow");
                    measured[p] = primary[p];
                    mask[p] = static_cast<unsigned char>(
                        value > pp.scatter.correction.object_mask_threshold);
                }
                const auto& guide = pp.scatter.guide;
                if (guide.enabled) {
                    scatter->smoothGuide(measured, guide.sigma_u_pixels, guide.sigma_v_pixels, guide_measured);
                    scatter->smoothGuide(air, guide.sigma_u_pixels, guide.sigma_v_pixels, guide_air);
                    primary = guide_measured;
                    for (size_t p = 0; p < pixels; ++p)
                        mask[p] = static_cast<unsigned char>(-std::log(std::max(
                            static_cast<double>(guide_measured[p])/guide_air[p], 1e-30))
                            > pp.scatter.correction.object_mask_threshold);
                }
                const auto& solve_measured = guide.enabled ? guide_measured : measured;
                const auto& solve_air = guide.enabled ? guide_air : air;
                if (pp.scatter.solver.type == "adaptive_experimental") {
#ifdef YK_SPECTRAL_ADAPTIVE_SCATTER
                    const auto solve_start = std::chrono::steady_clock::now();
                    const auto report = adaptive_solver->solve(solve_measured, solve_air, primary,
                        pp.scatter.solver.adaptive, pp.scatter.correction.scatter_scale,
                        [&](const auto& estimate, auto& prediction) {
                            scatter->predict(estimate, solve_air, mask, prediction);
                        }, &mask, [&](const float* estimate, const float* device_air, float* prediction) {
                            scatter->predictDevice(estimate, device_air, mask, prediction);
                        });
                    for (const auto& step : report.history)
                        std::cout << "scatter_iteration frame=" << view + 1
                                  << " iteration=" << step.iteration
                                  << " damping=" << step.damping
                                  << " residual=" << step.relative_residual
                                  << " change=" << step.relative_change << '\n';
                    std::cout << "adaptive scatter frame " << view + 1 << "/" << g.views
                              << " status=" << report.status << " iterations=" << report.iterations
                              << " predictions=" << report.predictions
                              << " rejected=" << report.rejected_steps
                              << " positivity_limited=" << report.positivity_limited_steps
                              << " low_transmission=" << report.low_transmission
                              << " transmission_risk=" << report.low_transmission_risk
                              << " damping=" << report.damping
                              << " residual=" << report.relative_residual
                              << " change=" << report.relative_change
                              << " elapsed_ms=" << std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - solve_start).count() << '\n';
#endif
                } else {
                    scatter->correctFixed(solve_measured, solve_air, mask,
                        pp.scatter.correction.iterations, pp.scatter.correction.damping,
                        pp.scatter.correction.scatter_scale, primary);
                }
                if (guide.enabled) {
                    scatter->predict(primary, guide_air, mask, guide_scatter);
                    size_t clipped = 0;
                    for (size_t p = 0; p < pixels; ++p) {
                        check(std::isfinite(guide_scatter[p]), "guide scatter is not finite");
                        const float residual = measured[p] - static_cast<float>(pp.scatter.correction.scatter_scale)
                            * std::max(0.0f, guide_scatter[p]);
                        const float floor = air[p]*1e-8f;
                        if (residual < floor) ++clipped;
                        primary[p] = std::max(floor, residual);
                    }
                    std::cout << "scatter_guide frame=" << view+1 << " clipped_pixels=" << clipped
                              << " total_pixels=" << pixels << '\n';
                }
                for (std::size_t p = 0; p < pixels; ++p) {
                    const double ratio = std::max(
                        static_cast<double>(primary[p]) / air[p],
                        static_cast<double>(std::numeric_limits<float>::min()));
                    corrected[p] = static_cast<float>(-std::log(ratio));
                }
            } else {
                corrected = log_projection;
            }

            if (beam_hardening.enabled)
                for (float& value : corrected) value = harden(value, beam_hardening);
            output.write(reinterpret_cast<const char*>(corrected.data()),
                         static_cast<std::streamsize>(corrected.size() * sizeof(float)));
            check(static_cast<bool>(output), "cannot write preprocessed projection");
        }
        output.close();
        check(static_cast<bool>(output), "cannot flush preprocessed projection");
        diagnostic = "reconstruction preprocessing complete: " + output_projection.string();
        return true;
    } catch (const std::exception& error) {
        diagnostic = error.what();
        return false;
    }
}

}
