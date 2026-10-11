#include "config/SimConfig.hpp"
#include "spek_spectrum/spectrum.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace yk::spectral;
void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}
void close(double a, double b, double tolerance, const char* message)
{
    require(std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b)), message);
}
int main()
{
    const auto dir = std::filesystem::current_path() / "spectrum-source-test-output";
    std::filesystem::create_directories(dir);
    try {
        ProjectionConfig c;
        c.spectrum.source = "generated";
        c.spectrum.filters = { { "Al", 2.0 } };
        c.mAs_per_view = 0.36;
        auto points = loadProjectionSpectrum(c);
        require(points.size() == 119, "wrong bin count");
        close(points.front().energy_keV, 1.5, 1e-12, "lost center energy");
        double photons = 0, energy = 0;
        for (auto p : points) {
            photons += p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
            energy += p.energy_keV * p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
        }
        close(energy / photons, 54.001898, 5e-4, "mean-energy regression");
        c.mAs_per_view = 8;
        const auto same = loadProjectionSpectrum(c);
        close(same[60].fluence_per_keV_cm2_mAs_at_1m, points[60].fluence_per_keV_cm2_mAs_at_1m, 1e-12, "dose applied twice");
        c.spectrum.export_file = dir / "generated.spec";
        points = loadProjectionSpectrum(c);
        const auto roundtrip = loadSpectrum(c.spectrum.export_file);
        require(roundtrip.size() == points.size(), "roundtrip count");
        for (std::size_t i = 0; i < points.size(); ++i) {
            close(roundtrip[i].energy_keV, points[i].energy_keV, 1e-12, "roundtrip energy");
            close(roundtrip[i].bin_width_keV, points[i].bin_width_keV, 1e-12, "roundtrip width");
            close(roundtrip[i].fluence_per_keV_cm2_mAs_at_1m, points[i].fluence_per_keV_cm2_mAs_at_1m, 1e-12, "roundtrip fluence");
        }
        spek_spectrum::Config native;
        native.filters = { { "Al", 2.0 } };
        native.mas = 3;
        native.distance_cm = 50;
        const auto native_result = spek_spectrum::generate(native);
        const auto mcgpu = dir / "mcgpu.spec";
        spek_spectrum::write_mcgpu_spec(native_result, native, mcgpu);
        const auto imported = loadSpectrum(mcgpu);
        require(imported.size() == points.size(), "MCGPU endpoint was treated as energy bin");
        for (std::size_t i = 0; i < points.size(); ++i) {
            close(imported[i].energy_keV, points[i].energy_keV, 1e-12, "MCGPU edge conversion");
            close(imported[i].fluence_per_keV_cm2_mAs_at_1m, points[i].fluence_per_keV_cm2_mAs_at_1m, 1e-10, "MCGPU normalization");
        }
        c.spectrum.filters = { { "Al", 4.0 } };
        const auto filtered = loadProjectionSpectrum(c);
        double hardPhotons = 0, hardEnergy = 0;
        for (auto p : filtered) {
            hardPhotons += p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
            hardEnergy += p.energy_keV * p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
        }
        require(hardPhotons < photons && hardEnergy / hardPhotons > energy / photons, "filtration physics");
        c.spectrum.model = "spekcalc";
        c.spectrum.filters = { { "Al", 2.0 } };
        const auto legacy = loadProjectionSpectrum(c);
        photons = 0;
        energy = 0;
        for (auto p : legacy) {
            photons += p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
            energy += p.energy_keV * p.fluence_per_keV_cm2_mAs_at_1m * p.bin_width_keV;
        }
        close(energy / photons, 53.334190, 5e-4, "spekcalc regression");
        const auto config_file = dir / "generated.toml";
        const std::string base = "schema_version=2\n[workflow]\nmode='project'\n[geometry]\nkind='flat_cbct'\n"
                                 "[projection]\n[projection.geometry_flux]\nenabled=true\n"
                                 "[projection.quantum_noise]\nenabled=false\n[projection.sampling]\nsamples_per_pixel=1\n"
                                 "[[projection.materials]]\nlabel=1\nname='water'\nformula='H2O'\n"
                                 "[projection.spectrum]\n";
        auto parse = [&](std::string options, bool valid) {
            {
                std::ofstream f(config_file);
                f << base << options;
            }
            bool accepted = false;
            try {
                auto cfg = loadConfig(config_file);
                accepted = true;
                if (valid) {
                    require(cfg.projection.spectrum.source == "generated", "source not parsed");
                    require(cfg.projection.spectrum.export_file == dir / "export.spec", "relative export path");
                }
            } catch (const std::exception&) {
                if (valid)
                    throw;
            }
            require(accepted == valid, "config validity mismatch");
        };
        parse("source='generated'\nkvp=120.0\nexport_file='export.spec'\nfilters=[{material='Al', thickness_mm=2.0}]\n", true);
        parse("source='other'\n", false);
        parse("source='generated'\nkvp=nan\n", false);
        parse("source='generated'\nanode_angle_deg=0.0\n", false);
        parse("source='generated'\nbin_width_keV=0.0\n", false);
        parse("source='generated'\nfilters=[{material='Al',thickness_mm=-1.0}]\n", false);
        parse("source='generated'\nfilters=[{material='bogus',thickness_mm=1.0}]\n", false);
        parse("source='generated'\nkvp='bad'\n", false);
        parse("source='generated'\nfilters='bad'\n", false);
        {
            std::ofstream f(config_file);
            f << base << "source='generated'\n"
                         "[reconstruction]\ntype='analytic'\n"
                         "[reconstruction.analytic]\npipeline='fdk'\n"
                         "[reconstruction.preprocessing]\nenabled=true\n"
                         "[reconstruction.preprocessing.beam_hardening]\nenabled=true\nfit_from_water_plate=true\norder=2\n";
            f.close();
            const auto cfg = loadConfig(config_file);
            require(cfg.projection.spectrum_file.empty(), "generated calibration unexpectedly needs a file");
        }
        std::cout << "Spectrum generation, dose normalization, filtration, file roundtrip and configuration passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
