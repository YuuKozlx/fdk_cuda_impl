#include "config/SimConfig.hpp"
#include "spek_spectrum/spectrum.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace yk::spectral {
void validateSpectrumSource(const SpectrumSourceConfig& c)
{
    if (c.source != "file" && c.source != "generated")
        throw std::runtime_error("spectrum.source must be file or generated");
    if (c.source == "file")
        return;
    if (c.model != "spekcalc" && c.model != "spekpy-v1")
        throw std::runtime_error("spectrum.model must be spekcalc or spekpy-v1");
    if (!std::isfinite(c.kvp) || c.kvp <= 1 || !std::isfinite(c.bin_width_keV) || c.bin_width_keV <= 0 || c.bin_width_keV > c.kvp - 1 || (c.kvp - 1) / c.bin_width_keV > 8192 || !std::isfinite(c.anode_angle_deg) || c.anode_angle_deg <= 0 || c.anode_angle_deg >= 90)
        throw std::runtime_error("invalid spectrum voltage, angle or bin width (maximum 8192 bins)");
    for (const auto& f : c.filters) {
        if (!std::isfinite(f.thickness_mm) || f.thickness_mm < 0)
            throw std::runtime_error("spectrum filter thickness must be finite and nonnegative");
        if (f.atomic_number != 0 || f.density_g_cm3 != 0) {
            if (!f.material.empty() || f.atomic_number < 1 || f.atomic_number > 92 || !std::isfinite(f.density_g_cm3) || f.density_g_cm3 <= 0)
                throw std::runtime_error("elemental filter requires Z=1..92 and positive density, without material name");
        } else {
            const std::string names[] = { "Be", "Ti", "Al", "Cu", "Mo", "Rh", "Ag", "Sn", "W", "Au", "Pb" };
            if (std::find(std::begin(names), std::end(names), f.material) == std::end(names))
                throw std::runtime_error("unsupported spectrum filter material: " + f.material);
        }
    }
}

void saveSpectrum(const std::vector<SpectrumPoint>& points, const std::filesystem::path& path)
{
    if (points.empty())
        throw std::runtime_error("cannot save empty spectrum");
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot create spectrum output: " + path.string());
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << "# Format: YK-SPECTRUM 1.0 (center energies, explicit bin widths)\n"
        << "# Energy[keV] N[keV cm^2 mAs]^-1 @ 1 meter bin_width[keV]\n"
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& p : points)
        out << p.energy_keV << ' ' << p.fluence_per_keV_cm2_mAs_at_1m << ' ' << p.bin_width_keV << '\n';
    out.close();
}

std::vector<SpectrumPoint> loadProjectionSpectrum(const ProjectionConfig& c)
{
    validateSpectrumSource(c.spectrum);
    std::vector<SpectrumPoint> points;
    if (c.spectrum.source == "file") {
        points = loadSpectrum(c.spectrum_file);
    } else {
        if (!c.spectrum_file.empty())
            throw std::runtime_error("generated spectrum conflicts with spectrum_file");
        spek_spectrum::Config generator;
        generator.model = c.spectrum.model == "spekcalc" ? spek_spectrum::Model::spekcalc : spek_spectrum::Model::spekpy_v1;
        generator.kvp = c.spectrum.kvp;
        generator.anode_angle_deg = c.spectrum.anode_angle_deg;
        generator.bin_width_keV = c.spectrum.bin_width_keV;
        generator.mas = 1.0;
        generator.distance_cm = 100.0;
        for (const auto& f : c.spectrum.filters)
            generator.filters.push_back({ f.material, f.thickness_mm, f.atomic_number, f.density_g_cm3 });
        const auto result = spek_spectrum::generate(generator);
        double total = 0;
        for (std::size_t i = 0; i < result.energy_keV.size(); ++i) {
            const double energy = result.energy_keV[i], fluence = result.fluence_per_keV_cm2[i];
            if (!std::isfinite(energy) || energy <= 0 || !std::isfinite(fluence) || fluence < 0)
                throw std::runtime_error("spectrum generator produced invalid values");
            points.push_back({ energy, fluence, generator.bin_width_keV });
            total += fluence * generator.bin_width_keV;
        }
        if (!std::isfinite(total) || total <= 0)
            throw std::runtime_error("generated spectrum has no photon fluence");
    }
    if (!c.spectrum.export_file.empty())
        saveSpectrum(points, c.spectrum.export_file);
    return points;
}
} // namespace yk::spectral
