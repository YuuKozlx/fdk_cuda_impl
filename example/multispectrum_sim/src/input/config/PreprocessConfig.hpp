#pragma once
#include "input/config/SimConfig.hpp"
#include <toml++/toml.hpp>
namespace yk::spectral::config_detail {
// Native paths are relative to the process working directory, while TOML paths
// are relative to the saved file. Preserve their meaning when saving elsewhere.
inline std::string serializedPath(const std::filesystem::path& path) {
    return path.empty() ? std::string{} : std::filesystem::absolute(path).lexically_normal().generic_string();
}
inline toml::table preprocessingTable(const ReconstructionConfig& r)
{
    toml::table recon;
    recon.insert("preprocessing", toml::table { { "enabled", r.preprocessing.enabled } });
    auto setEnabled = [](toml::table& parent, const char* name, bool enabled) {
        if (!parent[name].is_table())
            parent.insert_or_assign(name, toml::table {});
        parent[name].as_table()->insert_or_assign("enabled", enabled);
    };
    auto* pp = recon["preprocessing"].as_table();
    if (!pp) {
        recon.insert("preprocessing", toml::table {});
        pp = recon["preprocessing"].as_table();
    }
    pp->insert_or_assign("enabled", r.preprocessing.enabled);
    pp->erase("filters");
    if (!r.preprocessing.filters.empty()) {
        toml::array filters;
        for (const auto& f : r.preprocessing.filters)
            filters.push_back(toml::table {
                { "operation", f.operation }, { "radius_pixels", f.radius_pixels }, { "sigma_pixels", f.sigma_pixels }, { "epsilon", f.epsilon } });
        pp->insert("filters", std::move(filters));
    }
    const auto& bh = r.preprocessing.beam_hardening;
    toml::table hardening { { "enabled", bh.enabled }, { "fit_from_water_plate", bh.fit_from_water_plate }, { "order", bh.order },
        { "clamp_min", bh.clamp_min }, { "clamp_max", bh.clamp_max }, { "water_plate_max_mm", bh.water_plate_max_mm }, { "water_plate_samples", bh.water_plate_samples }, { "reference_energy_keV", bh.reference_energy_keV } };
    hardening.insert("calibration_file", serializedPath(bh.calibration_file));
    if (!bh.spectrum_file.empty())
        hardening.insert("spectrum_file", serializedPath(bh.spectrum_file));
    toml::array coefficients;
    for (double v : bh.coefficients)
        coefficients.push_back(v);
    hardening.insert("coefficients", std::move(coefficients));
    pp->insert_or_assign("beam_hardening", std::move(hardening));
    setEnabled(*pp, "scatter", r.preprocessing.scatter.enabled);
    const auto& sc = r.preprocessing.scatter;
    auto& scatter = *pp->get_as<toml::table>("scatter");
    scatter.insert_or_assign("air_projection_file", serializedPath(sc.air_projection_file));
    scatter.insert_or_assign("model_file", serializedPath(sc.model_file));
    toml::table correction { { "scatter_scale", sc.correction.scatter_scale }, { "object_mask_threshold", sc.correction.object_mask_threshold } };
    if (sc.solver.type == "fixed") {
        correction.insert("iterations", sc.correction.iterations);
        correction.insert("damping", sc.correction.damping);
    }
    scatter.insert_or_assign("correction", std::move(correction));
    toml::table kernel { { "mode", sc.kernel.mode } };
    if (sc.kernel.mode == "segmented")
        kernel.insert("segments", sc.kernel.segments);
    scatter.insert_or_assign("kernel", std::move(kernel));
    auto* solver = scatter["solver"].as_table();
    if (!solver) {
        scatter.insert("solver", toml::table {});
        solver = scatter["solver"].as_table();
    }
    solver->insert_or_assign("type", sc.solver.type);
    solver->erase("adaptive");
    if (sc.solver.type == "adaptive_experimental") {
        const auto& a = sc.solver.adaptive;
        solver->insert("adaptive", toml::table { { "iterations", a.iterations }, { "initial_damping", a.initial_damping }, { "min_damping", a.min_damping }, { "damping_decay", a.damping_decay }, { "transmission_guard", toml::table { { "low_quantile", a.transmission_guard.low_quantile }, { "risk_threshold", a.transmission_guard.risk_threshold }, { "positivity_margin", a.transmission_guard.positivity_margin } } } });
    }
    scatter.insert_or_assign("guide", toml::table { { "enabled", sc.guide.enabled }, { "sigma_u_pixels", sc.guide.sigma_u_pixels }, { "sigma_v_pixels", sc.guide.sigma_v_pixels } });

    return *recon["preprocessing"].as_table();
}
}
