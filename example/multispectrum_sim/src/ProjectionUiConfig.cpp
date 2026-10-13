#include "ProjectionUiConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <toml++/toml.hpp>
namespace yk::spectral {
namespace {
    const char* kind(GeometryKind k)
    {
        switch (k) {
        case GeometryKind::FlatCbct:
            return "flat_cbct";
        case GeometryKind::FlatHelical:
            return "flat_helical";
        case GeometryKind::PlanarSourceOrbit:
            return "planar_source_orbit";
        case GeometryKind::NikonTilted:
            return "nikon_tilted";
        default:
            throw std::runtime_error("ImGui supports only the four flat detector scan types");
        }
    }
    void rebase(toml::node& n, const std::filesystem::path& base)
    {
        if (auto* table = n.as_table()) {
            for (auto&& [key, value] : *table) {
                const std::string k(key.str());
                if (value.is_string() && (k == "label_volume" || k == "slice_prefix" || k == "snapshot_directory" || (k.size() > 5 && k.substr(k.size() - 5) == "_file"))) {
                    auto path = value.value<std::string>().value();
                    if (!path.empty()) {
                        std::filesystem::path p(path);
                        if (p.is_relative())
                            *value.as_string() = (base / p).lexically_normal().generic_string();
                    }
                } else
                    rebase(value, base);
            }
        } else if (auto* array = n.as_array())
            for (auto& value : *array)
                rebase(value, base);
    }
}
void saveProjectionUiConfig(const SimulationConfig& c, const std::filesystem::path& path, WorkflowMode mode)
{
    toml::table t;
    if (!c.source_config_file.empty()) {
        t = toml::parse_file(c.source_config_file.string());
        rebase(t, c.source_config_file.parent_path());
    }
    t.insert_or_assign("schema_version", 2);
    t.insert_or_assign("workflow", toml::table { { "mode", mode==WorkflowMode::Project?"project":mode==WorkflowMode::Reconstruct?"reconstruct":"project_and_reconstruct" } });
    auto& g = c.geometry.parameters;
    toml::table geometry = t["geometry"].is_table() ? *t["geometry"].as_table() : toml::table{};
    geometry.insert_or_assign("kind", kind(c.geometry.kind));
#define FIELD(name) geometry.insert_or_assign(#name, g.name)
    FIELD(views);
    FIELD(views_per_turn);
    FIELD(rotation_direction);
    FIELD(detector_u);
    FIELD(detector_v);
    FIELD(pixel_u_mm);
    FIELD(pixel_v_mm);
    FIELD(sid_mm);
    FIELD(sdd_mm);
    FIELD(offset_u_mm);
    FIELD(offset_n_mm);
    FIELD(offset_v_mm);
    FIELD(tilt_u_rad);
    FIELD(tilt_v_rad);
    FIELD(tilt_n_rad);
    FIELD(source_offset_x_mm);
    FIELD(source_offset_y_mm);
    FIELD(source_offset_z_mm);
    FIELD(start_angle_rad);
    FIELD(pitch_mm_per_turn);
    FIELD(start_z_mm);
    FIELD(scan_range_rad);
    FIELD(planar_detector_y_mm);
    FIELD(source_orbit_radius_mm);
    FIELD(source_lateral_offset_mm);
    FIELD(reconstruction_detector_u);
    FIELD(reconstruction_detector_v);
    FIELD(planar_resample_mode);
    FIELD(nikon_tilt_rad);
    FIELD(nikon_tilt_axis_azimuth_rad);
    FIELD(rotation_axis_pivot_x_mm);
    FIELD(rotation_axis_pivot_y_mm);
    FIELD(rotation_axis_pivot_z_mm);
    FIELD(voxel_x_mm);
    FIELD(voxel_y_mm);
    FIELD(voxel_z_mm);
    FIELD(volume_x);
    FIELD(volume_y);
    FIELD(volume_z);
    FIELD(phantom_offset_x_mm);
    FIELD(phantom_offset_y_mm);
    FIELD(phantom_offset_z_mm);
    FIELD(phantom_rotation_x_rad);
    FIELD(phantom_rotation_y_rad);
    FIELD(phantom_rotation_z_rad);
    FIELD(reconstruction_volume_x);FIELD(reconstruction_volume_y);FIELD(reconstruction_volume_z);
    FIELD(reconstruction_voxel_x_mm);FIELD(reconstruction_voxel_y_mm);FIELD(reconstruction_voxel_z_mm);
    FIELD(reconstruction_offset_x_mm);FIELD(reconstruction_offset_y_mm);FIELD(reconstruction_offset_z_mm);
#undef FIELD
    t.insert_or_assign("geometry", std::move(geometry));
    auto* p = t["projection"].as_table();
    if (!p) {
        t.insert("projection", toml::table {});
        p = t["projection"].as_table();
    }
    p->insert_or_assign("label_volume", c.projection.label_volume.generic_string());
    p->insert_or_assign("output_file", c.projection.output_file.generic_string());
    p->insert_or_assign("energy_output_file", c.projection.energy_output_file.generic_string());
    p->insert_or_assign("air_output_file", c.projection.air_output_file.generic_string());
    p->insert_or_assign("mAs_per_view", c.projection.mAs_per_view);
    p->insert_or_assign("save_metadata", c.projection.save_metadata);
    p->insert_or_assign("batching",toml::table{{"views",c.projection.batching.views},{"max_buffer_mib",c.projection.batching.max_buffer_mib}});
    p->insert_or_assign("sampling", toml::table { { "samples_per_pixel", c.projection.sampling.samples_per_pixel }, { "spatial_seed", c.projection.sampling.seed } });
    p->insert_or_assign("geometry_flux", toml::table { { "enabled", c.projection.geometry_flux.enabled } });
    p->insert_or_assign("quantum_noise", toml::table { { "enabled", c.projection.quantum_noise.enabled }, { "seed", c.projection.quantum_noise.seed } });
    const auto& sp = c.projection.spectrum;
    toml::table spectrum { { "source", sp.source } };
    if (sp.source == "generated") {
        p->erase("spectrum_file");
        spectrum.insert("model", sp.model);
        spectrum.insert("kvp", sp.kvp);
        spectrum.insert("anode_angle_deg", sp.anode_angle_deg);
        spectrum.insert("bin_width_keV", sp.bin_width_keV);
        spectrum.insert("export_file", sp.export_file.generic_string());
        toml::array filters;
        for (const auto& f : sp.filters) {
            toml::table filter { { "thickness_mm", f.thickness_mm } };
            if (f.atomic_number) {
                filter.insert("atomic_number", f.atomic_number);
                filter.insert("density_g_cm3", f.density_g_cm3);
            } else
                filter.insert("material", f.material);
            filters.push_back(std::move(filter));
        }
        spectrum.insert("filters", std::move(filters));
    } else
        p->insert_or_assign("spectrum_file", c.projection.spectrum_file.generic_string());
    p->insert_or_assign("spectrum", std::move(spectrum));
    toml::array materials;
    for (const auto& m : c.projection.materials) {
        toml::table material { { "label", int(m.label) }, { "name", m.name }, { "density_g_cm3", m.density_g_cm3 } };
        if (!m.preset.empty())
            material.insert("preset", m.preset);
        else
            material.insert("formula", m.formula);
        materials.push_back(std::move(material));
    }
    p->insert_or_assign("materials", std::move(materials));
    const auto& f = c.projection.focal_spot;
    p->insert_or_assign("focal_spot", toml::table { { "enabled", f.enabled }, { "defocus_enabled", f.defocus_enabled }, { "normal_size_u_mm", f.normal_size_u_mm }, { "normal_size_v_mm", f.normal_size_v_mm }, { "defocus_size_u_mm", f.defocus_size_u_mm }, { "defocus_size_v_mm", f.defocus_size_v_mm }, { "epsilon", f.epsilon } });
    auto setEnabled = [&](toml::table& parent, const char* name, bool enabled) {if(!parent[name].is_table())parent.insert_or_assign(name,toml::table{});parent[name].as_table()->insert_or_assign("enabled",enabled); };
    setEnabled(*p, "scatter", c.projection.scatter.enabled);
    p->get_as<toml::table>("scatter")->insert_or_assign("model_file", c.projection.scatter.model_file.generic_string());
    setEnabled(*p, "detector_response", c.projection.detector_response.enabled);
    if (!(*p)["detector_postprocess"].is_table())
        p->insert_or_assign("detector_postprocess", toml::table {});
    auto& post = *p->get_as<toml::table>("detector_postprocess");
    setEnabled(post, "optical_crosstalk", c.projection.detector_postprocess.optical_crosstalk.enabled);
    setEnabled(post, "electronic_crosstalk", c.projection.detector_postprocess.electronic_crosstalk.enabled);
    setEnabled(post, "afterglow", c.projection.detector_postprocess.afterglow.enabled);
    setEnabled(post, "das", c.projection.detector_postprocess.das.enabled);
    const auto& after = c.projection.detector_postprocess.afterglow;
    post.get_as<toml::table>("afterglow")->insert_or_assign("p", after.p);
    auto& glow=*post.get_as<toml::table>("afterglow");
    glow.insert_or_assign("initialization",after.initialization);glow.insert_or_assign("frame_time_ms",after.frame_time_ms);glow.insert_or_assign("exposure_time_ms",after.exposure_time_ms);
    toml::array weights,time_constants;for(double v:after.weights)weights.push_back(v);for(double v:after.time_constants_ms)time_constants.push_back(v);
    glow.insert_or_assign("weights",std::move(weights));glow.insert_or_assign("time_constants_ms",std::move(time_constants));
    const auto& das = c.projection.detector_postprocess.das;
    post.get_as<toml::table>("das")->insert_or_assign("electronic_noise_std_electrons", das.electronic_noise_std_electrons);
    const auto& r=c.reconstruction;
    toml::table recon=t["reconstruction"].is_table()?*t["reconstruction"].as_table():toml::table{};
    recon.insert_or_assign("type",r.type);recon.insert_or_assign("input_projection_file",r.input_projection_file.generic_string());
    recon.insert_or_assign("output_volume_file",r.output_volume_file.generic_string());recon.insert_or_assign("slice_prefix",r.slice_prefix.generic_string());recon.insert_or_assign("save_slices",r.save_slices);
    recon.erase("analytic");recon.erase("iterative");recon.erase("statistical");
    if(r.type=="analytic")recon.insert("analytic",toml::table{{"pipeline",r.analytic.pipeline},{"filter",r.analytic.filter},{"chunk_views",r.analytic.chunk_views}});
    if(r.type=="analytic"){
        auto* a=recon["analytic"].as_table();
        a->insert_or_assign("cutoff",r.analytic.cutoff);
        a->insert_or_assign("gain",r.analytic.gain);
        if(r.analytic.filter=="butterworth")a->insert_or_assign("butterworth_order",r.analytic.butterworth_order);
        if(r.analytic.filter=="kaiser")a->insert_or_assign("kaiser_beta",r.analytic.kaiser_beta);
        if(r.analytic.filter=="tukey")a->insert_or_assign("tukey_alpha",r.analytic.tukey_alpha);
    }
    if(r.type=="iterative") recon.insert("iterative",toml::table{{"algorithm",r.iterative.algorithm},{"iterations",r.iterative.iterations},{"relaxation",r.iterative.relaxation},{"subsets",r.iterative.subsets},
        {"forward_projector",r.iterative.forward_projector},{"back_projector",r.iterative.back_projector},{"tv_iterations",r.iterative.tv_iterations},{"tv_alpha",r.iterative.tv_alpha},{"tv_alpha_reduction",r.iterative.tv_alpha_reduction},
        {"maximum_update_ratio",r.iterative.maximum_update_ratio},{"non_negative",r.iterative.non_negative},
        {"regularization",toml::table{{"plugin",r.iterative.regularization.plugin},{"strength",r.iterative.regularization.strength},
          {"huber_delta",r.iterative.regularization.huber_delta},{"step_size",r.iterative.regularization.step_size},
          {"inner_iterations",r.iterative.regularization.inner_iterations},{"dimensions",r.iterative.regularization.dimensions},
          {"epsilon",r.iterative.regularization.epsilon},{"strength_reduction",r.iterative.regularization.strength_reduction}}}});
    if(r.type=="iterative"&&(r.iterative.regularization.plugin=="huber_tv"||r.iterative.regularization.plugin=="huber_tv_prox")){
        auto* table=recon["iterative"]["regularization"].as_table();
        table->erase("inner_iterations");table->erase("step_size");table->erase("strength_reduction");
    }
    if(r.type=="statistical"){
        const auto& s=r.statistical;
        recon.insert("statistical",toml::table{{"forward_projector",s.forward_projector},{"back_projector",s.back_projector},{"non_negative",s.non_negative},
            {"data_fidelity",toml::table{{"type",s.data_fidelity.type},{"scale",s.data_fidelity.scale},
                {"normalize_by_ray_count",s.data_fidelity.normalize_by_ray_count},{"require_verified_adjoint",s.data_fidelity.require_verified_adjoint},
                {"weights_file",s.data_fidelity.weights_file.generic_string()}}},
            {"regularization",toml::table{{"plugin",s.regularization.plugin},{"strength",s.regularization.strength},
                {"shape",s.regularization.shape},{"dimensions",s.regularization.dimensions}}},
            {"optimizer",toml::table{{"type",s.optimizer.type},{"iterations",s.optimizer.iterations},{"step_size",s.optimizer.step_size},
                {"backtracking",s.optimizer.backtracking},{"max_backtracking",s.optimizer.max_backtracking},{"step_shrink",s.optimizer.step_shrink},
                {"relative_tolerance",s.optimizer.relative_tolerance},{"minimum_iterations",s.optimizer.minimum_iterations}}}});
    }
    auto* pp=recon["preprocessing"].as_table();if(!pp){recon.insert("preprocessing",toml::table{});pp=recon["preprocessing"].as_table();}
    pp->insert_or_assign("enabled",r.preprocessing.enabled);
    const auto& bh=r.preprocessing.beam_hardening;
    toml::table hardening{{"enabled",bh.enabled},{"fit_from_water_plate",bh.fit_from_water_plate},{"order",bh.order},
        {"clamp_min",bh.clamp_min},{"clamp_max",bh.clamp_max},{"water_plate_max_mm",bh.water_plate_max_mm},{"water_plate_samples",bh.water_plate_samples},{"reference_energy_keV",bh.reference_energy_keV}};
    if(!bh.spectrum_file.empty())hardening.insert("spectrum_file",bh.spectrum_file.generic_string());
    toml::array coefficients;for(double v:bh.coefficients)coefficients.push_back(v);hardening.insert("coefficients",std::move(coefficients));
    pp->insert_or_assign("beam_hardening",std::move(hardening));
    setEnabled(*pp,"scatter",r.preprocessing.scatter.enabled);
    const auto& sc=r.preprocessing.scatter;auto& scatter=*pp->get_as<toml::table>("scatter");
    scatter.insert_or_assign("air_projection_file",sc.air_projection_file.generic_string());scatter.insert_or_assign("model_file",sc.model_file.generic_string());
    toml::table correction{{"scatter_scale",sc.correction.scatter_scale},{"object_mask_threshold",sc.correction.object_mask_threshold}};
    if(sc.solver.type=="fixed"){correction.insert("iterations",sc.correction.iterations);correction.insert("damping",sc.correction.damping);}
    scatter.insert_or_assign("correction",std::move(correction));
    toml::table kernel{{"mode",sc.kernel.mode}};if(sc.kernel.mode=="segmented")kernel.insert("segments",sc.kernel.segments);scatter.insert_or_assign("kernel",std::move(kernel));
    auto* solver=scatter["solver"].as_table();if(!solver){scatter.insert("solver",toml::table{});solver=scatter["solver"].as_table();}solver->insert_or_assign("type",sc.solver.type);
    solver->erase("adaptive");
    if(sc.solver.type=="adaptive_experimental"){
        const auto& a=sc.solver.adaptive;
        solver->insert("adaptive",toml::table{{"iterations",a.iterations},{"initial_damping",a.initial_damping},{"min_damping",a.min_damping},{"damping_decay",a.damping_decay},
            {"transmission_guard",toml::table{{"low_quantile",a.transmission_guard.low_quantile},{"risk_threshold",a.transmission_guard.risk_threshold},{"positivity_margin",a.transmission_guard.positivity_margin}}}});
    }
    scatter.insert_or_assign("guide",toml::table{{"enabled",sc.guide.enabled},{"sigma_u_pixels",sc.guide.sigma_u_pixels},{"sigma_v_pixels",sc.guide.sigma_v_pixels}});

    t.insert_or_assign("reconstruction",std::move(recon));
    const bool interactive=mode==WorkflowMode::Project;
    t.insert_or_assign("visualization", toml::table{{"enabled",interactive},{"interactive",interactive},{"volume_opacity_scale",c.visualization.volume_opacity_scale},{"auto_load_reconstruction",c.visualization.auto_load_reconstruction},
        {"texture_max_size",c.visualization.texture_max_size},{"max_fps",c.visualization.max_fps},{"projection",c.visualization.projection},
        {"hold_open",interactive},{"offscreen",false},{"snapshot_directory",c.visualization.snapshot_directory.generic_string()}});

    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot save UI configuration");
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << t;
    out.close();
}
void selectBuiltinMaterial(MaterialSpec& material,std::string_view id) {
    const auto* preset=findBuiltinMaterial(id);
    if(!preset)throw std::runtime_error("Unknown builtin material: "+std::string(id));
    material.preset=preset->id;
    material.formula.clear();
    material.name=preset->display_name;
    material.density_g_cm3=preset->density_g_cm3;
}
void validateProjectionUiInputs(const SimulationConfig& c, const std::vector<std::uint8_t>& labels)
{
    kind(c.geometry.kind);
    const auto& g = c.geometry.parameters;
    if (g.volume_x <= 0 || g.volume_y <= 0 || g.volume_z <= 0 || static_cast<std::uint64_t>(g.volume_x) * g.volume_y * g.volume_z != labels.size())
        throw std::runtime_error("Raw uint8 file size must equal X * Y * Z");
    if (g.sdd_mm <= g.sid_mm)
        throw std::runtime_error("SDD must be greater than SID");
    if (c.projection.materials.empty() || c.projection.materials.size() > 32)
        throw std::runtime_error("Use 1..32 material mappings");
    std::array<bool, 256> defined {};
    for (const auto& m : c.projection.materials) {
        if (defined[m.label])
            throw std::runtime_error("Duplicate material label");
        defined[m.label] = true;
    }
    for (const auto label : labels)
        if (label != 0 && !defined[label])
            throw std::runtime_error("Missing material for label " + std::to_string(label));
}
}
