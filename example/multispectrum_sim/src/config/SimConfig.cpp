#include "config/SimConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include <toml++/toml.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace yk::spectral {
    namespace {
        GeometryKind geometry(std::string_view v) { if (v == "flat_cbct")return GeometryKind::FlatCbct; if (v == "flat_helical")return GeometryKind::FlatHelical; if (v == "cyl_cbct")return GeometryKind::CylCbct; if (v == "cyl_helical")return GeometryKind::CylHelical; if (v == "planar_source_orbit")return GeometryKind::PlanarSourceOrbit; if (v == "nikon_tilted")return GeometryKind::NikonTilted; throw std::runtime_error("unknown geometry: " + std::string(v)); }
        WorkflowMode workflow(std::string_view v) { if (v == "project")return WorkflowMode::Project; if (v == "reconstruct")return WorkflowMode::Reconstruct; if (v == "project_and_reconstruct")return WorkflowMode::ProjectAndReconstruct; throw std::runtime_error("未知 workflow.mode: " + std::string(v)); }
        std::filesystem::path resolve(const std::filesystem::path& b, std::string_view v) {
            if (v.empty()) return {};
            std::filesystem::path p(v);
            return p.is_absolute() ? p : b / p;
        }
    }
}
namespace yk::spectral {
    SimulationConfig loadConfig(const std::filesystem::path& path, bool check_runtime_files) {
        const auto available = [check_runtime_files](const std::filesystem::path& p) {
            return !check_runtime_files || std::filesystem::is_regular_file(p);
        };
        auto t = toml::parse_file(path.string());
        const auto config_directory = std::filesystem::absolute(path).parent_path();
        SimulationConfig c;
        c.source_config_file = std::filesystem::absolute(path);
        c.schema_version = t["schema_version"].value_or(0);
        if (c.schema_version != 2)
            throw std::runtime_error("配置必须声明 schema_version=2");
        if (t.contains("simulation") || t.contains("geometry_config") ||
            t.contains("materials") || t.contains("detector_effects"))
            throw std::runtime_error("检测到旧配置表；请使用 workflow/projection/geometry/reconstruction 结构");
        auto* p = t["projection"].as_table();
        auto* w = t["workflow"].as_table();
        if (!w) throw std::runtime_error("缺少 [workflow] 配置表");
        c.workflow.mode = workflow((*w)["mode"].value_or<std::string>("project"));
        if(auto* display=t["visualization"].as_table()) {
            auto& v=c.visualization;
            v.auto_load_reconstruction=(*display)["auto_load_reconstruction"].value_or(false);
            v.volume_opacity_scale=(*display)["volume_opacity_scale"].value_or(0.03);
            if(!std::isfinite(v.volume_opacity_scale)||v.volume_opacity_scale<0||v.volume_opacity_scale>1)
                throw std::runtime_error("visualization.volume_opacity_scale must be 0..1");
            v.enabled=(*display)["enabled"].value_or(false);
            v.interactive=(*display)["interactive"].value_or(false);
            v.texture_max_size=(*display)["texture_max_size"].value_or(512);
            v.max_fps=(*display)["max_fps"].value_or(15.0);
            v.projection=(*display)["projection"].value_or(std::string("signal"));
            v.hold_open=(*display)["hold_open"].value_or(true);
            v.offscreen=(*display)["offscreen"].value_or(false);
            v.snapshot_directory=resolve(config_directory,(*display)["snapshot_directory"].value_or(std::string{}));
            if(v.texture_max_size<1||v.texture_max_size>8192||!std::isfinite(v.max_fps)||v.max_fps<=0||v.max_fps>120||
                (v.projection!="signal"&&v.projection!="primary"&&v.projection!="log"))
                throw std::runtime_error("invalid visualization options");
            if(v.enabled&&!c.runsProjection())throw std::runtime_error("visualization requires a projection workflow");
            if(v.interactive&&(!v.enabled||c.workflow.mode!=WorkflowMode::Project||v.offscreen||!v.hold_open))
                throw std::runtime_error("interactive visualization requires enabled=true, project workflow, offscreen=false and hold_open=true");
        }
        if (c.runsProjection() && !p)
            throw std::runtime_error("投影工作流需要 [projection]");
        auto* g = t["geometry"].as_table();
        if (!g) throw std::runtime_error("缺少 [geometry] 配置表");
        c.geometry.kind = geometry((*g)["kind"].value_or<std::string>("flat_cbct"));
        if (p) {
            if (p->contains("detector_effects"))
                throw std::runtime_error("projection.detector_effects has been removed; use projection.detector_response");
            for (const char* removed : { "path_cache_file", "use_library_fp", "use_cuda_spectral" })
                if (p->contains(removed))
                    throw std::runtime_error(std::string("projection.") + removed + " 已删除");
            c.projection.label_volume = resolve(config_directory, (*p)["label_volume"].value_or(std::string{}));
            c.projection.spectrum_file = resolve(config_directory, (*p)["spectrum_file"].value_or(std::string{}));
            if (auto* spectrum = (*p)["spectrum"].as_table()) {
                for (const char* key : {"source", "model", "export_file"})
                    if (spectrum->contains(key) && !(*spectrum)[key].is_string())
                        throw std::runtime_error(std::string("spectrum.") + key + " must be a string");
                for (const char* key : {"kvp", "anode_angle_deg", "bin_width_keV"})
                    if (spectrum->contains(key) && !(*spectrum)[key].is_number())
                        throw std::runtime_error(std::string("spectrum.") + key + " must be numeric");
                auto& sp = c.projection.spectrum;
                sp.source = (*spectrum)["source"].value_or(std::string("file"));
                sp.model = (*spectrum)["model"].value_or(std::string("spekpy-v1"));
                sp.kvp = (*spectrum)["kvp"].value_or(120.0);
                sp.anode_angle_deg = (*spectrum)["anode_angle_deg"].value_or(12.0);
                sp.bin_width_keV = (*spectrum)["bin_width_keV"].value_or(1.0);
                sp.export_file = resolve(config_directory, (*spectrum)["export_file"].value_or(std::string{}));
                if (auto* filters = (*spectrum)["filters"].as_array()) {
                    for (const auto& node : *filters) {
                        const auto* f = node.as_table();
                        if (!f) throw std::runtime_error("spectrum.filters must contain tables");
                        if ((f->contains("material") && !(*f)["material"].is_string()) ||
                            (f->contains("atomic_number") && !(*f)["atomic_number"].is_integer()))
                            throw std::runtime_error("invalid spectrum filter material or atomic number type");
                        for (const char* key : {"thickness_mm", "density_g_cm3"})
                            if (f->contains(key) && !(*f)[key].is_number())
                                throw std::runtime_error(std::string("spectrum filter ") + key + " must be numeric");
                        SpectrumFilterConfig filter;
                        filter.material = (*f)["material"].value_or(std::string{});
                        filter.thickness_mm = (*f)["thickness_mm"].value_or(0.0);
                        filter.atomic_number = (*f)["atomic_number"].value_or(0);
                        filter.density_g_cm3 = (*f)["density_g_cm3"].value_or(0.0);
                        sp.filters.push_back(filter);
                    }
                } else if (spectrum->contains("filters")) {
                    throw std::runtime_error("spectrum.filters must be an array of tables");
                }
                validateSpectrumSource(sp);
                if (sp.source == "generated" && !c.projection.spectrum_file.empty())
                    throw std::runtime_error("generated spectrum conflicts with projection.spectrum_file");
            }
            c.projection.output_file = resolve(config_directory, (*p)["output_file"].value_or(std::string("projection.raw")));
            c.projection.energy_output_file = resolve(config_directory, (*p)["energy_output_file"].value_or(std::string{}));
            c.projection.air_output_file = resolve(config_directory, (*p)["air_output_file"].value_or(std::string{}));
            c.projection.mAs_per_view = (*p)["mAs_per_view"].value_or(1.0);
            for (const char* removed : { "apply_geometry_flux", "quantum_noise_enabled", "quantum_noise_seed" })
                if (p->contains(removed))
                    throw std::runtime_error(std::string("projection.") + removed + " has been moved to a dedicated section");
            auto* flux = (*p)["geometry_flux"].as_table();
            if (!flux) throw std::runtime_error("missing [projection.geometry_flux]");
            c.projection.geometry_flux.enabled = (*flux)["enabled"].value_or(true);
            auto* noise = (*p)["quantum_noise"].as_table();
            if (!noise) throw std::runtime_error("missing [projection.quantum_noise]");
            c.projection.quantum_noise.enabled = (*noise)["enabled"].value_or(true);
            c.projection.quantum_noise.seed = (*noise)["seed"].value_or(67890u);
            if (!std::isfinite(c.projection.mAs_per_view) || c.projection.mAs_per_view <= 0.0)
                throw std::runtime_error("projection.mAs_per_view must be positive");
            c.projection.save_metadata = (*p)["save_metadata"].value_or(true);
            if (auto* scatter = (*p)["scatter"].as_table()) {
                c.projection.scatter.enabled = (*scatter)["enabled"].value_or(false);
                c.projection.scatter.model_file = resolve(config_directory,
                    (*scatter)["model_file"].value_or(std::string{}));
                bool kernel_segments_explicit = false;
                if (auto* kernel = (*scatter)["kernel"].as_table()) {
                    c.projection.scatter.kernel.mode = (*kernel)["mode"].value_or(std::string("all"));
                    c.projection.scatter.kernel.segments = (*kernel)["segments"].value_or(3);
                    kernel_segments_explicit = kernel->contains("segments");
                }
                if (c.projection.scatter.kernel.mode != "all" &&
                    c.projection.scatter.kernel.mode != "segmented")
                    throw std::runtime_error("projection.scatter.kernel.mode must be all or segmented");
                if (c.projection.scatter.kernel.mode == "segmented" &&
                    c.projection.scatter.kernel.segments < 2)
                    throw std::runtime_error("projection.scatter.kernel.segments must be at least 2");
                if (c.projection.scatter.kernel.mode == "all" && kernel_segments_explicit)
                    throw std::runtime_error(
                        "projection.scatter.kernel.segments is only valid for segmented mode");
                if (c.projection.scatter.enabled) {
                    if (!available(c.projection.scatter.model_file))
                        throw std::runtime_error("scatter.model_file must point to an existing kernel model");
                    if (c.geometry.kind == GeometryKind::CylCbct || c.geometry.kind == GeometryKind::CylHelical)
                        throw std::runtime_error("kernel scatter currently supports flat detectors only");
                }
            }
        }
        auto* reconstruction_table = t["reconstruction"].as_table();
        if (c.runsReconstruction() && !reconstruction_table)
            throw std::runtime_error("重建工作流需要 [reconstruction]");
        if (auto* r = reconstruction_table) {
            if (r->contains("enabled"))
                throw std::runtime_error("reconstruction.enabled 已删除，请使用 workflow.mode");
            for (const char* key : { "type", "output_volume_file", "slice_prefix" })
                if (r->contains(key) && !(*r)[key].is_string())
                    throw std::runtime_error(std::string("reconstruction.") + key + " 必须为字符串");
            c.reconstruction.type = (*r)["type"].value_or(std::string("analytic"));
            if (r->contains("save_slices") && !(*r)["save_slices"].is_boolean())
                throw std::runtime_error("reconstruction.save_slices 必须为布尔值");
            c.reconstruction.save_slices = (*r)["save_slices"].value_or(false);
            if (auto value = (*r)["input_projection_file"].value<std::string>())
                c.reconstruction.input_projection_file = resolve(config_directory, *value);
            if (auto* preprocessing = (*r)["preprocessing"].as_table()) {
                auto& pp = c.reconstruction.preprocessing;
                pp.enabled = (*preprocessing)["enabled"].value_or(false);
                if (auto* hardening = (*preprocessing)["beam_hardening"].as_table()) {
                    auto& bh = pp.beam_hardening;
                    bh.enabled = (*hardening)["enabled"].value_or(false);
                    bh.fit_from_water_plate = (*hardening)["fit_from_water_plate"].value_or(false);
                    if (auto value = (*hardening)["spectrum_file"].value<std::string>())
                        bh.spectrum_file = resolve(config_directory, *value);
                    bh.order = (*hardening)["order"].value_or(1);
                    bh.clamp_min = (*hardening)["clamp_min"].value_or(0.0);
                    bh.clamp_max = (*hardening)["clamp_max"].value_or(20.0);
                    bh.water_plate_max_mm = (*hardening)["water_plate_max_mm"].value_or(400.0);
                    bh.water_plate_samples = (*hardening)["water_plate_samples"].value_or(81);
                    bh.reference_energy_keV = (*hardening)["reference_energy_keV"].value_or(0.0);
                    if (auto* values = (*hardening)["coefficients"].as_array()) {
                        bh.coefficients.clear();
                        for (const auto& item : *values) {
                            const auto value = item.value<double>();
                            if (!value || !std::isfinite(*value))
                                throw std::runtime_error("reconstruction.preprocessing.beam_hardening.coefficients must contain finite numbers");
                            bh.coefficients.push_back(*value);
                        }
                    }
                }
                if (auto* scatter = (*preprocessing)["scatter"].as_table()) {
                    auto& sc = pp.scatter;
                    sc.enabled = (*scatter)["enabled"].value_or(false);
                    if (auto value = (*scatter)["air_projection_file"].value<std::string>())
                        sc.air_projection_file = resolve(config_directory, *value);
                    if (auto value = (*scatter)["model_file"].value<std::string>())
                        sc.model_file = resolve(config_directory, *value);
                    for (const char* key : { "iterations", "damping", "scatter_scale", "object_mask_threshold" })
                        if (scatter->contains(key))
                            throw std::runtime_error(std::string("reconstruction.preprocessing.scatter.") + key +
                                " has moved to reconstruction.preprocessing.scatter.correction");
                    if (auto* correction = (*scatter)["correction"].as_table()) {
                        sc.correction.iterations = (*correction)["iterations"].value_or(3);
                        sc.correction.damping = (*correction)["damping"].value_or(0.7);
                        sc.correction.scatter_scale = (*correction)["scatter_scale"].value_or(1.0);
                        sc.correction.object_mask_threshold =
                            (*correction)["object_mask_threshold"].value_or(1.0e-6);
                    }
                    if (auto* guide = (*scatter)["guide"].as_table()) {
                        sc.guide.enabled = (*guide)["enabled"].value_or(false);
                        sc.guide.sigma_u_pixels = (*guide)["sigma_u_pixels"].value_or(2.0);
                        sc.guide.sigma_v_pixels = (*guide)["sigma_v_pixels"].value_or(2.0);
                    }
                    if (auto* solver = (*scatter)["solver"].as_table()) {
                        sc.solver.type = (*solver)["type"].value_or(std::string("fixed"));
                        if (auto* adaptive = (*solver)["adaptive"].as_table()) {
                            auto& a = sc.solver.adaptive;
                            if (adaptive->contains("relative_tolerance") || adaptive->contains("max_iterations"))
                                throw std::runtime_error("adaptive scatter solver uses iterations; tolerance stopping has been removed");
                            a.iterations = (*adaptive)["iterations"].value_or(10);
                            a.initial_damping = (*adaptive)["initial_damping"].value_or(0.7);
                            a.min_damping = (*adaptive)["min_damping"].value_or(0.1);
                            a.damping_decay = (*adaptive)["damping_decay"].value_or(1.0);
                            if (adaptive->contains("guide"))
                                throw std::runtime_error("use reconstruction.preprocessing.scatter.guide for either solver");
                            if (auto* guard = (*adaptive)["transmission_guard"].as_table()) {
                                a.transmission_guard.low_quantile = (*guard)["low_quantile"].value_or(0.01);
                                a.transmission_guard.risk_threshold = (*guard)["risk_threshold"].value_or(0.02);
                                a.transmission_guard.positivity_margin = (*guard)["positivity_margin"].value_or(0.9);
                            }
                            if (sc.solver.type != "adaptive_experimental")
                                throw std::runtime_error("scatter.solver.adaptive requires adaptive_experimental");
                        }
                    }
                    if (sc.solver.type != "fixed" && sc.solver.type != "adaptive_experimental")
                        throw std::runtime_error("scatter.solver.type must be fixed or adaptive_experimental");
                    const auto& adaptive = sc.solver.adaptive;
                    if (!std::isfinite(adaptive.damping_decay) || adaptive.damping_decay <= 0 || adaptive.damping_decay > 1)
                        throw std::runtime_error("adaptive damping_decay must be in (0,1]");
                    if (sc.guide.enabled &&
                        (!std::isfinite(sc.guide.sigma_u_pixels) || sc.guide.sigma_u_pixels <= 0 || sc.guide.sigma_u_pixels > 32 ||
                         !std::isfinite(sc.guide.sigma_v_pixels) || sc.guide.sigma_v_pixels <= 0 || sc.guide.sigma_v_pixels > 32))
                        throw std::runtime_error("scatter guide sigma must be in (0,32] pixels");
                    if (adaptive.iterations <= 0 ||
                        !std::isfinite(adaptive.initial_damping) || adaptive.initial_damping <= 0.0 ||
                        adaptive.initial_damping > 1.0 || !std::isfinite(adaptive.min_damping) ||
                        adaptive.min_damping <= 0.0 || adaptive.min_damping > adaptive.initial_damping)
                        throw std::runtime_error("invalid experimental adaptive scatter solver parameters");
                    const auto& guard = adaptive.transmission_guard;
                    if (!std::isfinite(guard.low_quantile) || guard.low_quantile <= 0 || guard.low_quantile > 1 ||
                        !std::isfinite(guard.risk_threshold) || guard.risk_threshold <= 0 || guard.risk_threshold >= 1 ||
                        !std::isfinite(guard.positivity_margin) || guard.positivity_margin <= 0 || guard.positivity_margin >= 1)
                        throw std::runtime_error("invalid adaptive scatter transmission guard");
                    if (sc.solver.type == "adaptive_experimental") {
                        const auto* correction = (*scatter)["correction"].as_table();
                        if (correction && (correction->contains("iterations") || correction->contains("damping")))
                            throw std::runtime_error("adaptive scatter solver uses solver.adaptive iteration and damping parameters");
                    }
                    if (auto* kernel = (*scatter)["kernel"].as_table()) {
                        sc.kernel.mode = (*kernel)["mode"].value_or(std::string("all"));
                        sc.kernel.segments = (*kernel)["segments"].value_or(3);
                    }
                    if (sc.kernel.mode != "all" && sc.kernel.mode != "segmented")
                        throw std::runtime_error(
                            "reconstruction.preprocessing.scatter.kernel.mode must be all or segmented");
                    if (sc.kernel.mode == "segmented" && sc.kernel.segments < 2)
                        throw std::runtime_error(
                            "reconstruction.preprocessing.scatter.kernel.segments must be at least 2");
                    if (sc.kernel.mode == "all") {
                        const auto* kernel = (*scatter)["kernel"].as_table();
                        if (kernel && kernel->contains("segments"))
                            throw std::runtime_error(
                                "reconstruction.preprocessing.scatter.kernel.segments is only valid for segmented mode");
                    }
                }
            }
            auto* a = (*r)["analytic"].as_table();
            auto* i = (*r)["iterative"].as_table();
            if (c.reconstruction.type != "analytic" && c.reconstruction.type != "iterative" && c.reconstruction.type != "statistical")
                throw std::runtime_error("reconstruction.type 仅支持 analytic/iterative/statistical");
            if (c.reconstruction.type == "analytic" && !a)
                throw std::runtime_error("analytic 重建需要 [reconstruction.analytic]");
            if (c.reconstruction.type == "iterative" && !i)
                throw std::runtime_error("迭代重建需要 [reconstruction.iterative]");
            if(c.reconstruction.type=="statistical"){
                const auto* s=(*r)["statistical"].as_table();
                if(!s)throw std::runtime_error("Statistical reconstruction requires [reconstruction.statistical]");
                if(c.geometry.kind!=GeometryKind::FlatCbct&&c.geometry.kind!=GeometryKind::FlatHelical)
                    throw std::runtime_error("Statistical reconstruction currently requires flat circular/helical geometry");
                auto& cfg=c.reconstruction.statistical;
                if ((*s).contains("projection_model"))
                    throw std::runtime_error("statistical.projection_model was removed; use forward_projector and back_projector");
                cfg.forward_projector=(*s)["forward_projector"].value_or(std::string("auto"));
                cfg.back_projector=(*s)["back_projector"].value_or(std::string("auto"));
                cfg.non_negative=(*s)["non_negative"].value_or(true);
                if(cfg.forward_projector!="auto"&&cfg.forward_projector!="joseph"&&cfg.forward_projector!="siddon"&&cfg.forward_projector!="footprint")
                    throw std::runtime_error("Statistical forward_projector supports auto/joseph/siddon/footprint");
                if(cfg.back_projector!="auto"&&cfg.back_projector!="fdk_pseudo_matched"&&cfg.back_projector!="siddon_ray"&&cfg.back_projector!="joseph"&&cfg.back_projector!="joseph_v3"&&cfg.back_projector!="footprint")
                    throw std::runtime_error("Statistical back_projector is unsupported");
                if(auto* d=(*s)["data_fidelity"].as_table()){
                    cfg.data_fidelity.type=(*d)["type"].value_or(std::string("weighted_least_squares"));
                    cfg.data_fidelity.scale=(*d)["scale"].value_or(1.0);
                    cfg.data_fidelity.normalize_by_ray_count=(*d)["normalize_by_ray_count"].value_or(false);
                    cfg.data_fidelity.require_verified_adjoint=(*d)["require_verified_adjoint"].value_or(false);
                    if(auto file=(*d)["weights_file"].value<std::string>())if(!file->empty())cfg.data_fidelity.weights_file=resolve(config_directory,*file);
                }
                if(auto* reg=(*s)["regularization"].as_table()){
                    cfg.regularization.plugin=(*reg)["plugin"].value_or(std::string("huber_tv"));
                    cfg.regularization.strength=(*reg)["strength"].value_or(0.0);
                    cfg.regularization.shape=(*reg)["shape"].value_or(0.001);
                    cfg.regularization.dimensions=(*reg)["dimensions"].value_or(3);
                }
                if(auto* opt=(*s)["optimizer"].as_table()){
                    cfg.optimizer.type=(*opt)["type"].value_or(std::string("regularized_algebraic"));
                    cfg.optimizer.iterations=(*opt)["iterations"].value_or(10);
                    cfg.optimizer.step_size=(*opt)["step_size"].value_or(0.0);
                    cfg.optimizer.backtracking=(*opt)["backtracking"].value_or(true);
                    cfg.optimizer.max_backtracking=(*opt)["max_backtracking"].value_or(20);
                    cfg.optimizer.step_shrink=(*opt)["step_shrink"].value_or(0.5);
                    cfg.optimizer.relative_tolerance=(*opt)["relative_tolerance"].value_or(0.0);
                    cfg.optimizer.minimum_iterations=(*opt)["minimum_iterations"].value_or(1);
                }
                // Experimental composition tables in older files are ignored.
                // Integration strategy and inner solver settings are internal policy.
                const auto& d=cfg.data_fidelity;const auto& reg=cfg.regularization;const auto& opt=cfg.optimizer;
                if(d.type!="weighted_least_squares"||d.require_verified_adjoint)throw std::runtime_error("Statistical data fidelity unavailable or adjoint not verified");
                if(opt.type!="regularized_algebraic"&&opt.type!="gradient_descent"&&opt.type!="algebraic_gradient"&&opt.type!="raw_gradient_descent")throw std::runtime_error("Statistical optimizer not implemented: FISTA is reserved");
                if(reg.plugin!="none"&&reg.plugin!="smoothed_tv"&&reg.plugin!="huber_tv"&&reg.plugin!="huber_tv_prox"&&reg.plugin!="tikhonov"&&!(reg.plugin=="tv"&&(opt.type=="regularized_algebraic"||opt.type=="algebraic_gradient")))
                    throw std::runtime_error("Gradient Descent requires a differentiable regularizer; standard TV requires Prox");
                if(!std::isfinite(d.scale)||d.scale<=0||!std::isfinite(reg.strength)||reg.strength<0||
                    ((reg.plugin!="none"&&reg.plugin!="tikhonov"&&!(reg.plugin=="tv"&&(opt.type=="regularized_algebraic"||opt.type=="algebraic_gradient")))&&(!std::isfinite(reg.shape)||reg.shape<=0))||(reg.dimensions!=2&&reg.dimensions!=3)||
                    (reg.plugin=="none"&&reg.strength>0)||opt.iterations<=0||!std::isfinite(opt.step_size)||opt.step_size<0||
                    opt.max_backtracking<0||!std::isfinite(opt.step_shrink)||opt.step_shrink<=0||opt.step_shrink>=1||
                    !std::isfinite(opt.relative_tolerance)||opt.relative_tolerance<0||opt.minimum_iterations<=0)
                    throw std::runtime_error("Invalid statistical reconstruction parameters");
                if(check_runtime_files&&!d.weights_file.empty()){
                    const auto& g=c.geometry.parameters;const uintmax_t expected=uintmax_t(g.views)*g.detector_u*g.detector_v*sizeof(float);
                    if(!std::filesystem::exists(d.weights_file)||std::filesystem::file_size(d.weights_file)!=expected)
                        throw std::runtime_error("Statistical weights_file must be float32 with the projection shape");
                }
            }
            if (a) {
                for (const char* key : { "pipeline", "filter" })
                    if (a->contains(key) && !(*a)[key].is_string())
                        throw std::runtime_error(std::string("reconstruction.analytic.") + key + " 必须为字符串");
                for (const char* key : { "cutoff", "gain", "butterworth_order", "kaiser_beta", "tukey_alpha", "chunk_views" })
                    if (a->contains(key) && !(*a)[key].is_number())
                        throw std::runtime_error(std::string("reconstruction.analytic.") + key + " 必须为数值");
                auto& rc = c.reconstruction.analytic;
                rc.pipeline = (*a)["pipeline"].value_or(std::string("fdk"));
                rc.filter = (*a)["filter"].value_or(std::string(rc.pipeline == "wfbp" ? "ramlak" : "shepp-logan"));
                if(a->contains("wfbp_cutoff") || a->contains("wfbp_apodization"))
                    throw std::runtime_error("Use shared analytic.filter/cutoff; wfbp coefficient settings were removed");
                rc.cutoff=(*a)["cutoff"].value_or(.5); rc.gain=(*a)["gain"].value_or(1.);
                rc.butterworth_order=(*a)["butterworth_order"].value_or(2.);
                rc.kaiser_beta=(*a)["kaiser_beta"].value_or(8.6);
                rc.tukey_alpha=(*a)["tukey_alpha"].value_or(.5);
                rc.chunk_views = (*a)["chunk_views"].value_or(32);
                if (rc.pipeline != "fdk" && rc.pipeline != "wfbp")
                    throw std::runtime_error("analytic.pipeline 仅支持 fdk/wfbp");
                if (rc.filter != "ramlak" && rc.filter != "ram-lak" &&
                    rc.filter != "shepp-logan" && rc.filter != "cosine" &&
                    rc.filter != "hann" && rc.filter != "hamming" && rc.filter != "blackman" &&
                    rc.filter != "butterworth" && rc.filter != "kaiser" && rc.filter != "tukey")
                    throw std::runtime_error("未知 analytic.filter: " + rc.filter);
                if (rc.chunk_views<=0 || !std::isfinite(rc.cutoff) || rc.cutoff<=0 || rc.cutoff>.5 ||
                    !std::isfinite(rc.gain) || rc.gain<=0 || !std::isfinite(rc.butterworth_order) || rc.butterworth_order<=0 ||
                    !std::isfinite(rc.kaiser_beta) || rc.kaiser_beta<0 || !std::isfinite(rc.tukey_alpha) || rc.tukey_alpha<0 || rc.tukey_alpha>1)
                    throw std::runtime_error("Invalid shared analytic filter settings");
            }
            if (i) {
                auto& rc = c.reconstruction.iterative;
                rc.algorithm = (*i)["algorithm"].value_or(std::string("os_sart_tv"));
                rc.iterations = (*i)["iterations"].value_or(10);
                rc.relaxation = (*i)["relaxation"].value_or(1.0);
                rc.subsets = (*i)["subsets"].value_or(1);
                if ((*i).contains("projection_model"))
                    throw std::runtime_error("iterative.projection_model was removed; use forward_projector and back_projector");
                rc.forward_projector = (*i)["forward_projector"].value_or(std::string("auto"));
                rc.back_projector = (*i)["back_projector"].value_or(std::string("auto"));
                if (rc.forward_projector != "auto" && rc.forward_projector != "joseph" &&
                    rc.forward_projector != "siddon" && rc.forward_projector != "footprint")
                    throw std::runtime_error("forward_projector supports auto/joseph/siddon/footprint");
                if (rc.back_projector != "auto" && rc.back_projector != "fdk_pseudo_matched" &&
                    rc.back_projector != "siddon_ray" && rc.back_projector != "joseph" &&
                    rc.back_projector != "joseph_v3" && rc.back_projector != "footprint")
                    throw std::runtime_error("back_projector is unsupported");
                rc.tv_iterations = (*i)["tv_iterations"].value_or(20);
                rc.tv_alpha = (*i)["tv_alpha"].value_or(0.002);
                rc.tv_alpha_reduction = (*i)["tv_alpha_reduction"].value_or(0.95);
                rc.maximum_update_ratio = (*i)["maximum_update_ratio"].value_or(0.95);
                rc.non_negative = (*i)["non_negative"].value_or(true);
                if (auto* t=(*i)["regularization"].as_table()) {
                    auto& tv=rc.regularization;
                    tv.plugin=(*t)["plugin"].value_or(std::string("none"));
                    tv.strength=(*t)["strength"].value_or(0.0);
                    tv.huber_delta=(*t)["huber_delta"].value_or(0.001);
                    tv.step_size=(*t)["step_size"].value_or(0.0);
                    tv.inner_iterations=(*t)["inner_iterations"].value_or(5);
                    tv.dimensions=(*t)["dimensions"].value_or(3);
                    tv.epsilon=(*t)["epsilon"].value_or(1e-6);
                    tv.strength_reduction=(*t)["strength_reduction"].value_or(1.0);
                    if (tv.plugin!="none" && tv.plugin!="huber_tv_prox" && tv.plugin!="smoothed_tv")
                        throw std::runtime_error("Unknown regularization plugin");
                    if (tv.plugin!="none") {
                        if (rc.algorithm!="sart" && rc.algorithm!="sirt" && rc.algorithm!="ossart")
                            throw std::runtime_error("TV plugins require internal sart/sirt/ossart");
                        if (c.geometry.kind==GeometryKind::CylCbct || c.geometry.kind==GeometryKind::CylHelical)
                            throw std::runtime_error("TV plugins require flat detector");
                        if (!std::isfinite(tv.strength)||tv.strength<0||tv.inner_iterations<=0||
                            (tv.dimensions!=2&&tv.dimensions!=3)||!std::isfinite(tv.huber_delta)||tv.huber_delta<=0||
                            !std::isfinite(tv.step_size)||tv.step_size<0||!std::isfinite(tv.epsilon)||tv.epsilon<=0||
                            !std::isfinite(tv.strength_reduction)||tv.strength_reduction<=0)
                            throw std::runtime_error("Invalid TV plugin parameters");
                    }
                }
                if (rc.algorithm != "tigre_sart" && rc.algorithm != "tigre_sirt" &&
                    rc.algorithm != "tigre_os_sart" && rc.algorithm != "tigre_sart_tv" &&
                    rc.algorithm != "tigre_os_sart_tv" && rc.algorithm != "sart" &&
                    rc.algorithm != "sirt" &&
                    rc.algorithm != "ossart" && rc.algorithm != "cgls")
                    throw std::runtime_error("iterative.algorithm 不支持该值");
                const bool uses_tv = rc.algorithm == "tigre_sart_tv" ||
                    rc.algorithm == "tigre_os_sart_tv";
                if (uses_tv && rc.tv_alpha <= 0)
                    throw std::runtime_error("TIGRE TV algorithm requires tv_alpha > 0");
                const bool uses_subsets = rc.algorithm == "ossart" ||
                    rc.algorithm == "tigre_os_sart" ||
                    rc.algorithm == "tigre_os_sart_tv";
                if (!uses_subsets && rc.subsets != 1)
                    throw std::runtime_error("该 iterative.algorithm 不使用 subsets，配置值必须为 1");
                if (rc.iterations <= 0 || !std::isfinite(rc.relaxation) || rc.relaxation <= 0 || rc.subsets <= 0 ||
                    (uses_tv && (rc.tv_iterations <= 0 || !std::isfinite(rc.tv_alpha) || rc.tv_alpha < 0 ||
                    !std::isfinite(rc.tv_alpha_reduction) || rc.tv_alpha_reduction <= 0)) ||
                    !std::isfinite(rc.maximum_update_ratio) || rc.maximum_update_ratio <= 0)
                    throw std::runtime_error("iterative 参数必须为正数");
            }
            auto& pp = c.reconstruction.preprocessing;
            if (pp.enabled) {
                if (!pp.beam_hardening.enabled && !pp.scatter.enabled)
                    throw std::runtime_error("reconstruction.preprocessing must enable beam_hardening or scatter");
                const auto& bh = pp.beam_hardening;
                if (bh.enabled && (bh.order < 1 || bh.order > 16 ||
                    (!bh.fit_from_water_plate &&
                     bh.coefficients.size() != static_cast<std::size_t>(bh.order + 1)) ||
                    !std::isfinite(bh.clamp_min) || !std::isfinite(bh.clamp_max) ||
                    bh.clamp_min < 0.0 || bh.clamp_max <= bh.clamp_min))
                    throw std::runtime_error("invalid beam hardening polynomial configuration");
                if (bh.enabled && bh.fit_from_water_plate &&
                    ((!(bh.spectrum_file.empty() && c.projection.spectrum.source == "generated") &&
                     !available(bh.spectrum_file.empty()
                        ? c.projection.spectrum_file : bh.spectrum_file)) ||
                     !std::isfinite(bh.water_plate_max_mm) || bh.water_plate_max_mm <= 0.0 ||
                     bh.water_plate_samples < bh.order + 2 ||
                     !std::isfinite(bh.reference_energy_keV) || bh.reference_energy_keV < 0.0))
                    throw std::runtime_error("invalid water-plate beam hardening calibration configuration");
                auto& sc = pp.scatter;
                // GUI setup can carry defaults from the forward model, but execution remains strict.
                if(!check_runtime_files && sc.enabled){
                    if(sc.air_projection_file.empty())sc.air_projection_file=c.projection.air_output_file;
                    if(sc.model_file.empty())sc.model_file=c.projection.scatter.model_file;
                }
                if(sc.enabled){
                    if(sc.air_projection_file.empty())throw std::runtime_error("scatter correction requires an air projection file (set Correction air or projection.air_output_file)");
                    if(sc.model_file.empty())throw std::runtime_error("scatter correction requires a model file (set Correction model)");
                    const bool planned_air=c.runsProjection() && sc.air_projection_file.lexically_normal()==c.projection.air_output_file.lexically_normal();
                    if(!available(sc.air_projection_file) && !planned_air)throw std::runtime_error("scatter correction air projection file does not exist: "+sc.air_projection_file.string());
                    if(!available(sc.model_file))throw std::runtime_error("scatter correction model file does not exist: "+sc.model_file.string());
                    if(sc.correction.iterations<=0)throw std::runtime_error("scatter correction iterations must be positive");
                    if(!std::isfinite(sc.correction.damping)||sc.correction.damping<=0||sc.correction.damping>1)throw std::runtime_error("scatter correction damping must be in (0,1]");
                    if(!std::isfinite(sc.correction.scatter_scale)||sc.correction.scatter_scale<0)throw std::runtime_error("scatter correction scale must be finite and nonnegative");
                    if(sc.kernel.mode!="all" && sc.kernel.mode!="segmented")throw std::runtime_error("scatter correction kernel mode must be all or segmented");
                    if(sc.kernel.mode=="segmented" && sc.kernel.segments<2)throw std::runtime_error("scatter correction kernel segments must be at least 2");
                    if(!std::isfinite(sc.correction.object_mask_threshold)||sc.correction.object_mask_threshold<0)throw std::runtime_error("scatter correction object mask threshold must be finite and nonnegative");
                }
            }
            const auto output_tag = c.reconstruction.type == "iterative"
                ? c.reconstruction.iterative.algorithm : c.reconstruction.type=="statistical" ? std::string("statistical-gd") : c.reconstruction.analytic.pipeline;
            const auto& source_file = c.reconstruction.input_projection_file.empty()
                ? c.projection.output_file : c.reconstruction.input_projection_file;
            const auto default_volume = source_file.parent_path() /
                (source_file.stem().string() + "-" + output_tag + "-volume.raw");
            const auto default_prefix = source_file.parent_path() /
                (source_file.stem().string() + "-" + output_tag);
            if (auto value = (*r)["output_volume_file"].value<std::string>())
                c.reconstruction.output_volume_file = resolve(config_directory, *value);
            else
                c.reconstruction.output_volume_file = default_volume;
            if (auto value = (*r)["slice_prefix"].value<std::string>())
                c.reconstruction.slice_prefix = resolve(config_directory, *value);
            else
                c.reconstruction.slice_prefix = default_prefix;
        }
        if (auto* g = t["geometry"].as_table()) {
            auto integer = [&](const char* k, int d) { return (*g)[k].value_or(d); };
            auto real = [&](const char* k, double d) { return (*g)[k].value_or(d); };
            c.geometry.parameters.views = integer("views", 360); c.geometry.parameters.views_per_turn = integer("views_per_turn", c.geometry.parameters.views); c.geometry.parameters.rotation_direction = integer("rotation_direction", 1); c.geometry.parameters.detector_u = integer("detector_u", 256); c.geometry.parameters.detector_v = integer("detector_v", 128);
            c.geometry.parameters.pixel_u_mm = real("pixel_u_mm", 1.0); c.geometry.parameters.pixel_v_mm = real("pixel_v_mm", 1.0); c.geometry.parameters.sid_mm = real("sid_mm", 500.0); c.geometry.parameters.sdd_mm = real("sdd_mm", 1000.0);
            c.geometry.parameters.offset_u_mm = real("offset_u_mm", 0.0); c.geometry.parameters.offset_n_mm = real("offset_n_mm", 0.0); c.geometry.parameters.offset_v_mm = real("offset_v_mm", 0.0);
            c.geometry.parameters.tilt_u_rad = real("tilt_u_rad", 0.0); c.geometry.parameters.tilt_v_rad = real("tilt_v_rad", 0.0); c.geometry.parameters.tilt_n_rad = real("tilt_n_rad", 0.0);
            c.geometry.parameters.source_offset_x_mm = real("source_offset_x_mm", 0.0); c.geometry.parameters.source_offset_y_mm = real("source_offset_y_mm", 0.0); c.geometry.parameters.source_offset_z_mm = real("source_offset_z_mm", 0.0); c.geometry.parameters.start_angle_rad = real("start_angle_rad", 0.0); c.geometry.parameters.pitch_mm_per_turn = real("pitch_mm_per_turn", 0.0); c.geometry.parameters.start_z_mm = real("start_z_mm", 0.0);
            c.geometry.parameters.scan_range_rad = real("scan_range_rad", 6.28318530717958647692);
            c.geometry.parameters.planar_detector_y_mm = real("planar_detector_y_mm", c.geometry.parameters.sdd_mm - c.geometry.parameters.sid_mm);
            c.geometry.parameters.source_orbit_radius_mm = real("source_orbit_radius_mm", 0.0);
            c.geometry.parameters.source_lateral_offset_mm = real("source_lateral_offset_mm", 0.0);
            c.geometry.parameters.reconstruction_detector_u = integer("reconstruction_detector_u", c.geometry.parameters.detector_u);
            c.geometry.parameters.reconstruction_detector_v = integer("reconstruction_detector_v", c.geometry.parameters.detector_v);
            c.geometry.parameters.planar_resample_mode = (*g)["planar_resample_mode"].value_or(std::string("perspective_orthogonal"));
            c.geometry.parameters.nikon_tilt_rad = real("nikon_tilt_rad", 0.0);
            c.geometry.parameters.nikon_tilt_axis_azimuth_rad = real("nikon_tilt_axis_azimuth_rad", 0.0);
            c.geometry.parameters.rotation_axis_pivot_x_mm = real("rotation_axis_pivot_x_mm", 0.0);
            c.geometry.parameters.rotation_axis_pivot_y_mm = real("rotation_axis_pivot_y_mm", 0.0);
            c.geometry.parameters.rotation_axis_pivot_z_mm = real("rotation_axis_pivot_z_mm", 0.0);
            c.geometry.parameters.voxel_x_mm = real("voxel_x_mm", 1.0); c.geometry.parameters.voxel_y_mm = real("voxel_y_mm", 1.0); c.geometry.parameters.voxel_z_mm = real("voxel_z_mm", 1.0); c.geometry.parameters.volume_x = integer("volume_x", 0); c.geometry.parameters.volume_y = integer("volume_y", 0); c.geometry.parameters.volume_z = integer("volume_z", 0);
            c.geometry.parameters.reconstruction_volume_x = integer("reconstruction_volume_x", c.geometry.parameters.volume_x); c.geometry.parameters.reconstruction_volume_y = integer("reconstruction_volume_y", c.geometry.parameters.volume_y); c.geometry.parameters.reconstruction_volume_z = integer("reconstruction_volume_z", c.geometry.parameters.volume_z); c.geometry.parameters.reconstruction_voxel_x_mm = real("reconstruction_voxel_x_mm", c.geometry.parameters.voxel_x_mm); c.geometry.parameters.reconstruction_voxel_y_mm = real("reconstruction_voxel_y_mm", c.geometry.parameters.voxel_y_mm); c.geometry.parameters.reconstruction_voxel_z_mm = real("reconstruction_voxel_z_mm", c.geometry.parameters.voxel_z_mm);
            c.geometry.parameters.phantom_offset_x_mm = real("phantom_offset_x_mm", 0.0); c.geometry.parameters.phantom_offset_y_mm = real("phantom_offset_y_mm", 0.0); c.geometry.parameters.phantom_offset_z_mm = real("phantom_offset_z_mm", 0.0);
            c.geometry.parameters.phantom_rotation_x_rad = real("phantom_rotation_x_rad", 0.0); c.geometry.parameters.phantom_rotation_y_rad = real("phantom_rotation_y_rad", 0.0); c.geometry.parameters.phantom_rotation_z_rad = real("phantom_rotation_z_rad", 0.0);
            if (!std::isfinite(c.geometry.parameters.phantom_rotation_x_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_y_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_z_rad)) throw std::runtime_error("geometry.phantom_rotation_*_rad 必须为有限数值");
            c.geometry.parameters.reconstruction_offset_x_mm = real("reconstruction_offset_x_mm", 0.0); c.geometry.parameters.reconstruction_offset_y_mm = real("reconstruction_offset_y_mm", 0.0); c.geometry.parameters.reconstruction_offset_z_mm = real("reconstruction_offset_z_mm", 0.0);
        }
        if (p) {
            {
                if(auto* batch=(*p)["batching"].as_table()) {
                    c.projection.batching.views=(*batch)["views"].value_or(64);
                    c.projection.batching.max_buffer_mib=(*batch)["max_buffer_mib"].value_or(256);
                }
                auto* s = (*p)["sampling"].as_table();
                if (!s) throw std::runtime_error("缺少 [projection.sampling]");
                for (const char* removed : { "photons_per_pixel", "photon_count_mode", "photon_seed" })
                    if (s->contains(removed))
                        throw std::runtime_error(std::string("projection sampling option has been removed: ") + removed);
                c.projection.sampling.samples_per_pixel = (*s)["samples_per_pixel"].value_or(1);
                c.projection.sampling.seed = (*s)["spatial_seed"].value_or(12345u);
            }
        }
        if (c.runsProjection() &&
            c.geometry.kind != GeometryKind::FlatCbct && c.geometry.kind != GeometryKind::FlatHelical &&
            c.geometry.kind != GeometryKind::CylCbct && c.geometry.kind != GeometryKind::CylHelical &&
            c.geometry.kind != GeometryKind::PlanarSourceOrbit &&
            c.geometry.kind != GeometryKind::NikonTilted)
            throw std::runtime_error("随机多能谱前投不支持该几何类型");
        if (c.runsProjection() &&
            c.projection.materials.size() > 32)
            throw std::runtime_error("随机多能谱前投当前最多支持32种材料");
        if (auto* d = p ? (*p)["detector_response"].as_table() : nullptr) {
            auto& response = c.projection.detector_response;
            response.enabled = (*d)["enabled"].value_or(false);
            response.oblique_path_correction = (*d)["oblique_path_correction"].value_or(true);
            if (d->contains("sensor") || d->contains("cover_layers"))
                throw std::runtime_error("detector response uses protective_layer, impurity_layer, and scintillator_layer");
            auto parse_layer = [&](const char* key, DetectorLayerConfig& value,
                                   bool required, bool default_enabled) {
                auto* layer = (*d)[key].as_table();
                if (!layer) {
                    value.enabled = false;
                    if (response.enabled && required)
                        throw std::runtime_error(std::string("missing projection.detector_response.") + key);
                    return;
                }
                value.enabled = (*layer)["enabled"].value_or(default_enabled);
                value.name = (*layer)["name"].value_or(std::string(key));
                value.preset = (*layer)["preset"].value_or(std::string{});
                value.thickness_mm = (*layer)["thickness_mm"].value_or(0.0);
                if (auto map = (*layer)["thickness_map_file"].value<std::string>())
                    value.thickness_map_file = *map;
                if (layer->contains("formula") || layer->contains("density_g_cm3"))
                    throw std::runtime_error("detector response layers use preset; formula and density_g_cm3 are not supported");
                if (!value.enabled) return;
                const auto* material = findBuiltinMaterial(value.preset);
                if (!material)
                    throw std::runtime_error(std::string("unknown detector layer preset in ") + key + ": " + value.preset);
                value.density_g_cm3 = material->density_g_cm3;
                if (value.thickness_mm <= 0 && value.thickness_map_file.empty())
                    throw std::runtime_error(std::string(key) + " requires positive thickness_mm or thickness_map_file");
            };
            parse_layer("protective_layer", response.protective_layer, true, true);
            parse_layer("impurity_layer", response.impurity_layer, false, false);
            parse_layer("scintillator_layer", response.scintillator_layer, true, true);
            for (auto* layer : {&response.protective_layer, &response.impurity_layer,
                                &response.scintillator_layer})
                if (!layer->thickness_map_file.empty())
                    layer->thickness_map_file = resolve(config_directory,
                        layer->thickness_map_file.string());
        }
        if (auto* post = p ? (*p)["detector_postprocess"].as_table() : nullptr) {
            auto parse_kernel = [](const toml::table& table, const char* key) {
                std::vector<double> values;
                auto* array = table[key].as_array();
                if (!array) return std::vector<double>{1.0};
                for (const auto& item : *array) {
                    auto value = item.value<double>();
                    if (!value || !std::isfinite(*value) || *value < 0)
                        throw std::runtime_error(std::string(key) + " must contain finite non-negative numbers");
                    values.push_back(*value);
                }
                if (values.empty() || values.size() % 2 == 0 || values.size() > 31)
                    throw std::runtime_error(std::string(key) + " must have an odd length between 1 and 31");
                double sum = 0.0;
                for (double value : values) sum += value;
                if (std::abs(sum - 1.0) > 1e-6)
                    throw std::runtime_error(std::string(key) + " weights must sum to 1");
                return values;
            };
            auto parse_crosstalk = [&](const char* key, CrosstalkConfig& value) {
                auto* table = (*post)[key].as_table();
                if (!table) return;
                value.enabled = (*table)["enabled"].value_or(false);
                value.kernel_u = parse_kernel(*table, "kernel_u");
                value.kernel_v = parse_kernel(*table, "kernel_v");
            };
            parse_crosstalk("optical_crosstalk",
                c.projection.detector_postprocess.optical_crosstalk);
            if (auto* table = (*post)["afterglow"].as_table()) {
                auto& value = c.projection.detector_postprocess.afterglow;
                value.enabled = (*table)["enabled"].value_or(false);
                value.p = (*table)["p"].value_or(0.0);
                value.initialization = (*table)["initialization"].value_or(std::string("zero"));
                if (table->contains("equilibrium_initialization"))
                    throw std::runtime_error("afterglow always starts from zero state; equilibrium_initialization is not supported");
                value.frame_time_ms = (*table)["frame_time_ms"].value_or(1.0);
                value.exposure_time_ms = (*table)["exposure_time_ms"].value_or(1.0);
                auto read_array = [&](const char* key, std::vector<double>& target) {
                    if (auto* array = (*table)[key].as_array())
                        for (const auto& item : *array) {
                            auto number = item.value<double>();
                            if (!number || !std::isfinite(*number))
                                throw std::runtime_error(std::string("afterglow.") + key + " must contain finite numbers");
                            target.push_back(*number);
                        }
                };
                read_array("weights", value.weights);
                read_array("time_constants_ms", value.time_constants_ms);
                if (value.initialization != "zero" && value.initialization != "steady_state")
                    throw std::runtime_error("afterglow.initialization must be zero or steady_state");
                if (value.enabled) {
                    if (value.p < 0 || value.p >= 1 || value.frame_time_ms <= 0 ||
                        value.exposure_time_ms <= 0 || value.exposure_time_ms > value.frame_time_ms || value.weights.empty() ||
                        value.weights.size() != value.time_constants_ms.size())
                        throw std::runtime_error("afterglow requires 0 < exposure_time_ms <= frame_time_ms and equally sized weights/time_constants_ms");
                    double sum = 0.0;
                    for (size_t i=0; i<value.weights.size(); ++i) {
                        if (value.weights[i] < 0 || value.time_constants_ms[i] <= 0)
                            throw std::runtime_error("afterglow weights must be non-negative and time constants positive");
                        sum += value.weights[i];
                    }
                    if (std::abs(sum - 1.0) > 1e-6)
                        throw std::runtime_error("afterglow weights must sum to 1");
                }
            }
            parse_crosstalk("electronic_crosstalk",
                c.projection.detector_postprocess.electronic_crosstalk);
            if (auto* table = (*post)["das"].as_table()) {
                auto& value = c.projection.detector_postprocess.das;
                value.enabled = (*table)["enabled"].value_or(false);
                value.gain_electrons_per_keV = (*table)["gain_electrons_per_keV"].value_or(1.0);
                value.offset_electrons = (*table)["offset_electrons"].value_or(0.0);
                value.electronic_noise_std_electrons = (*table)["electronic_noise_std_electrons"].value_or(0.0);
                value.saturation_electrons = (*table)["saturation_electrons"].value_or(0.0);
                value.adc_lsb_electrons = (*table)["adc_lsb_electrons"].value_or(0.0);
                value.noise_seed = (*table)["noise_seed"].value_or(24680u);
                if (value.enabled && (!std::isfinite(value.gain_electrons_per_keV) || value.gain_electrons_per_keV <= 0 ||
                    !std::isfinite(value.offset_electrons) || !std::isfinite(value.electronic_noise_std_electrons) || value.electronic_noise_std_electrons < 0 ||
                    !std::isfinite(value.saturation_electrons) || value.saturation_electrons < 0 ||
                    !std::isfinite(value.adc_lsb_electrons) || value.adc_lsb_electrons < 0))
                    throw std::runtime_error("invalid detector DAS parameters");
            }
        }
        if (auto* f = p ? (*p)["focal_spot"].as_table() : nullptr) {
            auto& spot = c.projection.focal_spot;
            spot.enabled = (*f)["enabled"].value_or(false);
            // Old TOML used enabled as the master switch for both effects.
            spot.defocus_enabled = (*f)["defocus_enabled"].value_or(spot.enabled);
            spot.normal_size_u_mm = (*f)["normal_size_u_mm"].value_or(0.0);
            spot.normal_size_v_mm = (*f)["normal_size_v_mm"].value_or(0.0);
            spot.defocus_size_u_mm = (*f)["defocus_size_u_mm"].value_or(0.0);
            spot.defocus_size_v_mm = (*f)["defocus_size_v_mm"].value_or(0.0);
            if (f->contains("size_u_mm") || f->contains("size_v_mm"))
                throw std::runtime_error("use normal_size_*_mm and defocus_size_*_mm for focal spot sizes");
            spot.epsilon = (*f)["epsilon"].value_or(0.0);
            if (f->contains("rotation_deg")) throw std::runtime_error("矩形焦点不支持 rotation_deg");
            if (!std::isfinite(spot.epsilon) || spot.epsilon < 0.0 || spot.epsilon > 1.0)
                throw std::runtime_error("projection.focal_spot.epsilon must be in [0, 1]");
            if ((spot.enabled || spot.defocus_enabled) &&
                ((!std::isfinite(spot.normal_size_u_mm) || spot.normal_size_u_mm < 0) ||
                 (!std::isfinite(spot.normal_size_v_mm) || spot.normal_size_v_mm < 0) ||
                 (!std::isfinite(spot.defocus_size_u_mm) || spot.defocus_size_u_mm < 0) ||
                 (!std::isfinite(spot.defocus_size_v_mm) || spot.defocus_size_v_mm < 0) ||
                 (spot.enabled && (!spot.defocus_enabled || spot.epsilon < 1.0) && (spot.normal_size_u_mm <= 0 || spot.normal_size_v_mm <= 0)) ||
                 (spot.defocus_enabled && spot.epsilon > 0.0 && (spot.defocus_size_u_mm <= 0 || spot.defocus_size_v_mm <= 0))))
                throw std::runtime_error("矩形焦点尺寸必须为正");
            if(spot.enabled&&spot.defocus_enabled&&spot.epsilon>0.0&&
                (spot.defocus_size_u_mm<spot.normal_size_u_mm||spot.defocus_size_v_mm<spot.normal_size_v_mm))
                throw std::runtime_error("散焦矩形必须包含正常焦点矩形，才能在外圈分层采样");
        }
        if(c.projection.batching.views<1||c.projection.batching.views>64||
            c.projection.batching.max_buffer_mib<1||c.projection.batching.max_buffer_mib>256)
            throw std::runtime_error("projection.batching: views must be 1..64, max_buffer_mib must be 1..256");
        if (c.projection.sampling.samples_per_pixel <= 0 || c.projection.sampling.samples_per_pixel > 4096)
            throw std::runtime_error("projection.sampling.samples_per_pixel must be in 1..4096");
        const bool planar = c.geometry.kind == GeometryKind::PlanarSourceOrbit;
        const bool tilted = c.geometry.kind == GeometryKind::NikonTilted;
        if ((planar || tilted) &&
            (!(c.geometry.parameters.scan_range_rad > 0.0) ||
             !std::isfinite(c.geometry.parameters.scan_range_rad)))
            throw std::runtime_error("non-standard geometry requires positive scan_range_rad");
        if (planar &&
            (!(c.geometry.parameters.planar_detector_y_mm > 0.0) ||
             c.geometry.parameters.reconstruction_detector_u <= 0 ||
             c.geometry.parameters.reconstruction_detector_v <= 0 ||
             (c.geometry.parameters.planar_resample_mode != "in_plane_rotated" &&
              c.geometry.parameters.planar_resample_mode != "perspective_orthogonal")))
            throw std::runtime_error("invalid planar_source_orbit geometry parameters");
        if ((planar || tilted) && c.projection.detector_response.enabled)
            throw std::runtime_error("non-standard trajectories do not yet support detector response");
        std::array<bool, 256> used_labels{};
        if (auto* a = p ? (*p)["materials"].as_array() : nullptr)for (auto& n : *a) { auto* m = n.as_table(); if (!m)throw std::runtime_error("projection.materials 必须是表数组"); MaterialSpec x; x.label = (std::uint8_t)(*m)["label"].value_or(0); x.name = (*m)["name"].value_or(std::string("material")); x.formula = (*m)["formula"].value_or(std::string{}); x.preset = (*m)["preset"].value_or(std::string{}); x.density_g_cm3 = (*m)["density_g_cm3"].value_or(1.0); if ((x.formula.empty() == x.preset.empty()) || x.density_g_cm3 <= 0)throw std::runtime_error("材料配置必须且只能填写 formula 或 preset，且密度必须为正"); if (used_labels[x.label])throw std::runtime_error("材料 label 不得重复: " + std::to_string(x.label)); used_labels[x.label] = true; if (!x.preset.empty()) { const auto* b = findBuiltinMaterial(x.preset); if (!b)throw std::runtime_error("未知内置材料 preset: " + x.preset); if (!m->contains("density_g_cm3"))x.density_g_cm3 = b->density_g_cm3; }c.projection.materials.push_back(std::move(x)); }
        if (c.runsProjection() && c.projection.materials.empty())
            throw std::runtime_error("投影工作流至少需要一个材料映射");
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            c.reconstruction.input_projection_file = c.projection.output_file;
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            throw std::runtime_error("重建工作流需要 reconstruction.input_projection_file");
        if (c.runsReconstruction()) {
            if ((c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "fdk" && c.geometry.kind != GeometryKind::FlatCbct && c.geometry.kind != GeometryKind::CylCbct && c.geometry.kind != GeometryKind::PlanarSourceOrbit && c.geometry.kind != GeometryKind::NikonTilted) ||
                (c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "wfbp" && c.geometry.kind != GeometryKind::CylHelical && c.geometry.kind != GeometryKind::FlatHelical))
                throw std::runtime_error("解析重建支持 flat_cbct/cyl_cbct+fdk、flat_helical/cyl_helical+wfbp");
        }
        return c;
    }
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path& p) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("无法读取能谱文件: " + p.string());
        std::vector<SpectrumPoint> result;
        std::string line;
        bool in_spectrum = false;
        bool explicit_widths = false;
        bool mcgpu_edges = false;
        while (std::getline(f, line)) {
            if (line.find("Format: YK-SPECTRUM 1.0") != std::string::npos) explicit_widths = true;
            if (line.find("Format: MCGPU-SPEC 1.0") != std::string::npos) mcgpu_edges = true;
            if (line.find("Energy[keV]") != std::string::npos && line.find("N[") != std::string::npos) {
                in_spectrum = true;
                continue;
            }
            if (!in_spectrum || line.empty() || line[0] == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream stream(line);
            SpectrumPoint point;
            if (stream >> point.energy_keV >> point.fluence_per_keV_cm2_mAs_at_1m &&
                point.energy_keV > 0 && point.fluence_per_keV_cm2_mAs_at_1m >= 0) {
                if (explicit_widths && (!(stream >> point.bin_width_keV) ||
                    !std::isfinite(point.bin_width_keV) || point.bin_width_keV <= 0))
                    throw std::runtime_error("invalid explicit spectrum bin width");
                result.push_back(point);
            }
        }
        if (result.empty())
            throw std::runtime_error("能谱没有正的有效光子权重");
        for (size_t i=1; i<result.size(); ++i)
            if (!(result[i].energy_keV > result[i-1].energy_keV))
                throw std::runtime_error("spectrum energies must be strictly increasing");
        if (mcgpu_edges) {
            if (result.size() < 2 || result.back().fluence_per_keV_cm2_mAs_at_1m != 0)
                throw std::runtime_error("MCGPU-SPEC requires an upper edge with zero fluence");
            for (size_t i = 0; i + 1 < result.size(); ++i) {
                result[i].bin_width_keV = result[i+1].energy_keV - result[i].energy_keV;
                result[i].energy_keV += 0.5 * result[i].bin_width_keV;
            }
            result.pop_back();
            return result;
        }
        if (explicit_widths) return result;
        for (size_t i=0; i<result.size(); ++i) {
            if (result.size() == 1) result[i].bin_width_keV = 1.0;
            else if (i == 0) result[i].bin_width_keV = result[1].energy_keV-result[0].energy_keV;
            else if (i+1 == result.size()) result[i].bin_width_keV = result[i].energy_keV-result[i-1].energy_keV;
            else result[i].bin_width_keV = 0.5*(result[i+1].energy_keV-result[i-1].energy_keV);
        }
        return result;
    }
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path& p) { std::ifstream f(p, std::ios::binary); if (!f)throw std::runtime_error("无法读取标签模体: " + p.string()); return { std::istreambuf_iterator<char>(f),{} }; }
}
