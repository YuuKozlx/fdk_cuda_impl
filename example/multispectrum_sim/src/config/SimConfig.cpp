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
        GeometryKind geometry(std::string_view v) { if (v == "flat_cbct")return GeometryKind::FlatCbct; if (v == "flat_helical")return GeometryKind::FlatHelical; if (v == "cyl_cbct")return GeometryKind::CylCbct; if (v == "cyl_helical")return GeometryKind::CylHelical; throw std::runtime_error("未知 geometry: " + std::string(v)); }
        WorkflowMode workflow(std::string_view v) { if (v == "project")return WorkflowMode::Project; if (v == "reconstruct")return WorkflowMode::Reconstruct; if (v == "project_and_reconstruct")return WorkflowMode::ProjectAndReconstruct; throw std::runtime_error("未知 workflow.mode: " + std::string(v)); }
        std::filesystem::path resolve(const std::filesystem::path& b, std::string_view v) {
            if (v.empty()) return {};
            std::filesystem::path p(v);
            return p.is_absolute() ? p : b / p;
        }
    }
}
namespace yk::spectral {
    SimulationConfig loadConfig(const std::filesystem::path& path) {
        auto t = toml::parse_file(path.string());
        const auto config_directory = path.parent_path();
        SimulationConfig c;
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
            c.projection.xcom_data_directory = resolve(config_directory, (*p)["xcom_data_directory"].value_or(std::string{}));
            c.projection.output_file = resolve(config_directory, (*p)["output_file"].value_or(std::string("projection.raw")));
            c.projection.energy_output_file = resolve(config_directory, (*p)["energy_output_file"].value_or(std::string{}));
            c.projection.air_output_file = resolve(config_directory, (*p)["air_output_file"].value_or(std::string{}));
            c.projection.apply_geometry_flux = (*p)["apply_geometry_flux"].value_or(true);
            c.projection.mAs_per_view = (*p)["mAs_per_view"].value_or(1.0);
            c.projection.quantum_noise_enabled = (*p)["quantum_noise_enabled"].value_or(true);
            c.projection.quantum_noise_seed = (*p)["quantum_noise_seed"].value_or(67890u);
            if (!std::isfinite(c.projection.mAs_per_view) || c.projection.mAs_per_view <= 0.0)
                throw std::runtime_error("projection.mAs_per_view must be positive");
            c.projection.save_metadata = (*p)["save_metadata"].value_or(true);
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
            auto* a = (*r)["analytic"].as_table();
            auto* i = (*r)["iterative"].as_table();
            if (c.reconstruction.type != "analytic" && c.reconstruction.type != "iterative")
                throw std::runtime_error("reconstruction.type 仅支持 analytic/iterative");
            if (c.reconstruction.type == "analytic" && !a)
                throw std::runtime_error("analytic 重建需要 [reconstruction.analytic]");
            if (c.reconstruction.type == "iterative" && !i)
                throw std::runtime_error("迭代重建需要 [reconstruction.iterative]");
            if (a) {
                for (const char* key : { "pipeline", "filter" })
                    if (a->contains(key) && !(*a)[key].is_string())
                        throw std::runtime_error(std::string("reconstruction.analytic.") + key + " 必须为字符串");
                for (const char* key : { "wfbp_cutoff", "wfbp_apodization", "chunk_views" })
                    if (a->contains(key) && !(*a)[key].is_number())
                        throw std::runtime_error(std::string("reconstruction.analytic.") + key + " 必须为数值");
                auto& rc = c.reconstruction.analytic;
                rc.pipeline = (*a)["pipeline"].value_or(std::string("fdk"));
                rc.filter = (*a)["filter"].value_or(std::string(rc.pipeline == "wfbp" ? "ramlak" : "shepp-logan"));
                rc.wfbp_cutoff = (*a)["wfbp_cutoff"].value_or(1.0);
                rc.wfbp_apodization = (*a)["wfbp_apodization"].value_or(1.0);
                rc.chunk_views = (*a)["chunk_views"].value_or(32);
                if (rc.pipeline != "fdk" && rc.pipeline != "wfbp")
                    throw std::runtime_error("analytic.pipeline 仅支持 fdk/wfbp");
                if (rc.filter != "ramlak" && rc.filter != "ram-lak" &&
                    rc.filter != "shepp-logan" && rc.filter != "cosine" &&
                    rc.filter != "hann" && rc.filter != "hamming")
                    throw std::runtime_error("未知 analytic.filter: " + rc.filter);
                if (rc.pipeline != "wfbp" &&
                    (a->contains("wfbp_cutoff") || a->contains("wfbp_apodization")))
                    throw std::runtime_error("wfbp_* 滤波参数只适用于 wfbp");
                if (rc.pipeline == "wfbp" && rc.filter != "ramlak" && rc.filter != "ram-lak")
                    throw std::runtime_error("wFBP 使用 FreeCT ramp 核，filter 必须为 ramlak");
                if (rc.chunk_views <= 0 || !std::isfinite(rc.wfbp_cutoff) || rc.wfbp_cutoff <= 0 || rc.wfbp_cutoff > 1 ||
                    !std::isfinite(rc.wfbp_apodization) || rc.wfbp_apodization < 0 || rc.wfbp_apodization > 1)
                    throw std::runtime_error("analytic 的 chunk_views/wfbp 参数无效");
            }
            if (i) {
                auto& rc = c.reconstruction.iterative;
                rc.algorithm = (*i)["algorithm"].value_or(std::string("os_sart_tv"));
                rc.iterations = (*i)["iterations"].value_or(10);
                rc.relaxation = (*i)["relaxation"].value_or(1.0);
                rc.subsets = (*i)["subsets"].value_or(1);
                if ((*i).contains("forward_projector") || (*i).contains("back_projector"))
                    throw std::runtime_error("Use iterative.projection_model; separate FP/BP selectors have been removed");
                rc.projection_model = (*i)["projection_model"].value_or(std::string("joseph"));
                if (rc.projection_model != "joseph" && rc.projection_model != "siddon")
                    throw std::runtime_error("projection_model supports joseph/siddon");
                rc.tv_iterations = (*i)["tv_iterations"].value_or(20);
                rc.tv_alpha = (*i)["tv_alpha"].value_or(0.002);
                rc.tv_alpha_reduction = (*i)["tv_alpha_reduction"].value_or(0.95);
                rc.maximum_update_ratio = (*i)["maximum_update_ratio"].value_or(0.95);
                rc.non_negative = (*i)["non_negative"].value_or(true);
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
            const auto output_tag = c.reconstruction.type == "iterative"
                ? c.reconstruction.iterative.algorithm : c.reconstruction.analytic.pipeline;
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
            c.geometry.parameters.voxel_x_mm = real("voxel_x_mm", 1.0); c.geometry.parameters.voxel_y_mm = real("voxel_y_mm", 1.0); c.geometry.parameters.voxel_z_mm = real("voxel_z_mm", 1.0); c.geometry.parameters.volume_x = integer("volume_x", 0); c.geometry.parameters.volume_y = integer("volume_y", 0); c.geometry.parameters.volume_z = integer("volume_z", 0);
            c.geometry.parameters.reconstruction_volume_x = integer("reconstruction_volume_x", c.geometry.parameters.volume_x); c.geometry.parameters.reconstruction_volume_y = integer("reconstruction_volume_y", c.geometry.parameters.volume_y); c.geometry.parameters.reconstruction_volume_z = integer("reconstruction_volume_z", c.geometry.parameters.volume_z); c.geometry.parameters.reconstruction_voxel_x_mm = real("reconstruction_voxel_x_mm", c.geometry.parameters.voxel_x_mm); c.geometry.parameters.reconstruction_voxel_y_mm = real("reconstruction_voxel_y_mm", c.geometry.parameters.voxel_y_mm); c.geometry.parameters.reconstruction_voxel_z_mm = real("reconstruction_voxel_z_mm", c.geometry.parameters.voxel_z_mm);
            c.geometry.parameters.phantom_offset_x_mm = real("phantom_offset_x_mm", 0.0); c.geometry.parameters.phantom_offset_y_mm = real("phantom_offset_y_mm", 0.0); c.geometry.parameters.phantom_offset_z_mm = real("phantom_offset_z_mm", 0.0);
            c.geometry.parameters.phantom_rotation_x_rad = real("phantom_rotation_x_rad", 0.0); c.geometry.parameters.phantom_rotation_y_rad = real("phantom_rotation_y_rad", 0.0); c.geometry.parameters.phantom_rotation_z_rad = real("phantom_rotation_z_rad", 0.0);
            if (!std::isfinite(c.geometry.parameters.phantom_rotation_x_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_y_rad) || !std::isfinite(c.geometry.parameters.phantom_rotation_z_rad)) throw std::runtime_error("geometry.phantom_rotation_*_rad 必须为有限数值");
            c.geometry.parameters.reconstruction_offset_x_mm = real("reconstruction_offset_x_mm", 0.0); c.geometry.parameters.reconstruction_offset_y_mm = real("reconstruction_offset_y_mm", 0.0); c.geometry.parameters.reconstruction_offset_z_mm = real("reconstruction_offset_z_mm", 0.0);
        }
        if (p) {
            {
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
            c.geometry.kind != GeometryKind::CylCbct && c.geometry.kind != GeometryKind::CylHelical)
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
            spot.size_u_mm = (*f)["size_u_mm"].value_or(0.0);
            spot.size_v_mm = (*f)["size_v_mm"].value_or(0.0);
            if (f->contains("rotation_deg")) throw std::runtime_error("矩形焦点不支持 rotation_deg");
            if (spot.enabled && (spot.size_u_mm <= 0 || spot.size_v_mm <= 0))
                throw std::runtime_error("矩形焦点尺寸必须为正");
        }
        if (c.projection.sampling.samples_per_pixel <= 0 || c.projection.sampling.samples_per_pixel > 4096)
            throw std::runtime_error("projection.sampling.samples_per_pixel must be in 1..4096");
        std::array<bool, 256> used_labels{};
        if (auto* a = p ? (*p)["materials"].as_array() : nullptr)for (auto& n : *a) { auto* m = n.as_table(); if (!m)throw std::runtime_error("projection.materials 必须是表数组"); MaterialSpec x; x.label = (std::uint8_t)(*m)["label"].value_or(0); x.name = (*m)["name"].value_or(std::string("material")); x.formula = (*m)["formula"].value_or(std::string{}); x.preset = (*m)["preset"].value_or(std::string{}); x.density_g_cm3 = (*m)["density_g_cm3"].value_or(1.0); if ((x.formula.empty() == x.preset.empty()) || x.density_g_cm3 <= 0)throw std::runtime_error("材料配置必须且只能填写 formula 或 preset，且密度必须为正"); if (used_labels[x.label])throw std::runtime_error("材料 label 不得重复: " + std::to_string(x.label)); used_labels[x.label] = true; if (!x.preset.empty()) { const auto* b = findBuiltinMaterial(x.preset); if (!b)throw std::runtime_error("未知内置材料 preset: " + x.preset); if (!m->contains("density_g_cm3"))x.density_g_cm3 = b->density_g_cm3; }c.projection.materials.push_back(std::move(x)); }
        if (c.runsProjection() && c.projection.materials.empty())
            throw std::runtime_error("投影工作流至少需要一个材料映射");
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            c.reconstruction.input_projection_file = c.projection.output_file;
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            throw std::runtime_error("重建工作流需要 reconstruction.input_projection_file");
        if (c.runsReconstruction()) {
            if ((c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "fdk" && c.geometry.kind != GeometryKind::FlatCbct && c.geometry.kind != GeometryKind::CylCbct) ||
                (c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "wfbp" && c.geometry.kind != GeometryKind::CylHelical))
                throw std::runtime_error("解析重建支持 flat_cbct/cyl_cbct+fdk、cyl_helical+wfbp");
        }
        return c;
    }
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path& p) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("无法读取能谱文件: " + p.string());
        std::vector<SpectrumPoint> result;
        std::string line;
        bool in_spectrum = false;
        while (std::getline(f, line)) {
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
                result.push_back(point);
            }
        }
        if (result.empty())
            throw std::runtime_error("能谱没有正的有效光子权重");
        for (size_t i=1; i<result.size(); ++i)
            if (!(result[i].energy_keV > result[i-1].energy_keV))
                throw std::runtime_error("spectrum energies must be strictly increasing");
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
