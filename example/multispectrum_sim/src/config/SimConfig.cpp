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
        std::filesystem::path resolve(const std::filesystem::path& b, std::string_view v) { std::filesystem::path p(v); return p.is_absolute() ? p : b / p; }
    }
}
namespace yk::spectral {
    SimulationConfig loadConfig(const std::filesystem::path& path) {
        auto t = toml::parse_file(path.string()); auto base = path.parent_path(); SimulationConfig c;
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
            c.projection.label_volume = resolve(base, (*p)["label_volume"].value_or(std::string{}));
            c.projection.spectrum_file = resolve(base, (*p)["spectrum_file"].value_or(std::string{}));
            c.projection.xcom_data_directory = resolve(base, (*p)["xcom_data_directory"].value_or(std::string{}));
            c.projection.output_file = resolve(base, (*p)["output_file"].value_or(std::string("projection.raw")));
            c.projection.path_cache_file = resolve(base, (*p)["path_cache_file"].value_or(std::string("paths.ykpc")));
            c.projection.use_library_fp = (*p)["use_library_fp"].value_or(false);
            c.projection.use_cuda_spectral = (*p)["use_cuda_spectral"].value_or(false);
            c.projection.energy_output_file = resolve(base, (*p)["energy_output_file"].value_or(std::string{}));
            c.projection.apply_geometry_flux = (*p)["apply_geometry_flux"].value_or(true);
        }
        if (c.projection.energy_output_file.empty())
            c.projection.energy_output_file = c.projection.output_file.parent_path() /
                (c.projection.output_file.stem().string() + "-energy.raw");
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
            if (auto value = (*r)["input_projection_file"].value<std::string>())
                c.reconstruction.input_projection_file = resolve(base, *value);
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
                rc.forward_projector = (*i)["forward_projector"].value_or(std::string("joseph"));
                rc.back_projector = (*i)["back_projector"].value_or(std::string("joseph_v3"));
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
                if (rc.forward_projector != "joseph" && rc.forward_projector != "siddon")
                    throw std::runtime_error("iterative.forward_projector 仅支持 joseph/siddon");
                if (rc.back_projector != "joseph_v3" && rc.back_projector != "joseph" &&
                    rc.back_projector != "siddon" && rc.back_projector != "siddon_v2" && rc.back_projector != "siddon_v3")
                    throw std::runtime_error("iterative.back_projector 不支持该值");
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
                c.reconstruction.output_volume_file = resolve(base, *value);
            else
                c.reconstruction.output_volume_file = default_volume;
            if (auto value = (*r)["slice_prefix"].value<std::string>())
                c.reconstruction.slice_prefix = resolve(base, *value);
            else
                c.reconstruction.slice_prefix = default_prefix;
        }
        if (auto* g = t["geometry"].as_table()) {
            if (g->contains("tilt_u_rad") || g->contains("tilt_v_rad") ||
                g->contains("tilt_n_rad"))
                throw std::runtime_error(
                    "geometry_config 当前不开放探测器 tilt；仅支持 offset_u/n/v_mm");
            auto number = [&](const char* k, double d) {return (*g)[k].value_or(d); };
            c.geometry.parameters.views = (int)number("views", 360); c.geometry.parameters.views_per_turn = (int)number("views_per_turn", c.geometry.parameters.views); c.geometry.parameters.rotation_direction = (int)number("rotation_direction", 1); c.geometry.parameters.detector_u = (int)number("detector_u", 256); c.geometry.parameters.detector_v = (int)number("detector_v", 128);
            c.geometry.parameters.pixel_u_mm = number("pixel_u_mm", 1.0); c.geometry.parameters.pixel_v_mm = number("pixel_v_mm", 1.0); c.geometry.parameters.sid_mm = number("sid_mm", 500.0); c.geometry.parameters.sdd_mm = number("sdd_mm", 1000.0);
            c.geometry.parameters.offset_u_mm = number("offset_u_mm", 0.0); c.geometry.parameters.offset_n_mm = number("offset_n_mm", 0.0); c.geometry.parameters.offset_v_mm = number("offset_v_mm", 0.0);
            c.geometry.parameters.source_offset_x_mm = number("source_offset_x_mm", 0.0); c.geometry.parameters.source_offset_y_mm = number("source_offset_y_mm", 0.0); c.geometry.parameters.source_offset_z_mm = number("source_offset_z_mm", 0.0); c.geometry.parameters.start_angle_rad = number("start_angle_rad", 0.0); c.geometry.parameters.pitch_mm_per_turn = number("pitch_mm_per_turn", 0.0); c.geometry.parameters.start_z_mm = number("start_z_mm", 0.0);
            c.geometry.parameters.voxel_x_mm = number("voxel_x_mm", 1.0); c.geometry.parameters.voxel_y_mm = number("voxel_y_mm", 1.0); c.geometry.parameters.voxel_z_mm = number("voxel_z_mm", 1.0); c.geometry.parameters.volume_x = (int)number("volume_x", 0); c.geometry.parameters.volume_y = (int)number("volume_y", 0); c.geometry.parameters.volume_z = (int)number("volume_z", 0);
            c.geometry.parameters.reconstruction_volume_x = (int)number("reconstruction_volume_x", c.geometry.parameters.volume_x); c.geometry.parameters.reconstruction_volume_y = (int)number("reconstruction_volume_y", c.geometry.parameters.volume_y); c.geometry.parameters.reconstruction_volume_z = (int)number("reconstruction_volume_z", c.geometry.parameters.volume_z); c.geometry.parameters.reconstruction_voxel_x_mm = number("reconstruction_voxel_x_mm", c.geometry.parameters.voxel_x_mm); c.geometry.parameters.reconstruction_voxel_y_mm = number("reconstruction_voxel_y_mm", c.geometry.parameters.voxel_y_mm); c.geometry.parameters.reconstruction_voxel_z_mm = number("reconstruction_voxel_z_mm", c.geometry.parameters.voxel_z_mm);
            c.geometry.parameters.phantom_offset_x_mm = number("phantom_offset_x_mm", 0.0); c.geometry.parameters.phantom_offset_y_mm = number("phantom_offset_y_mm", 0.0); c.geometry.parameters.phantom_offset_z_mm = number("phantom_offset_z_mm", 0.0); c.geometry.parameters.reconstruction_offset_x_mm = number("reconstruction_offset_x_mm", 0.0); c.geometry.parameters.reconstruction_offset_y_mm = number("reconstruction_offset_y_mm", 0.0); c.geometry.parameters.reconstruction_offset_z_mm = number("reconstruction_offset_z_mm", 0.0);
        }
        if (p) {
            c.projection.engine = (*p)["engine"].value_or(std::string("deterministic"));
            if (c.projection.engine == "pixel_local_random" || c.projection.engine == "detector_global_random") {
                auto* s = (*p)[c.projection.engine].as_table();
                if (!s) throw std::runtime_error("projection.engine 对应的配置块缺失: [projection." + c.projection.engine + "]");
                c.projection.sampling.mode = c.projection.engine;
                c.projection.sampling.photon_count_mode =
                    (*s)["photon_count_mode"].value_or(std::string("poisson"));
                c.projection.sampling.samples_per_pixel = (*s)["samples_per_pixel"].value_or(1);
                c.projection.sampling.total_samples = (*s)["total_samples"].value_or<std::uint64_t>(0);
                c.projection.sampling.photons_per_pixel = (*s)["photons_per_pixel"].value_or(100000.0);
                c.projection.sampling.seed = (*s)["spatial_seed"].value_or(12345u);
                c.projection.sampling.photon_seed = (*s)["photon_seed"].value_or(67890u);
                if (c.projection.sampling.mode == "detector_global_random" && c.projection.sampling.total_samples == 0)
                    throw std::runtime_error("[projection.detector_global_random] 需要 total_samples");
            }
        }
        if ((c.projection.engine == "pixel_local_random" || c.projection.engine == "detector_global_random") &&
            c.geometry.kind != GeometryKind::FlatCbct &&
            c.geometry.kind != GeometryKind::FlatHelical)
            throw std::runtime_error("随机多能谱前投仅支持平板探测器");
        if ((c.projection.engine == "pixel_local_random" || c.projection.engine == "detector_global_random") &&
            c.projection.materials.size() > 32)
            throw std::runtime_error("随机多能谱前投当前最多支持32种材料");
        if (auto* e = p ? (*p)["detector_effects"].as_table() : nullptr) {
            c.projection.detector_effects.efficiency_enabled = (*e)["efficiency_enabled"].value_or(false);
            c.projection.detector_effects.efficiency = (*e)["efficiency"].value_or(1.0);
            c.projection.detector_effects.scatter_enabled = (*e)["scatter_enabled"].value_or(false);
            c.projection.detector_effects.optical_crosstalk_enabled = (*e)["optical_crosstalk_enabled"].value_or(false);
            c.projection.detector_effects.afterglow_enabled = (*e)["afterglow_enabled"].value_or(false);
            c.projection.detector_effects.electronic_noise_enabled = (*e)["electronic_noise_enabled"].value_or(false);
            c.projection.detector_effects.electronic_noise_sigma = (*e)["electronic_noise_sigma"].value_or(0.0);
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
        if (c.projection.engine == "pixel_local_random" &&
            (c.projection.sampling.samples_per_pixel <= 0 || c.projection.sampling.samples_per_pixel > 4096))
            throw std::runtime_error("projection.pixel_local_random.samples_per_pixel 必须在1..4096之间");
        if (c.projection.engine == "detector_global_random" && c.projection.sampling.total_samples == 0)
            throw std::runtime_error("projection.detector_global_random.total_samples 必须为正");
        if ((c.projection.engine == "pixel_local_random" || c.projection.engine == "detector_global_random") &&
            c.projection.sampling.photons_per_pixel <= 0)
            throw std::runtime_error("随机前投的 photons_per_pixel 必须为正");
        if ((c.projection.engine == "pixel_local_random" || c.projection.engine == "detector_global_random") &&
            c.projection.sampling.photon_count_mode != "poisson" &&
            c.projection.sampling.photon_count_mode != "fixed")
            throw std::runtime_error("photon_count_mode 仅支持 poisson/fixed");
        std::array<bool, 256> used_labels{};
        if (auto* a = p ? (*p)["materials"].as_array() : nullptr)for (auto& n : *a) { auto* m = n.as_table(); if (!m)throw std::runtime_error("projection.materials 必须是表数组"); MaterialSpec x; x.label = (std::uint8_t)(*m)["label"].value_or(0); x.name = (*m)["name"].value_or(std::string("material")); x.formula = (*m)["formula"].value_or(std::string{}); x.preset = (*m)["preset"].value_or(std::string{}); x.density_g_cm3 = (*m)["density_g_cm3"].value_or(1.0); if ((x.formula.empty() == x.preset.empty()) || x.density_g_cm3 <= 0)throw std::runtime_error("材料配置必须且只能填写 formula 或 preset，且密度必须为正"); if (used_labels[x.label])throw std::runtime_error("材料 label 不得重复: " + std::to_string(x.label)); used_labels[x.label] = true; if (!x.preset.empty()) { const auto* b = findBuiltinMaterial(x.preset); if (!b)throw std::runtime_error("未知内置材料 preset: " + x.preset); if (!m->contains("density_g_cm3"))x.density_g_cm3 = b->density_g_cm3; }c.projection.materials.push_back(std::move(x)); }
        if (c.runsProjection() && c.projection.materials.empty())
            throw std::runtime_error("投影工作流至少需要一个材料映射");
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            c.reconstruction.input_projection_file = c.projection.output_file;
        if (c.runsReconstruction() && c.reconstruction.input_projection_file.empty())
            throw std::runtime_error("重建工作流需要 reconstruction.input_projection_file");
        if (c.runsReconstruction()) {
            if (c.runsProjection() && !c.projection.use_library_fp) throw std::runtime_error("投影并重建要求 projection.use_library_fp=true");
            if ((c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "fdk" && c.geometry.kind != GeometryKind::FlatCbct) ||
                (c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "wfbp" && c.geometry.kind != GeometryKind::CylHelical))
                throw std::runtime_error("示例重建仅支持 flat_cbct+fdk、cyl_helical+wfbp；迭代重建支持四类几何");
        }
        return c;
    }
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path& p) {
        std::ifstream f(p);
        if (!f) throw std::runtime_error("无法读取能谱文件: " + p.string());
        std::vector<SpectrumPoint> result;
        std::string line;
        double total_photons = 0.0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream stream(line);
            SpectrumPoint point;
            if (stream >> point.energy_keV >> point.relative_photons &&
                point.energy_keV > 0 && point.relative_photons >= 0) {
                total_photons += point.relative_photons;
                result.push_back(point);
            }
        }
        if (result.empty() || !std::isfinite(total_photons) || total_photons <= 0.0)
            throw std::runtime_error("能谱没有正的有效光子权重");
        for (auto& point : result)
            point.relative_photons /= total_photons;
        return result;
    }
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path& p) { std::ifstream f(p, std::ios::binary); if (!f)throw std::runtime_error("无法读取标签模体: " + p.string()); return { std::istreambuf_iterator<char>(f),{} }; }
}
