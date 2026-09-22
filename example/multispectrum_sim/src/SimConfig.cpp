#include "SimConfig.hpp"
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
        std::filesystem::path resolve(const std::filesystem::path& b, std::string_view v) { std::filesystem::path p(v); return p.is_absolute() ? p : b / p; }
    }
}
namespace yk::spectral {
    SimulationConfig loadConfig(const std::filesystem::path& path) {
        auto t = toml::parse_file(path.string()); auto base = path.parent_path(); SimulationConfig c;
        auto* s = t["simulation"].as_table(); if (!s)throw std::runtime_error("缺少 [simulation] 配置表");
        c.geometry = geometry((*s)["geometry"].value_or<std::string>("flat_cbct"));
        c.label_volume = resolve(base, (*s)["label_volume"].value_or(std::string{})); c.spectrum_file = resolve(base, (*s)["spectrum_file"].value_or(std::string{})); c.xcom_data_directory = resolve(base, (*s)["xcom_data_directory"].value_or(std::string{})); c.output_file = resolve(base, (*s)["output_file"].value_or(std::string("projection.raw"))); c.path_cache_file = resolve(base, (*s)["path_cache_file"].value_or(std::string("paths.ykpc")));
        c.use_library_fp = (*s)["use_library_fp"].value_or(false);
        c.use_cuda_spectral = (*s)["use_cuda_spectral"].value_or(false);
        if (auto* p = t["projection"].as_table()) {
            c.energy_output_file = resolve(base, (*p)["energy_output_file"].value_or(std::string{}));
            c.apply_geometry_flux = (*p)["apply_geometry_flux"].value_or(true);
        }
        if (c.energy_output_file.empty())
            c.energy_output_file = c.output_file.parent_path() /
                (c.output_file.stem().string() + "-energy.raw");
        if (auto* r = t["reconstruction"].as_table()) {
            for (const char* key : { "type", "output_volume_file", "slice_prefix" })
                if (r->contains(key) && !(*r)[key].is_string())
                    throw std::runtime_error(std::string("reconstruction.") + key + " 必须为字符串");
            c.reconstruction.enabled = (*r)["enabled"].value_or(false);
            c.reconstruction.type = (*r)["type"].value_or(std::string("analytic"));
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
            const auto default_volume = c.output_file.parent_path() /
                (c.output_file.stem().string() + "-" + output_tag + "-volume.raw");
            const auto default_prefix = c.output_file.parent_path() /
                (c.output_file.stem().string() + "-" + output_tag);
            if (auto value = (*r)["output_volume_file"].value<std::string>())
                c.reconstruction.output_volume_file = resolve(base, *value);
            else
                c.reconstruction.output_volume_file = default_volume;
            if (auto value = (*r)["slice_prefix"].value<std::string>())
                c.reconstruction.slice_prefix = resolve(base, *value);
            else
                c.reconstruction.slice_prefix = default_prefix;
        }
        if (auto* g = t["geometry_config"].as_table()) {
            if (g->contains("tilt_u_rad") || g->contains("tilt_v_rad") ||
                g->contains("tilt_n_rad"))
                throw std::runtime_error(
                    "geometry_config 当前不开放探测器 tilt；仅支持 offset_u/n/v_mm");
            auto number = [&](const char* k, double d) {return (*g)[k].value_or(d); };
            c.geometry_config.views = (int)number("views", 360); c.geometry_config.views_per_turn = (int)number("views_per_turn", c.geometry_config.views); c.geometry_config.rotation_direction = (int)number("rotation_direction", 1); c.geometry_config.detector_u = (int)number("detector_u", 256); c.geometry_config.detector_v = (int)number("detector_v", 128);
            c.geometry_config.pixel_u_mm = number("pixel_u_mm", 1.0); c.geometry_config.pixel_v_mm = number("pixel_v_mm", 1.0); c.geometry_config.sid_mm = number("sid_mm", 500.0); c.geometry_config.sdd_mm = number("sdd_mm", 1000.0);
            c.geometry_config.offset_u_mm = number("offset_u_mm", 0.0); c.geometry_config.offset_n_mm = number("offset_n_mm", 0.0); c.geometry_config.offset_v_mm = number("offset_v_mm", 0.0);
            c.geometry_config.source_offset_x_mm = number("source_offset_x_mm", 0.0); c.geometry_config.source_offset_y_mm = number("source_offset_y_mm", 0.0); c.geometry_config.source_offset_z_mm = number("source_offset_z_mm", 0.0); c.geometry_config.start_angle_rad = number("start_angle_rad", 0.0); c.geometry_config.pitch_mm_per_turn = number("pitch_mm_per_turn", 0.0); c.geometry_config.start_z_mm = number("start_z_mm", 0.0);
            c.geometry_config.voxel_x_mm = number("voxel_x_mm", 1.0); c.geometry_config.voxel_y_mm = number("voxel_y_mm", 1.0); c.geometry_config.voxel_z_mm = number("voxel_z_mm", 1.0); c.geometry_config.volume_x = (int)number("volume_x", 0); c.geometry_config.volume_y = (int)number("volume_y", 0); c.geometry_config.volume_z = (int)number("volume_z", 0);
            c.geometry_config.phantom_offset_x_mm = number("phantom_offset_x_mm", 0.0); c.geometry_config.phantom_offset_y_mm = number("phantom_offset_y_mm", 0.0); c.geometry_config.phantom_offset_z_mm = number("phantom_offset_z_mm", 0.0); c.geometry_config.reconstruction_offset_x_mm = number("reconstruction_offset_x_mm", 0.0); c.geometry_config.reconstruction_offset_y_mm = number("reconstruction_offset_y_mm", 0.0); c.geometry_config.reconstruction_offset_z_mm = number("reconstruction_offset_z_mm", 0.0);
        }
        if (auto* e = t["detector_effects"].as_table()) {
            c.detector_effects.efficiency_enabled = (*e)["efficiency_enabled"].value_or(false);
            c.detector_effects.efficiency = (*e)["efficiency"].value_or(1.0);
            c.detector_effects.scatter_enabled = (*e)["scatter_enabled"].value_or(false);
            c.detector_effects.optical_crosstalk_enabled = (*e)["optical_crosstalk_enabled"].value_or(false);
            c.detector_effects.afterglow_enabled = (*e)["afterglow_enabled"].value_or(false);
            c.detector_effects.electronic_noise_enabled = (*e)["electronic_noise_enabled"].value_or(false);
            c.detector_effects.electronic_noise_sigma = (*e)["electronic_noise_sigma"].value_or(0.0);
        }
        std::array<bool, 256> used_labels{};
        if (auto* a = t["materials"].as_array())for (auto& n : *a) { auto* m = n.as_table(); if (!m)throw std::runtime_error("materials 必须是表数组"); MaterialSpec x; x.label = (std::uint8_t)(*m)["label"].value_or(0); x.name = (*m)["name"].value_or(std::string("material")); x.formula = (*m)["formula"].value_or(std::string{}); x.preset = (*m)["preset"].value_or(std::string{}); x.density_g_cm3 = (*m)["density_g_cm3"].value_or(1.0); if ((x.formula.empty() == x.preset.empty()) || x.density_g_cm3 <= 0)throw std::runtime_error("材料配置必须且只能填写 formula 或 preset，且密度必须为正"); if (used_labels[x.label])throw std::runtime_error("材料 label 不得重复: " + std::to_string(x.label)); used_labels[x.label] = true; if (!x.preset.empty()) { const auto* b = findBuiltinMaterial(x.preset); if (!b)throw std::runtime_error("未知内置材料 preset: " + x.preset); if (!m->contains("density_g_cm3"))x.density_g_cm3 = b->density_g_cm3; }c.materials.push_back(std::move(x)); }
        if (c.materials.empty())throw std::runtime_error("至少需要一个材料映射");
        if (c.reconstruction.enabled) {
            if (!c.use_library_fp) throw std::runtime_error("重建要求 use_library_fp=true");
            if ((c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "fdk" && c.geometry != GeometryKind::FlatCbct) ||
                (c.reconstruction.type == "analytic" && c.reconstruction.analytic.pipeline == "wfbp" && c.geometry != GeometryKind::CylHelical))
                throw std::runtime_error("示例重建仅支持 flat_cbct+fdk、cyl_helical+wfbp；迭代重建支持四类几何");
        }
        return c;
    }
    std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path& p) { std::ifstream f(p); if (!f)throw std::runtime_error("无法读取能谱文件: " + p.string()); std::vector<SpectrumPoint> r; std::string l; while (std::getline(f, l)) { if (l.empty() || l[0] == '#')continue; std::replace(l.begin(), l.end(), ',', ' '); std::istringstream s(l); SpectrumPoint x; if (s >> x.energy_keV >> x.relative_photons && x.energy_keV > 0 && x.relative_photons >= 0)r.push_back(x); }if (r.empty())throw std::runtime_error("能谱没有有效数据"); return r; }
    std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path& p) { std::ifstream f(p, std::ios::binary); if (!f)throw std::runtime_error("无法读取标签模体: " + p.string()); return { std::istreambuf_iterator<char>(f),{} }; }
}
