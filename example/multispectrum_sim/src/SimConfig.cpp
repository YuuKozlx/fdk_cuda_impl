#include "SimConfig.hpp"
#include "XcomAttenuationProvider.hpp"
#include <toml++/toml.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
namespace yk::spectral { namespace {
GeometryKind geometry(std::string_view v){if(v=="flat_cbct")return GeometryKind::FlatCbct;if(v=="flat_helical")return GeometryKind::FlatHelical;if(v=="cyl_cbct")return GeometryKind::CylCbct;if(v=="cyl_helical")return GeometryKind::CylHelical;throw std::runtime_error("未知 geometry: "+std::string(v));}
std::filesystem::path resolve(const std::filesystem::path& b,std::string_view v){std::filesystem::path p(v);return p.is_absolute()?p:b/p;}
}}
namespace yk::spectral {
SimulationConfig loadConfig(const std::filesystem::path& path){
    auto t=toml::parse_file(path.string()); auto base=path.parent_path(); SimulationConfig c;
    auto* s=t["simulation"].as_table(); if(!s)throw std::runtime_error("缺少 [simulation] 配置表");
    c.geometry=geometry((*s)["geometry"].value_or<std::string>("flat_cbct"));
    c.label_volume=resolve(base,(*s)["label_volume"].value_or(std::string{})); c.spectrum_file=resolve(base,(*s)["spectrum_file"].value_or(std::string{})); c.xcom_data_directory=resolve(base,(*s)["xcom_data_directory"].value_or(std::string{})); c.output_file=resolve(base,(*s)["output_file"].value_or(std::string("projection.raw"))); c.path_cache_file=resolve(base,(*s)["path_cache_file"].value_or(std::string("paths.ykpc")));
    c.use_library_fp=(*s)["use_library_fp"].value_or(false);
    c.use_cuda_spectral=(*s)["use_cuda_spectral"].value_or(false);
    if (auto* r = t["reconstruction"].as_table()) {
        for (const char* key : {"pipeline", "filter", "output_volume_file", "slice_prefix"})
            if (r->contains(key) && !(*r)[key].is_string())
                throw std::runtime_error(std::string("reconstruction.") + key + " 必须为字符串");
        for (const char* key : {"wfbp_cutoff", "wfbp_apodization"})
            if (r->contains(key) && !(*r)[key].is_number())
                throw std::runtime_error(std::string("reconstruction.") + key + " 必须为数值");
        if (r->contains("pipeline"))
            c.reconstruction.pipeline = (*r)["pipeline"].value_or(
                std::string("fdk"));
        c.reconstruction.enabled = (*r)["enabled"].value_or(false);
        auto& rc = c.reconstruction;
        rc.filter = (*r)["filter"].value_or(std::string(rc.pipeline == "wfbp" ? "ramlak" : "shepp-logan"));
        rc.wfbp_cutoff = (*r)["wfbp_cutoff"].value_or(1.0);
        rc.wfbp_apodization = (*r)["wfbp_apodization"].value_or(1.0);
        if (rc.pipeline != "fdk" && rc.pipeline != "wfbp")
            throw std::runtime_error("reconstruction.pipeline 仅支持 fdk/wfbp");
        if (rc.filter != "ramlak" && rc.filter != "ram-lak" &&
            rc.filter != "shepp-logan" && rc.filter != "cosine" &&
            rc.filter != "hann" && rc.filter != "hamming")
            throw std::runtime_error("未知 reconstruction.filter: " + rc.filter);
        if (rc.pipeline == "wfbp" && rc.filter != "ramlak" && rc.filter != "ram-lak")
            throw std::runtime_error("wFBP 使用 FreeCT ramp 核，filter 必须为 ramlak；加窗使用 wfbp_apodization");
        if (rc.pipeline != "wfbp" && (r->contains("wfbp_cutoff") || r->contains("wfbp_apodization")))
            throw std::runtime_error("wfbp_* 滤波参数只适用于 wfbp");
        if (!std::isfinite(rc.wfbp_cutoff) || rc.wfbp_cutoff <= 0 || rc.wfbp_cutoff > 1 ||
            !std::isfinite(rc.wfbp_apodization) || rc.wfbp_apodization < 0 || rc.wfbp_apodization > 1)
            throw std::runtime_error("wfbp_cutoff 必须在 (0,1]，wfbp_apodization 必须在 [0,1]");
        c.reconstruction.chunk_views = static_cast<int>(
            (*r)["chunk_views"].value_or(32));
        const auto default_volume = c.output_file.parent_path() /
            (c.output_file.stem().string() + "-fdk-volume.raw");
        const auto default_prefix = c.output_file.parent_path() /
            (c.output_file.stem().string() + "-fdk");
        if (auto value = (*r)["output_volume_file"].value<std::string>())
            c.reconstruction.output_volume_file = resolve(base, *value);
        else
            c.reconstruction.output_volume_file = default_volume;
        if (auto value = (*r)["slice_prefix"].value<std::string>())
            c.reconstruction.slice_prefix = resolve(base, *value);
        else
            c.reconstruction.slice_prefix = default_prefix;
    }
    if (c.reconstruction.chunk_views <= 0)
        throw std::runtime_error("reconstruction.chunk_views 必须为正");
    if(auto* g=t["geometry_config"].as_table()) {
        if (g->contains("tilt_u_rad") || g->contains("tilt_v_rad") ||
            g->contains("tilt_n_rad"))
            throw std::runtime_error(
                "geometry_config 当前不开放探测器 tilt；仅支持 offset_u/n/v_mm");
        auto number=[&](const char* k,double d){return (*g)[k].value_or(d);};
        c.geometry_config.views=(int)number("views",360); c.geometry_config.views_per_turn=(int)number("views_per_turn",c.geometry_config.views); c.geometry_config.rotation_direction=(int)number("rotation_direction",1); c.geometry_config.detector_u=(int)number("detector_u",256); c.geometry_config.detector_v=(int)number("detector_v",128);
        c.geometry_config.pixel_u_mm=number("pixel_u_mm",1.0); c.geometry_config.pixel_v_mm=number("pixel_v_mm",1.0); c.geometry_config.sid_mm=number("sid_mm",500.0); c.geometry_config.sdd_mm=number("sdd_mm",1000.0);
        c.geometry_config.offset_u_mm=number("offset_u_mm",0.0); c.geometry_config.offset_n_mm=number("offset_n_mm",0.0); c.geometry_config.offset_v_mm=number("offset_v_mm",0.0);
        c.geometry_config.source_offset_x_mm=number("source_offset_x_mm",0.0); c.geometry_config.source_offset_y_mm=number("source_offset_y_mm",0.0); c.geometry_config.source_offset_z_mm=number("source_offset_z_mm",0.0); c.geometry_config.start_angle_rad=number("start_angle_rad",0.0); c.geometry_config.pitch_mm_per_turn=number("pitch_mm_per_turn",0.0); c.geometry_config.start_z_mm=number("start_z_mm",0.0);
        c.geometry_config.voxel_x_mm=number("voxel_x_mm",1.0); c.geometry_config.voxel_y_mm=number("voxel_y_mm",1.0); c.geometry_config.voxel_z_mm=number("voxel_z_mm",1.0); c.geometry_config.volume_x=(int)number("volume_x",0); c.geometry_config.volume_y=(int)number("volume_y",0); c.geometry_config.volume_z=(int)number("volume_z",0);
        c.geometry_config.phantom_offset_x_mm=number("phantom_offset_x_mm",0.0); c.geometry_config.phantom_offset_y_mm=number("phantom_offset_y_mm",0.0); c.geometry_config.phantom_offset_z_mm=number("phantom_offset_z_mm",0.0); c.geometry_config.reconstruction_offset_x_mm=number("reconstruction_offset_x_mm",0.0); c.geometry_config.reconstruction_offset_y_mm=number("reconstruction_offset_y_mm",0.0); c.geometry_config.reconstruction_offset_z_mm=number("reconstruction_offset_z_mm",0.0);
    }
    if(auto* e=t["detector_effects"].as_table()){c.detector_effects.efficiency_enabled=(*e)["efficiency_enabled"].value_or(false);c.detector_effects.efficiency=(*e)["efficiency"].value_or(1.0);c.detector_effects.scatter_enabled=(*e)["scatter_enabled"].value_or(false);c.detector_effects.optical_crosstalk_enabled=(*e)["optical_crosstalk_enabled"].value_or(false);c.detector_effects.afterglow_enabled=(*e)["afterglow_enabled"].value_or(false);c.detector_effects.electronic_noise_enabled=(*e)["electronic_noise_enabled"].value_or(false);c.detector_effects.electronic_noise_sigma=(*e)["electronic_noise_sigma"].value_or(0.0);}
    std::array<bool, 256> used_labels{};
    if(auto* a=t["materials"].as_array())for(auto& n:*a){auto* m=n.as_table();if(!m)throw std::runtime_error("materials 必须是表数组");MaterialSpec x; x.label=(std::uint8_t)(*m)["label"].value_or(0);x.name=(*m)["name"].value_or(std::string("material"));x.formula=(*m)["formula"].value_or(std::string{});x.preset=(*m)["preset"].value_or(std::string{});x.density_g_cm3=(*m)["density_g_cm3"].value_or(1.0);if((x.formula.empty() == x.preset.empty())||x.density_g_cm3<=0)throw std::runtime_error("材料配置必须且只能填写 formula 或 preset，且密度必须为正");if(used_labels[x.label])throw std::runtime_error("材料 label 不得重复: "+std::to_string(x.label));used_labels[x.label]=true;if(!x.preset.empty()){const auto* b=findBuiltinMaterial(x.preset);if(!b)throw std::runtime_error("未知内置材料 preset: "+x.preset);if(!m->contains("density_g_cm3"))x.density_g_cm3=b->density_g_cm3;}c.materials.push_back(std::move(x));}
    if(c.materials.empty())throw std::runtime_error("至少需要一个材料映射");
    if (c.reconstruction.enabled) {
        if (!c.use_library_fp) throw std::runtime_error("重建要求 use_library_fp=true");
        if ((c.reconstruction.pipeline == "fdk" && c.geometry != GeometryKind::FlatCbct) ||
            (c.reconstruction.pipeline == "wfbp" && c.geometry != GeometryKind::CylHelical))
            throw std::runtime_error("示例重建仅支持 flat_cbct+fdk 或 cyl_helical+wfbp");
    }
    return c;
}
std::vector<SpectrumPoint> loadSpectrum(const std::filesystem::path& p){std::ifstream f(p);if(!f)throw std::runtime_error("无法读取能谱文件: "+p.string());std::vector<SpectrumPoint> r;std::string l;while(std::getline(f,l)){if(l.empty()||l[0]=='#')continue;std::replace(l.begin(),l.end(),',',' ');std::istringstream s(l);SpectrumPoint x;if(s>>x.energy_keV>>x.relative_photons&&x.energy_keV>0&&x.relative_photons>=0)r.push_back(x);}if(r.empty())throw std::runtime_error("能谱没有有效数据");return r;}
std::vector<std::uint8_t> loadLabelVolume(const std::filesystem::path& p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("无法读取标签模体: "+p.string());return {std::istreambuf_iterator<char>(f),{}};}
}
