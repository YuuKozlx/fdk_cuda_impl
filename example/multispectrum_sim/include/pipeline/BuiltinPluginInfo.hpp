#pragma once
#include "recon/bhc/BoneBhcStage.hpp"
#include "PluginInfo.hpp"
#include "recon/preprocess/correction/TwoMaterialBhcStage.hpp"
#include "image/ImagePluginInfo.hpp"
#include "recon/postprocess/stages/IntensityPluginInfo.hpp"
#include <algorithm>
namespace yk::spectral::pipeline {
enum class BuiltinPlugin {
    CpuScatter,
    CpuQuantum,
    CpuOptical,
    CpuAfterglow,
    CpuElectronic,
    CpuDas,
    CudaScatter,
    CudaQuantum,
    CudaOptical,
    CudaAfterglow,
    CudaElectronic,
    CudaDas,
    EnergyOutput,
    EnergyRestore,
    ScatterCorrection,
    Bhc,
    LogConversion,
    CorrectedOutput,
    HuConversion,
    RawOutput,
    BmpOutput,
    ExampleBhc,
    MultiSpecFp
};
// A descriptor catalog, not a claim that every factory is compiled/registered.
// Reading it performs no algorithm construction or device work.
inline const std::vector<PluginInfo>& builtinPluginDescriptions()
{
    static const auto descriptions = [] {
        std::vector<PluginInfo> result;
        auto add = [&](const char* id, const char* name, const char* description, PluginKind kind, const char* category, std::vector<std::string> tags, std::vector<ParameterInfo> parameters = {}) {
            PluginInfo p;p.id=id;p.name=name;p.description=description;p.version="1";p.kind=kind;p.category=category;p.tags=std::move(tags);p.vendor="YKCBCT";
            p.documentation="example/multispectrum_sim/include/pipeline/README.md";p.parameters=std::move(parameters);result.push_back(std::move(p)); };
        for (const auto* backend : { "cpu", "cuda" }) {
            const std::string prefix = std::string("projection.") + backend + ".";
            add((prefix + "scatter").c_str(), "scatter", "Apply segmented or blended FASKS scatter to primary energy signals.", PluginKind::ProjectionStage, "scatter", { backend, "fasks" }, { { "model_file", "Scatter model", "Scatter model file.", ParameterType::String, "" } });
            add((prefix + "quantum_noise").c_str(), "scatter_noise", "Sample photon statistics and scatter-related quantum noise.", PluginKind::ProjectionStage, "noise", { backend, "quantum" });
            add((prefix + "optical_crosstalk").c_str(), "optical_crosstalk", "Apply detector optical crosstalk to signal and air reference.", PluginKind::ProjectionStage, "detector", { backend, "crosstalk" });
            add((prefix + "afterglow").c_str(), "afterglow", "Apply temporal detector afterglow with separate signal and air histories.", PluginKind::ProjectionStage, "detector", { backend, "temporal" });
            add((prefix + "electronic_crosstalk").c_str(), "electronic_crosstalk", "Apply electronic crosstalk to signal and air reference.", PluginKind::ProjectionStage, "detector", { backend, "crosstalk" });
            add((prefix + "das").c_str(), "das", "Apply gain, offset, electronic noise, saturation and ADC conversion.", PluginKind::ProjectionStage, "detector", { backend, "readout" });
        }
        add("export.projection.energy", "Energy projection output", "Write energy, air reference and negative-log projection RAW files.", PluginKind::Exporter, "raw", { "cpu", "projection" });
        add("preprocess.energy_restore", "Energy restore", "Restore energy signal from negative-log projection and air reference.", PluginKind::PreprocessStage, "signal_conversion", { "cuda", "air" });
        add("preprocess.scatter", "Scatter correction", "Estimate and remove scatter before negative-log conversion.", PluginKind::PreprocessStage, "scatter", { "cuda", "fasks" });
        const std::vector<ParameterInfo> bhc = { { "coefficients", "Coefficients", "Polynomial coefficients in ascending order; fitted upstream.", ParameterType::RealArray, "" }, { "clamp_min", "Output minimum", "Lower corrected line-integral limit.", ParameterType::Real, "1" }, { "clamp_max", "Output maximum", "Upper corrected line-integral limit.", ParameterType::Real, "1" } };
        add("preprocess.bhc", "BHC / log conversion", "Convert energy when needed and apply double Horner beam-hardening correction.", PluginKind::PreprocessStage, "bhc", { "cuda", "polynomial" }, bhc);
        add("preprocess.log", "Log conversion", "Convert energy signal to negative-log attenuation.", PluginKind::PreprocessStage, "signal_conversion", { "cuda", "log" });
        add("export.projection.corrected_raw", "Corrected projection RAW", "Write corrected float32 negative-log projections in view order.", PluginKind::Exporter, "raw", { "cpu", "projection" });
        add("postprocess.hu", "Attenuation to HU", "Convert linear attenuation to Hounsfield units using water attenuation.", PluginKind::PostprocessStage, "unit_conversion", { "cpu", "hu" }, { { "water_mu_mm_inv", "Water attenuation", "Water attenuation at the BHC reference energy.", ParameterType::Real, "mm^-1" } });
        add("export.volume.raw", "RAW float32 export", "Write float32 attenuation volume in z/y/x order.", PluginKind::Exporter, "raw", { "cpu", "volume" });
        add("export.volume.bmp", "Orthogonal BMP export", "Write axial, coronal and sagittal center slices with diagnostic grayscale mapping.", PluginKind::Exporter, "bmp", { "cpu", "volume" });
        add("example.preprocess.bhc", "Example CPU polynomial BHC", "Teaching example: apply a calibrated polynomial directly to negative-log projections.", PluginKind::PreprocessStage, "bhc", { "cpu", "example", "polynomial" }, bhc);
        result.back().implementation_status = ImplementationStatus::Experimental;
        result.back().documentation = "example/multispectrum_sim/examples/README.md";
        add("projection.source.multi_spec_fp", "Multi-spectrum forward projection",
            "Generate projection batches from volume, geometry, materials and spectrum using the fused backend.",
            PluginKind::ProjectionSource, "forward_projection", { "spectral", "source", "batch" },
            { { "rays_per_pixel", "Rays per pixel", "Monte Carlo ray samples per detector pixel.", ParameterType::Integer, "1" } });
        result.push_back(preprocess::twoMaterialBhcInfo());
        result.push_back(bone_bhc::boneBhcInfo());
        const auto& images=image::pluginDescriptions();
        result.insert(result.end(),images.begin(),images.end());
        const auto& intensities=postprocess::intensityPluginDescriptions();
        result.insert(result.end(),intensities.begin(),intensities.end());
        return result;
    }();
    return descriptions;
}
inline const PluginInfo& builtinPluginInfo(BuiltinPlugin plugin) noexcept { return builtinPluginDescriptions()[static_cast<size_t>(plugin)]; }
inline const PluginInfo* findBuiltinPluginInfo(const std::string& id)
{
    const auto& all = builtinPluginDescriptions();
    const auto it = std::find_if(all.begin(), all.end(), [&](const PluginInfo& p) { return p.id == id; });
    return it == all.end() ? nullptr : &*it;
}
}
