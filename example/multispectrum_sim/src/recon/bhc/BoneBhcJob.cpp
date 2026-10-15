#include "BoneBhcIo.hpp"
#include "LibraryReconstructor.hpp"
#include "recon/bhc/BoneBhcStage.hpp"
#include "recon/preprocess/BeamHardeningFit.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>
namespace yk::spectral::bone_bhc {
using Json = nlohmann::json;
using detail::openInput;
using detail::openOutput;
using detail::product;
void runBoneBhcJob(const std::filesystem::path& job_file)
{
    std::ifstream stream(job_file);
    stream.exceptions(std::ios::badbit);
    Json job;
    stream >> job;
    const auto base = std::filesystem::absolute(job_file).parent_path();
    auto resolve = [&](const char* key) {auto p=std::filesystem::u8path(job.at(key).get<std::string>());return p.is_absolute()?p:base/p; };
    Domain domain;
    if (job.at("input_domain") == "scatter_corrected_log_before_water_bhc")
        domain = Domain::ScatterCorrectedLog;
    else if (job.at("input_domain") == "scatter_and_water_corrected_log")
        domain = Domain::WaterCorrectedLog;
    else
        throw std::invalid_argument("unknown bone BHC input domain");
    auto config = loadConfig(resolve("config"));
    config.projection.spectrum.export_file.clear();
    // Keep the configured analytic reconstruction; TV/other postprocessing must not
    // silently change the material model inside this experimental physical loop.
    if (config.reconstruction.type != "analytic")
        throw std::invalid_argument("iBHC job currently requires analytic FDK/WFBP");
    const size_t elements = product(config.geometry.parameters.detector_u, config.geometry.parameters.detector_v, config.geometry.parameters.views);
    auto fit = config.reconstruction.preprocessing.beam_hardening;
    if (fit.enabled && fit.fit_from_water_plate)
        fit = preprocess::fitWaterPlate(config, fit, {});
    SegmentationOptions segmentation { job.value("air_threshold_hu", -500.), job.value("bone_threshold_hu", 300.) };
    std::unique_ptr<IBoneBhcAlgorithm> algorithm;
    const auto method = job.at("method").get<std::string>();
    if (method == "physical") {
        config.projection.materials.clear();
        for (int m = 1; m < 3; ++m) {
            const auto& value = job.at(m == 1 ? "soft_material" : "bone_material");
            MaterialSpec material;
            material.label = m;
            material.name = m == 1 ? "soft" : "bone";
            material.preset = value.at("preset").get<std::string>();
            material.density_g_cm3 = value.at("density_g_cm3").get<double>();
            config.projection.materials.push_back(material);
        }
        algorithm = std::make_unique<PhysicalBoneBhc>(config, domain, fit, segmentation);
    } else if (method == "two_material_lut") {
        if (domain != Domain::ScatterCorrectedLog)
            throw std::invalid_argument("LUT BHC requires data before water BHC");
        algorithm = std::make_unique<TwoMaterialLutBhc>(config, preprocess::TwoMaterialCalibration::load(resolve("calibration")), job.value("bone_thickness_scale", 1.), segmentation);
    } else
        throw std::invalid_argument("unknown bone BHC method");
    const std::string metric = algorithm->errorMetric();
    config.reconstruction.preprocessing.enabled = false;
    config.reconstruction.save_slices = false;
    Dataset data;
    data.original_projection = resolve("input_projection");
    data.domain = domain;
    Context context;
    context.output_directory = resolve("output_directory");
    context.options.max_iterations = job.value("max_iterations", 1);
    context.options.stop_ratio = job.value("stop_ratio", .78);
    context.reconstruct = [&](const auto& input, const auto& output) {
        std::string error;
        if (!reconstructWithLibraryFdk(config, input, output, {}, error))
            throw std::runtime_error(error);
    };
    if (fit.enabled && domain == Domain::ScatterCorrectedLog) {
        if (fit.coefficients.empty())
            throw std::invalid_argument("empty water BHC polynomial");
        context.initialize = [&](const auto& input, const auto& output) {
            auto in = openInput(input, elements * sizeof(float));
            auto out = openOutput(output);
            std::vector<float> b(std::min(elements, size_t(1 << 20)));
            for (size_t first = 0; first < elements; first += b.size()) {
                const size_t n = std::min(b.size(), elements - first);
                in.read(reinterpret_cast<char*>(b.data()), n * sizeof(float));
                for (size_t i = 0; i < n; ++i) {
                    if (!std::isfinite(b[i]))
                        throw std::runtime_error("nonfinite input projection");
                    double v = 0;
                    for (auto it = fit.coefficients.rbegin(); it != fit.coefficients.rend(); ++it)
                        v = v * b[i] + *it;
                    if (!std::isfinite(v))
                        throw std::runtime_error("nonfinite initial water correction");
                    b[i] = float(std::clamp(v, fit.clamp_min, fit.clamp_max));
                }
                out.write(reinterpret_cast<const char*>(b.data()), n * sizeof(float));
            }
        };
    }
    Json report = { { "method", method }, { "error_metric", metric }, { "job", job }, { "iterations", Json::array() },
        { "quality_validated", false }, { "frequency_splitting", false } };
    context.progress = [&](const Iteration& i) {
        report["iterations"].push_back({ { "iteration", i.number }, { "model_error", i.model_error ? Json(*i.model_error) : Json(nullptr) },
            { "model_ms", i.model_ms }, { "reconstruction_ms", i.reconstruction_ms } });
        auto out = openOutput(context.output_directory / "run.json");
        out << report.dump(2);
        std::cout << "ibhc iteration=" << i.number << " mse=" << (i.model_error ? std::to_string(*i.model_error) : "not_available") << " model_ms=" << i.model_ms << " reconstruction_ms=" << i.reconstruction_ms << std::endl;
    };
    auto result = std::make_shared<Result>();
    auto stage = std::make_unique<BoneBhcStage>(std::move(algorithm), result);
    stage->process(data, context);
    data.domain = stage->contract().output_domain;
    report["stopped_on_ratio"] = result->stopped_on_ratio;
    report["projection"] = data.corrected_projection.generic_string();
    report["reconstruction"] = data.reconstruction.generic_string();
    auto out = openOutput(context.output_directory / "run.json");
    out << report.dump(2);
    std::cout << "ibhc reconstruction=" << data.reconstruction << std::endl;
}
}
