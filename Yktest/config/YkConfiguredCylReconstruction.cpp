#include "config/YkConfiguredCylReconstruction.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <toml++/toml.hpp>

#include "CylFpBp/Analytic/YkCylFdkPipeline.hpp"
#include "CylFpBp/FP/YkCylForwardProjection.hpp"
#include "YkTestGeometry.hpp"
#include "Reconstruction/Iterative/Helical/YkHelicalCylIterativeReconstructor.hpp"
#include "CylFpBp/Analytic/YkCylAnalyticReconstruction.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkExecutionContext.hpp"
#include "global/YkLog.h"
#include "global/YkMem3d.hpp"

namespace YK::TestConfig {
namespace {

constexpr float kPi = 3.14159265358979323846f;

template <typename T>
T required(const toml::table& table, std::string_view key,
    std::string_view context)
{
    if (auto value = table[key].value<T>()) return *value;
    throw std::runtime_error(std::string(context) + "." + std::string(key) +
        " 缺失或类型错误");
}

template <typename T>
T optional(const toml::table& table, std::string_view key, T fallback)
{
    if (auto value = table[key].value<T>()) return *value;
    return fallback;
}

const toml::table& requiredTable(const toml::table& table,
    std::string_view key, std::string_view context)
{
    const auto* result = table[key].as_table();
    if (!result) throw std::runtime_error(std::string(context) + "." +
        std::string(key) + " 必须是 table");
    return *result;
}

enum class Pipeline { Fdk, Sirt, Sart, Ossart, Cgls, Pwls, FdkOssart, FdkCgls };

struct Case {
    std::string name;
    int device = 0;
    SReconstructionParams params{};
    float pitch_mm = 0.f;
    float start_z_mm = 0.f;
    int views_per_rotation = 0;
    float curvature_radius_mm = 0.f;
    float channel_angle_step_rad = 0.f;
    float samples_per_voxel = 1.f;
    int projection_batch_views = 0;
    bool replay_streaming = false;
    Pipeline pipeline = Pipeline::Ossart;
    CylFpBp::IterativeConfig iterative{};
    CylFpBp::CylPwlsConfig pwls{};
    SFilterKernelDesc filter = SFilterKernelDesc::RamLak();
    std::filesystem::path output;
    std::filesystem::path preview;
};

Pipeline parsePipeline(const std::string& value)
{
    if (value == "fdk") return Pipeline::Fdk;
    if (value == "sirt") return Pipeline::Sirt;
    if (value == "sart") return Pipeline::Sart;
    if (value == "ossart") return Pipeline::Ossart;
    if (value == "cgls") return Pipeline::Cgls;
    if (value == "pwls") return Pipeline::Pwls;
    if (value == "fdk-ossart") return Pipeline::FdkOssart;
    if (value == "fdk-cgls") return Pipeline::FdkCgls;
    throw std::runtime_error(
        "圆柱 pipeline 必须是 fdk、sirt、sart、ossart、cgls、pwls、"
        "fdk-ossart 或 fdk-cgls");
}

CylFpBp::EIterativeForwardModel parseForwardModel(const std::string& value)
{
    if (value == "joseph") return CylFpBp::EIterativeForwardModel::Joseph;
    if (value == "siddon") return CylFpBp::EIterativeForwardModel::Siddon;
    if (value == "joseph-matched-reference")
        return CylFpBp::EIterativeForwardModel::JosephMatchedReference;
    throw std::runtime_error("forward_model 必须是 joseph、siddon 或 joseph-matched-reference");
}

CylFpBp::EIterativeBackprojectorModel parseBackprojectorModel(
    const std::string& value)
{
    if (value == "joseph") return CylFpBp::EIterativeBackprojectorModel::Joseph;
    if (value == "siddon")
        return CylFpBp::EIterativeBackprojectorModel::Siddon;
    if (value == "siddon-v2")
        return CylFpBp::EIterativeBackprojectorModel::SiddonV2;
    if (value == "siddon-v3")
        return CylFpBp::EIterativeBackprojectorModel::SiddonV3;
    if (value == "siddon-ray-driven")
        return CylFpBp::EIterativeBackprojectorModel::SiddonRayDriven;
    if (value == "joseph-v3")
        return CylFpBp::EIterativeBackprojectorModel::JosephV3;
    if (value == "fdk") return CylFpBp::EIterativeBackprojectorModel::Fdk;
    if (value == "fdk-matched")
        return CylFpBp::EIterativeBackprojectorModel::FdkMatched;
    throw std::runtime_error(
        "backprojector_model 必须是 joseph、siddon、siddon-v2、"
        "siddon-ray-driven、joseph-v3、fdk 或 fdk-matched");
}

std::filesystem::path resolve(const std::filesystem::path& base,
    const std::string& value)
{
    const std::filesystem::path path(value);
    return path.is_absolute() ? path.lexically_normal() :
        (base / path).lexically_normal();
}

Case parseCase(const toml::table& table, const std::filesystem::path& base)
{
    Case result{};
    result.name = required<std::string>(table, "name", "case");
    result.device = static_cast<int>(optional<int64_t>(table, "device", 0));
    result.pipeline = parsePipeline(required<std::string>(table, "pipeline", result.name));

    auto& p = result.params;
    const auto& scan = requiredTable(table, "scan", result.name);
    p.scan.Nu = static_cast<int>(required<int64_t>(scan, "nu", "scan"));
    p.scan.Nv = static_cast<int>(required<int64_t>(scan, "nv", "scan"));
    p.scan.NAng = static_cast<int>(required<int64_t>(scan, "views", "scan"));
    p.scan.du_mm = static_cast<float>(optional<double>(scan, "du_mm", 1.0));
    p.scan.dv_mm = static_cast<float>(optional<double>(scan, "dv_mm", 1.0));
    p.scan.offsetU_mm = static_cast<float>(optional<double>(scan, "offset_u_mm", 0.0));
    p.scan.offsetV_mm = static_cast<float>(optional<double>(scan, "offset_v_mm", 0.0));
    p.scan.sid_mm = static_cast<float>(required<double>(scan, "sid_mm", "scan"));
    p.scan.sdd_mm = static_cast<float>(required<double>(scan, "sdd_mm", "scan"));
    const float start_angle = static_cast<float>(
        optional<double>(scan, "start_angle_deg", 0.0) * kPi / 180.0);
    const float range = static_cast<float>(
        optional<double>(scan, "scan_range_deg", 360.0) * kPi / 180.0);
    p.scan.angles.resize(p.scan.NAng);
    for (int i = 0; i < p.scan.NAng; ++i)
        p.scan.angles[i] = start_angle + range * static_cast<float>(i) / p.scan.NAng;

    const auto& volume = requiredTable(table, "volume", result.name);
    p.volume.Nx = static_cast<int>(required<int64_t>(volume, "nx", "volume"));
    p.volume.Ny = static_cast<int>(required<int64_t>(volume, "ny", "volume"));
    p.volume.Nz = static_cast<int>(required<int64_t>(volume, "nz", "volume"));
    const auto* voxel = volume["voxel_mm"].as_array();
    if (!voxel || voxel->size() != 3) throw std::runtime_error(
        "volume.voxel_mm 必须包含 3 个数");
    p.volume.voxelX_mm = static_cast<float>((*voxel)[0].value_or(0.0));
    p.volume.voxelY_mm = static_cast<float>((*voxel)[1].value_or(0.0));
    p.volume.voxelZ_mm = static_cast<float>((*voxel)[2].value_or(0.0));

    const auto& geometry = requiredTable(table, "geometry", result.name);
    result.pitch_mm = static_cast<float>(optional<double>(geometry, "pitch_mm", 0.0));
    result.start_z_mm = static_cast<float>(optional<double>(geometry, "start_z_mm", 0.0));
    result.views_per_rotation = static_cast<int>(optional<int64_t>(geometry,
        "views_per_rotation", p.scan.NAng));
    result.curvature_radius_mm = static_cast<float>(optional<double>(geometry,
        "curvature_radius_mm", p.scan.sdd_mm));
    result.channel_angle_step_rad = static_cast<float>(optional<double>(geometry,
        "channel_angle_step_rad", 0.0));

    if (const auto* algorithm = table["algorithm"].as_table()) {
        result.iterative.iterations = static_cast<int>(optional<int64_t>(
            *algorithm, "iterations", 5));
        result.iterative.subset_count = static_cast<int>(optional<int64_t>(
            *algorithm, "subsets", 8));
        result.iterative.relaxation = static_cast<float>(optional<double>(
            *algorithm, "relaxation", 0.2));
        result.iterative.relaxation_reduction = static_cast<float>(optional<double>(
            *algorithm, "relaxation_reduction", 1.0));
        result.iterative.epsilon = static_cast<float>(optional<double>(
            *algorithm, "epsilon", 1e-6));
        result.iterative.nonnegative = optional<bool>(*algorithm, "nonnegative", true);
        result.iterative.use_max = optional<bool>(*algorithm, "use_max", false);
        result.iterative.maximum = static_cast<float>(optional<double>(
            *algorithm, "maximum", 1e30));
        result.iterative.forward_model = parseForwardModel(optional<std::string>(
            *algorithm, "forward_model", "joseph"));
        if ((*algorithm).contains("backprojector_model"))
            result.iterative.backprojector_model = parseBackprojectorModel(
                required<std::string>(*algorithm, "backprojector_model", "algorithm"));
        else
            result.iterative.backprojector_model = optional<bool>(
                *algorithm, "use_v3_backprojector", true) ?
                CylFpBp::EIterativeBackprojectorModel::JosephV3 :
                CylFpBp::EIterativeBackprojectorModel::Joseph;
        if (const auto* convergence = (*algorithm)["convergence"].as_table()) {
            auto& c = result.iterative.convergence;
            c.relative_residual_tolerance = static_cast<float>(optional<double>(
                *convergence, "relative_residual", 0.0));
            c.relative_update_tolerance = static_cast<float>(optional<double>(
                *convergence, "relative_update", 0.0));
            c.relative_improvement_tolerance = static_cast<float>(optional<double>(
                *convergence, "relative_improvement", 0.0));
            c.minimum_iterations = static_cast<int>(optional<int64_t>(
                *convergence, "minimum_iterations", 1));
            c.check_interval = static_cast<int>(optional<int64_t>(
                *convergence, "check_interval", 1));
            c.patience = static_cast<int>(optional<int64_t>(
                *convergence, "patience", 1));
        }
        if (const auto* weighting = (*algorithm)["weighting"].as_table()) {
            result.iterative.weighting.view_quadrature = optional<bool>(
                *weighting, "view_quadrature", true);
            result.iterative.weighting.ray_coverage = optional<bool>(
                *weighting, "helical_ray_coverage", true);
        }
        result.samples_per_voxel = static_cast<float>(optional<double>(
            *algorithm, "samples_per_voxel", 1.0));
        result.projection_batch_views = static_cast<int>(optional<int64_t>(
            *algorithm, "projection_batch_views", 0));
        result.replay_streaming = optional<bool>(*algorithm,
            "replay_streaming", false);
    }
    if (const auto* pwls = table["pwls"].as_table()) {
        result.pwls.iterations = static_cast<int>(optional<int64_t>(*pwls,
            "iterations", 5));
        result.pwls.relaxation = static_cast<float>(optional<double>(*pwls,
            "relaxation", 0.8));
        result.pwls.regularization = static_cast<float>(optional<double>(*pwls,
            "regularization", 1e-3));
        result.pwls.huber_delta = static_cast<float>(optional<double>(*pwls,
            "huber_delta", 3e-3));
        result.pwls.epsilon = static_cast<float>(optional<double>(*pwls,
            "epsilon", 1e-6));
        result.pwls.lower_bound = static_cast<float>(optional<double>(*pwls,
            "lower_bound", 0.0));
        result.pwls.upper_bound = static_cast<float>(optional<double>(*pwls,
            "upper_bound", 0.12));
        result.pwls.projection_weight_scale = static_cast<float>(
            optional<double>(*pwls, "projection_weight_scale", 1.0));
        const std::string regularizer = optional<std::string>(*pwls,
            "regularizer", "quadratic");
        if (regularizer == "none")
            result.pwls.regularizer = Iter::EPwlsRegularizer::None;
        else if (regularizer == "quadratic")
            result.pwls.regularizer = Iter::EPwlsRegularizer::Quadratic;
        else if (regularizer == "huber")
            result.pwls.regularizer = Iter::EPwlsRegularizer::Huber;
        else throw std::runtime_error(
            "pwls.regularizer 必须是 none、quadratic 或 huber");
        const std::string data_model = optional<std::string>(*pwls,
            "data_model", "joseph-matched");
        if (data_model == "joseph-matched")
            result.pwls.data_model = CylFpBp::ECylPwlsDataModel::JosephMatched;
        else if (data_model == "siddon")
            result.pwls.data_model = CylFpBp::ECylPwlsDataModel::Siddon;
        else throw std::runtime_error(
            "pwls.data_model 必须是 joseph-matched 或 siddon");
    }
    result.iterative.method = result.pipeline == Pipeline::Sirt ?
        CylFpBp::EIterativeMethod::Sirt : result.pipeline == Pipeline::Sart ?
        CylFpBp::EIterativeMethod::Sart :
        (result.pipeline == Pipeline::Cgls || result.pipeline == Pipeline::FdkCgls) ?
        CylFpBp::EIterativeMethod::Cgls : CylFpBp::EIterativeMethod::Ossart;

    const auto& input = requiredTable(table, "input", result.name);
    if (optional<std::string>(input, "mode", "phantom") != "phantom" ||
        optional<std::string>(input, "phantom", "catphan") != "catphan")
        throw std::runtime_error("圆柱重建测试当前只接受内置 catphan 模体");
    const auto& output = requiredTable(table, "output", result.name);
    result.output = resolve(base, required<std::string>(output, "volume", "output"));
    const std::string preview = optional<std::string>(output, "preview", "");
    if (!preview.empty()) result.preview = resolve(base, preview);

    const float configured_step = range / p.scan.NAng;
    const float expected_step = 2.f * kPi / result.views_per_rotation;
    if (result.device < 0 || p.scan.Nu < 2 || p.scan.Nv < 2 || p.scan.NAng < 4 ||
        p.volume.Nx <= 0 || p.volume.Ny <= 0 || p.volume.Nz <= 0 ||
        p.volume.voxelX_mm <= 0.f || p.volume.voxelY_mm <= 0.f ||
        p.volume.voxelZ_mm <= 0.f || p.scan.sid_mm <= 0.f ||
        p.scan.sdd_mm <= p.scan.sid_mm || result.curvature_radius_mm <= 0.f ||
        result.projection_batch_views < 0 ||
        result.views_per_rotation < 4 ||
        std::fabs(configured_step - expected_step) > 1e-5f * expected_step)
        throw std::runtime_error(result.name + " 的尺寸或圆柱几何参数无效");
    if ((result.pipeline == Pipeline::Fdk || result.pipeline == Pipeline::FdkOssart ||
         result.pipeline == Pipeline::FdkCgls) &&
        std::fabs(result.pitch_mm) > 1e-6f)
        throw std::runtime_error("柱面解析 FDK 当前只接受静态完整圆扫");
    if (result.replay_streaming &&
        (result.pipeline == Pipeline::Fdk || result.pipeline == Pipeline::Cgls ||
         result.pipeline == Pipeline::Pwls ||
         result.pipeline == Pipeline::FdkCgls ||
         result.pipeline == Pipeline::FdkOssart))
        throw std::runtime_error("replay_streaming 当前只支持不含 FDK 初值的 SIRT/SART/OSSART");
    return result;
}

bool writeRaw(const std::filesystem::path& path,
    const std::vector<float>& values)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

bool runCase(const Case& config)
{
    const auto& p = config.params;
    SHeliCTParam h{};
    h.iPU = p.scan.Nu; h.iPV = p.scan.Nv;
    h.du_mm = p.scan.du_mm; h.dv_mm = p.scan.dv_mm;
    h.offsetU_mm = p.scan.offsetU_mm; h.offsetV_mm = p.scan.offsetV_mm;
    h.SID = p.scan.sid_mm; h.SDD = p.scan.sdd_mm;
    h.iVX = p.volume.Nx; h.iVY = p.volume.Ny; h.iVZ = p.volume.Nz;
    h.vox_x_mm = p.volume.voxelX_mm; h.vox_y_mm = p.volume.voxelY_mm;
    h.vox_z_mm = p.volume.voxelZ_mm;
    h.pitch_mm = config.pitch_mm; h.start_z_mm = config.start_z_mm;
    h.views_per_rot = config.views_per_rotation; h.angle_list = p.scan.angles;
    const auto geometry = config.pitch_mm == 0.f
        ? TestGeometry::staticCyl(h, config.curvature_radius_mm,
            config.channel_angle_step_rad)
        : TestGeometry::helicalCyl(h, config.curvature_radius_mm,
            config.channel_angle_step_rad);
    SVolGeom volume_geometry = SVolGeom::make_centered(p.volume.Nx,
        p.volume.Ny, p.volume.Nz, p.volume.voxelX_mm,
        p.volume.voxelY_mm, p.volume.voxelZ_mm);
    const auto phantom = TestPhantom::makeCatphanLike(p);
    const size_t volume_count = phantom.size();

    cudaStream_t stream = nullptr;
    if (cudaSetDevice(config.device) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess) return false;
    bool ok = true;
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> output(volume_count);
    {
        Mem::MemoryController memory;
        auto d_truth = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, config.device);
        auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
            p.volume.Nz, config.device);
        auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv,
            p.scan.NAng, config.device);
        ok = cudaMemcpyAsync(d_truth.data(), phantom.data(),
            volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) == cudaSuccess;
        ResourceContext resources;
        resources.attach(stream, config.device);
        CylFpBp::Config operator_config{};
        operator_config.samples_per_voxel = config.samples_per_voxel;
        auto forward = CylFpBp::makeForwardProjection(
            CylFpBp::EForwardProjection::Joseph);
        ok = ok && forward->prepare(volume_geometry, p.scan.Nu, p.scan.Nv,
            geometry, operator_config, resources) && forward->apply(
                d_truth.data(), d_projection.data(), false, resources);
        forward->release();
        std::vector<float> replay_projection;
        if (ok && config.replay_streaming) {
            replay_projection.resize(static_cast<size_t>(p.scan.Nu) *
                p.scan.Nv * p.scan.NAng);
            ok = cudaStreamSynchronize(stream) == cudaSuccess &&
                cudaMemcpy(replay_projection.data(), d_projection.data(),
                    replay_projection.size() * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
            // 验证重建阶段不依赖完整投影显存。
            d_projection = {};
        }
        ok = ok && cudaMemsetAsync(d_volume.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;
        if (ok && config.pipeline == Pipeline::Fdk) {
            CylFpBp::Analytic::Reconstruction pipeline;
            CylFpBp::Analytic::ReconstructionConfig analytic_config{};
            analytic_config.source_to_detector_mm = p.scan.sdd_mm;
            analytic_config.launch = operator_config.launch;
            ok = pipeline.prepare(volume_geometry, p.scan.Nu, p.scan.Nv,
                geometry, analytic_config, config.filter, stream, config.device) &&
                pipeline.reconstruct(d_projection.data(), d_volume.data(), true);
        }
        else if (ok) {
            const bool use_fdk_initial = config.pipeline == Pipeline::FdkOssart ||
                config.pipeline == Pipeline::FdkCgls;
            if (use_fdk_initial) {
                CylFpBp::Analytic::Reconstruction initializer;
                CylFpBp::Analytic::ReconstructionConfig analytic_config{};
                analytic_config.source_to_detector_mm = p.scan.sdd_mm;
                analytic_config.launch = operator_config.launch;
                ok = initializer.prepare(volume_geometry, p.scan.Nu, p.scan.Nv,
                    geometry, analytic_config, config.filter, stream, config.device) &&
                    initializer.reconstruct(d_projection.data(), d_volume.data(), true);
            }
            Helical::Iterative::CylConfig iterative{};
            iterative.method = config.pipeline == Pipeline::Pwls ?
                Helical::Iterative::EMethod::Pwls :
                config.iterative.method ==
                    CylFpBp::EIterativeMethod::Sirt ?
                Helical::Iterative::EMethod::Sirt :
                config.iterative.method == CylFpBp::EIterativeMethod::Sart ?
                    Helical::Iterative::EMethod::Sart :
                config.iterative.method == CylFpBp::EIterativeMethod::Cgls ?
                    Helical::Iterative::EMethod::Cgls :
                    Helical::Iterative::EMethod::Ossart;
            iterative.algebraic = config.iterative;
            iterative.pwls = config.pwls;
            iterative.operators = operator_config;
            Helical::Iterative::CylReconstructor reconstructor;
            ok = ok && reconstructor.prepare(volume_geometry, p.scan.Nu,
                p.scan.Nv, geometry, iterative, stream, config.device);
            if (ok && config.replay_streaming) {
                const size_t view_elements = static_cast<size_t>(p.scan.Nu) *
                    p.scan.Nv;
                ok = reconstructor.reconstructStreaming(
                    [&](int, int, const std::vector<int>& indices,
                        float* destination, cudaStream_t loader_stream) {
                        for (size_t local = 0; local < indices.size(); ++local) {
                            if (cudaMemcpyAsync(destination + local * view_elements,
                                    replay_projection.data() +
                                        static_cast<size_t>(indices[local]) * view_elements,
                                    view_elements * sizeof(float),
                                    cudaMemcpyHostToDevice, loader_stream) != cudaSuccess)
                                return false;
                        }
                        return true;
                    }, d_volume.data());
            }
            else if (ok && config.projection_batch_views > 0) {
                ok = reconstructor.beginProjectionBatches();
                const size_t view_elements = static_cast<size_t>(p.scan.Nu) *
                    p.scan.Nv;
                for (int first = 0; ok && first < p.scan.NAng;
                     first += config.projection_batch_views) {
                    const int count = std::min(config.projection_batch_views,
                        p.scan.NAng - first);
                    ok = reconstructor.uploadProjectionBatch(
                        d_projection.data() + static_cast<size_t>(first) * view_elements,
                        first, count, true);
                }
                ok = ok && reconstructor.reconstructUploaded(d_volume.data());
            }
            else if (ok) {
                ok = reconstructor.reconstruct(d_projection.data(), d_volume.data());
            }
            if (ok && Iter::convergenceEnabled(config.iterative.convergence)) {
                const auto& stats = reconstructor.convergenceStatistics();
                YK_LOGI("[{}] completed iterations={} convergence checks={}",
                    config.name, stats.completed_iterations,
                    stats.convergence_checks);
            }
        }
        ok = ok && cudaStreamSynchronize(stream) == cudaSuccess &&
            cudaMemcpy(output.data(), d_volume.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
        resources.release();
    }
    cudaStreamDestroy(stream);
    ok = ok && std::all_of(output.begin(), output.end(), [](float value) {
        return std::isfinite(value);
    }) && writeRaw(config.output, output);
    if (ok && !config.preview.empty()) {
        const auto maximum = *std::max_element(output.begin(), output.end());
        const std::vector<TestImage::GrayPanel> panels{{ &output, p.volume.Nx,
            p.volume.Ny, p.volume.Nz, p.volume.Nz / 2, 1.f, 0.f,
            std::max(maximum, 1e-6f), false }};
        ok = TestImage::writeGrayMontageBmp(config.preview, panels, 1, 0, 2);
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    YK_LOGI("[{}] cylindrical reconstruction {}, {:.3f} ms", config.name,
        ok ? "PASS" : "FAIL", elapsed_ms);
    return ok;
}

} // namespace

int runConfiguredCylReconstruction(const std::filesystem::path& file,
    const std::string& case_name)
{
    try {
        const toml::table document = toml::parse_file(file.string());
        if (document["task"].value_or(std::string{}) !=
            "cylindrical-reconstruction")
            throw std::runtime_error("task 必须是 cylindrical-reconstruction");
        const auto* cases = document["case"].as_array();
        if (!cases || cases->empty()) throw std::runtime_error(
            "配置必须至少包含一个 [[case]]");
        const auto base = std::filesystem::absolute(file).parent_path();
        std::unordered_set<std::string> names;
        int selected = 0, failed = 0;
        for (const auto& node : *cases) {
            const auto* table = node.as_table();
            if (!table) throw std::runtime_error("case 必须是 table");
            const Case config = parseCase(*table, base);
            if (!names.insert(config.name).second)
                throw std::runtime_error("case.name 重复: " + config.name);
            if (!case_name.empty() && config.name != case_name) continue;
            ++selected;
            failed += !runCase(config);
        }
        if (!selected) return 2;
        return failed ? 1 : 0;
    }
    catch (const std::exception& error) {
        YK_LOGE("圆柱重建配置执行失败: {}", error.what());
        return 1;
    }
}

} // namespace YK::TestConfig
