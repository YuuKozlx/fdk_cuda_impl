#include "config/YkConfiguredReconstruction.hpp"
#include "config/YkConfiguredForwardProjection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>

#include <cuda_runtime.h>
#include <toml.hpp>

#include "FDK/YkFdkPipeline.hpp"
#include "Iter/YkAlgebraicReconstructorEx.hpp"
#include "Iter/YkCglsReconstructorEx.hpp"
#include "Iter/YkParallelPwlsReconstructor.hpp"
#include "Iter/YkTigreGradientReconstructorEx.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "global/YkLog.h"

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

template <typename T>
T required(const toml::array& array, size_t index, std::string_view context)
{
    if (auto value = array[index].value<T>()) return *value;
    throw std::runtime_error(std::string(context) + "[" +
        std::to_string(index) + "] 类型错误");
}

const toml::table& requiredTable(const toml::table& table,
    std::string_view key, std::string_view context)
{
    const auto* result = table[key].as_table();
    if (!result)
        throw std::runtime_error(std::string(context) + "." + std::string(key) +
            " 必须是 table");
    return *result;
}

Pipeline parsePipeline(const std::string& text)
{
    if (text == "fdk") return Pipeline::Fdk;
    if (text == "sirt") return Pipeline::Sirt;
    if (text == "sart") return Pipeline::Sart;
    if (text == "ossart") return Pipeline::Ossart;
    if (text == "cgls") return Pipeline::Cgls;
    if (text == "tigre-gradient-local") return Pipeline::TigreGradientLocal;
    if (text == "parallel-pwls") return Pipeline::ParallelPwls;
    if (text == "fdk-ossart") return Pipeline::FdkOssart;
    if (text == "fdk-cgls") return Pipeline::FdkCgls;
    throw std::runtime_error("未知 pipeline: " + text);
}

Iter::ETigreGradientAlgorithm parseTigreMethod(const std::string& text)
{
    using Method = Iter::ETigreGradientAlgorithm;
    if (text == "sart") return Method::Sart;
    if (text == "os-sart") return Method::OsSart;
    if (text == "sirt") return Method::Sirt;
    if (text == "asd-pocs") return Method::AsdPocs;
    if (text == "os-asd-pocs") return Method::OsAsdPocs;
    if (text == "b-asd-pocs-beta") return Method::BAsdPocsBeta;
    if (text == "pcsd") return Method::Pcsd;
    if (text == "os-pcsd") return Method::OsPcsd;
    if (text == "aw-pcsd") return Method::AwPcsd;
    if (text == "os-aw-pcsd") return Method::OsAwPcsd;
    if (text == "aw-asd-pocs") return Method::AwAsdPocs;
    if (text == "os-aw-asd-pocs") return Method::OsAwAsdPocs;
    throw std::runtime_error("未知 TIGRE 梯度方法: " + text);
}

Iter::EPwlsRegularizer parsePwlsRegularizer(const std::string& text)
{
    if (text == "none") return Iter::EPwlsRegularizer::None;
    if (text == "quadratic") return Iter::EPwlsRegularizer::Quadratic;
    if (text == "huber") return Iter::EPwlsRegularizer::Huber;
    throw std::runtime_error("未知 PWLS 正则器: " + text);
}

Iter::EAlgebraicWeightModel parseWeightModel(const std::string& text)
{
    if (text == "detailed") return Iter::EAlgebraicWeightModel::DetailedSubset;
    if (text == "tigre-approx") return Iter::EAlgebraicWeightModel::TigreApprox;
    throw std::runtime_error("未知代数权重模型: " + text);
}

Iter::EAlgebraicSubsetOrder parseSubsetOrder(const std::string& text)
{
    if (text == "sequential") return Iter::EAlgebraicSubsetOrder::Sequential;
    if (text == "golden-ratio") return Iter::EAlgebraicSubsetOrder::GoldenRatio;
    throw std::runtime_error("未知子集顺序: " + text);
}

Iter::ECglsStrategy parseCglsStrategy(const std::string& text)
{
    if (text == "robust-restart") return Iter::ECglsStrategy::RobustRestart;
    if (text == "astra-classic") return Iter::ECglsStrategy::AstraClassic;
    throw std::runtime_error("未知 CGLS 策略: " + text);
}

ETask parseForwardProjector(const std::string& text)
{
    if (text == "joseph") return ETask::FP_Joseph;
    if (text == "siddon") return ETask::FP_Siddon;
    throw std::runtime_error("未知正投算子: " + text);
}

ETask parseBackProjector(const std::string& text)
{
    if (text == "joseph") return ETask::BP_Joseph;
    if (text == "joseph-v2") return ETask::BP_Joseph_v2;
    if (text == "joseph-v3") return ETask::BP_Joseph_v3;
    if (text == "siddon-ray") return ETask::BP_Siddon_RayDriven;
    if (text == "siddon-voxel") return ETask::BP_Siddon_VoxDriven;
    throw std::runtime_error("未知反投算子: " + text);
}

EFilterKernel parseFilterKernel(const std::string& text)
{
    if (text == "none") return EFilterKernel::None;
    if (text == "ramlak") return EFilterKernel::RamLak;
    if (text == "shepp-logan") return EFilterKernel::SheppLogan;
    if (text == "cosine") return EFilterKernel::Cosine;
    if (text == "hann") return EFilterKernel::Hann;
    if (text == "hamming") return EFilterKernel::Hamming;
    if (text == "blackman") return EFilterKernel::Blackman;
    if (text == "butterworth") return EFilterKernel::Butterworth;
    if (text == "kaiser") return EFilterKernel::Kaiser;
    if (text == "tukey") return EFilterKernel::Tukey;
    throw std::runtime_error("未知滤波核: " + text);
}

EWeightsBuildSource parseFilterSource(const std::string& text)
{
    if (text == "analytic") return EWeightsBuildSource::AnalyticFreq;
    if (text == "discrete-fft") return EWeightsBuildSource::DiscreteRLFFT;
    throw std::runtime_error("未知滤波权重来源: " + text);
}

std::filesystem::path resolvePath(const std::filesystem::path& base,
    const std::string& text)
{
    if (text.empty()) return {};
    std::filesystem::path path(text);
    return path.is_absolute() ? path.lexically_normal() :
        (base / path).lexically_normal();
}

ReconstructionCase parseCase(const toml::table& table,
    const std::filesystem::path& base)
{
    ReconstructionCase result{};
    result.name = required<std::string>(table, "name", "case");
    result.pipeline = parsePipeline(required<std::string>(table, "pipeline", result.name));
    result.device = static_cast<int>(optional<int64_t>(table, "device", 0));
    result.chunk_views = static_cast<int>(optional<int64_t>(table, "chunk_views", 32));
    result.batch_views = static_cast<int>(optional<int64_t>(table, "batch_views",
        result.chunk_views));

    const auto& scan = requiredTable(table, "scan", result.name);
    auto& p = result.params;
    p.iPU = static_cast<int>(required<int64_t>(scan, "nu", "scan"));
    p.iPV = static_cast<int>(required<int64_t>(scan, "nv", "scan"));
    p.iPAng = static_cast<int>(required<int64_t>(scan, "views", "scan"));
    p.iPAngTotal = p.iPAng;
    p.du_mm = static_cast<float>(optional<double>(scan, "du_mm", 1.0));
    p.dv_mm = static_cast<float>(optional<double>(scan, "dv_mm", 1.0));
    p.offsetU_mm = static_cast<float>(optional<double>(scan, "offset_u_mm", 0.0));
    p.offsetV_mm = static_cast<float>(optional<double>(scan, "offset_v_mm", 0.0));
    p.SID = static_cast<float>(required<double>(scan, "sid_mm", "scan"));
    p.SDD = static_cast<float>(required<double>(scan, "sdd_mm", "scan"));
    p.scan_start_angle_rad = static_cast<float>(
        optional<double>(scan, "start_angle_deg", 0.0) * kPi / 180.0);
    p.scan_range_rad = static_cast<float>(
        optional<double>(scan, "scan_range_deg", 360.0) * kPi / 180.0);
    p.bShortScan = optional<bool>(scan, "short_scan", false);
    p.nDirSign = static_cast<int>(optional<int64_t>(scan, "direction", 1));
    p.angle_list.resize(p.iPAng);
    // scan_range 表示整个扫描覆盖范围；均匀圆轨迹不重复采集终点。
    for (int i = 0; i < p.iPAng; ++i)
        p.angle_list[i] = p.scan_start_angle_rad + p.nDirSign *
            p.scan_range_rad * static_cast<float>(i) / p.iPAng;

    const auto& volume = requiredTable(table, "volume", result.name);
    p.iVX = static_cast<int>(required<int64_t>(volume, "nx", "volume"));
    p.iVY = static_cast<int>(required<int64_t>(volume, "ny", "volume"));
    p.iVZ = static_cast<int>(required<int64_t>(volume, "nz", "volume"));
    const auto* voxel = volume["voxel_mm"].as_array();
    if (!voxel || voxel->size() != 3)
        throw std::runtime_error("volume.voxel_mm 必须包含 3 个数");
    p.vox_x_mm = static_cast<float>(required<double>(*voxel, 0, "volume.voxel_mm"));
    p.vox_y_mm = static_cast<float>(required<double>(*voxel, 1, "volume.voxel_mm"));
    p.vox_z_mm = static_cast<float>(required<double>(*voxel, 2, "volume.voxel_mm"));
    if (const auto* offset = volume["offset_mm"].as_array()) {
        if (offset->size() != 3) throw std::runtime_error("volume.offset_mm 必须包含 3 个数");
        p.vol_offset_x_mm = static_cast<float>(required<double>(*offset, 0, "volume.offset_mm"));
        p.vol_offset_y_mm = static_cast<float>(required<double>(*offset, 1, "volume.offset_mm"));
        p.vol_offset_z_mm = static_cast<float>(required<double>(*offset, 2, "volume.offset_mm"));
    }

    const auto& input = requiredTable(table, "input", result.name);
    const std::string mode = optional<std::string>(input, "mode", "phantom");
    if (mode == "phantom") {
        result.input.mode = InputMode::Phantom;
        result.input.phantom = optional<std::string>(input, "phantom", "catphan");
        if (result.input.phantom != "basic" && result.input.phantom != "catphan")
            throw std::runtime_error("未知内置模体: " + result.input.phantom);
    }
    else if (mode == "projection-raw") {
        result.input.mode = InputMode::ProjectionRaw;
        result.input.projection = resolvePath(base,
            required<std::string>(input, "projection", "input"));
        result.input.air = resolvePath(base,
            optional<std::string>(input, "air", ""));
        result.input.minimum_ratio = static_cast<float>(
            optional<double>(input, "minimum_ratio", 1e-6));
    }
    else throw std::runtime_error("未知 input.mode: " + mode);

    if (const auto* algorithm = table["algorithm"].as_table()) {
        result.algorithm.iterations = static_cast<int>(
            optional<int64_t>(*algorithm, "iterations", 1));
        result.algorithm.subsets = static_cast<int>(
            optional<int64_t>(*algorithm, "subsets", 2));
        result.algorithm.relaxation = static_cast<float>(
            optional<double>(*algorithm, "relaxation", 0.2));
        result.algorithm.relaxation_reduction = static_cast<float>(
            optional<double>(*algorithm, "relaxation_reduction", 1.0));
        result.algorithm.epsilon = static_cast<float>(
            optional<double>(*algorithm, "epsilon", 1e-6));
        result.algorithm.use_min = optional<bool>(*algorithm, "use_min", true);
        result.algorithm.minimum = static_cast<float>(
            optional<double>(*algorithm, "minimum", 0.0));
        result.algorithm.use_max = optional<bool>(*algorithm, "use_max", false);
        result.algorithm.maximum = static_cast<float>(
            optional<double>(*algorithm, "maximum", 1e30));
        result.algorithm.forward_projector = optional<std::string>(*algorithm,
            "forward", "joseph");
        result.algorithm.back_projector = optional<std::string>(*algorithm,
            "back", "joseph-v3");
        result.algorithm.weight_model = optional<std::string>(*algorithm,
            "weight_model", "detailed");
        result.algorithm.subset_order = optional<std::string>(*algorithm,
            "subset_order", "sequential");
        result.algorithm.cgls_strategy = optional<std::string>(*algorithm,
            "strategy", "robust-restart");
        result.algorithm.restart_on_divergence = optional<bool>(*algorithm,
            "restart_on_divergence", true);
        parseForwardProjector(result.algorithm.forward_projector);
        parseBackProjector(result.algorithm.back_projector);

        if (const auto* regularization = (*algorithm)["regularization"].as_table()) {
            result.algorithm.regularizer = optional<std::string>(*regularization,
                "type", "none");
            result.algorithm.regularization_strength = static_cast<float>(
                optional<double>(*regularization, "strength", 0.0));
            result.algorithm.regularization_iterations = static_cast<int>(
                optional<int64_t>(*regularization, "inner_iterations", 5));
            result.algorithm.tv_dimensionality = optional<std::string>(
                *regularization, "dimensionality", "3d");
            result.algorithm.regularization_epsilon = static_cast<float>(
                optional<double>(*regularization, "epsilon", 1e-6));
            result.algorithm.regularization_reduction = static_cast<float>(
                optional<double>(*regularization, "strength_reduction", 1.0));
        }
        if (const auto* convergence = (*algorithm)["convergence"].as_table()) {
            result.algorithm.relative_residual_tolerance = static_cast<float>(
                optional<double>(*convergence, "relative_residual", 0.0));
            result.algorithm.relative_update_tolerance = static_cast<float>(
                optional<double>(*convergence, "relative_update", 0.0));
            result.algorithm.relative_improvement_tolerance = static_cast<float>(
                optional<double>(*convergence, "relative_improvement", 0.0));
            result.algorithm.minimum_iterations = static_cast<int>(
                optional<int64_t>(*convergence, "minimum_iterations", 1));
            result.algorithm.check_interval = static_cast<int>(
                optional<int64_t>(*convergence, "check_interval", 1));
            result.algorithm.patience = static_cast<int>(
                optional<int64_t>(*convergence, "patience", 1));
        }
    }

    if (const auto* tigre = table["tigre"].as_table()) {
        result.tigre.method = optional<std::string>(*tigre, "method", "os-sart");
        result.tigre.block_size = static_cast<int>(
            optional<int64_t>(*tigre, "block_size", 20));
        result.tigre.lambda = static_cast<float>(optional<double>(*tigre, "lambda", 1.0));
        result.tigre.lambda_reduction = static_cast<float>(
            optional<double>(*tigre, "lambda_reduction", 1.0));
        result.tigre.relaxation_mode = optional<std::string>(*tigre,
            "relaxation_mode", "scalar");
        result.tigre.initialization = optional<std::string>(*tigre,
            "initialization", "zero");
        result.tigre.non_negative = optional<bool>(*tigre, "non_negative", true);
        result.tigre.tv_iterations = static_cast<int>(
            optional<int64_t>(*tigre, "tv_iterations", 20));
        result.tigre.alpha = static_cast<float>(optional<double>(*tigre, "alpha", 0.002));
        result.tigre.alpha_reduction = static_cast<float>(
            optional<double>(*tigre, "alpha_reduction", 0.95));
        result.tigre.maximum_update_ratio = static_cast<float>(
            optional<double>(*tigre, "maximum_update_ratio", 0.95));
        result.tigre.max_l2_error = static_cast<float>(
            optional<double>(*tigre, "max_l2_error", -1.0));
        result.tigre.minimum_beta = static_cast<float>(
            optional<double>(*tigre, "minimum_beta", 0.005));
        result.tigre.tv_epsilon = static_cast<float>(
            optional<double>(*tigre, "tv_epsilon", 1e-8));
        result.tigre.adaptive_delta = static_cast<float>(
            optional<double>(*tigre, "adaptive_delta", -0.005));
        result.tigre.bregman_beta = static_cast<float>(
            optional<double>(*tigre, "bregman_beta", 1.0));
        result.tigre.bregman_beta_reduction = static_cast<float>(
            optional<double>(*tigre, "bregman_beta_reduction", 0.75));
        result.tigre.bregman_interval = static_cast<int>(
            optional<int64_t>(*tigre, "bregman_interval", 5));
        parseTigreMethod(result.tigre.method);
    }

    if (const auto* pwls = table["parallel_pwls"].as_table()) {
        result.parallel_pwls.subsets = static_cast<int>(
            optional<int64_t>(*pwls, "subsets", 1));
        result.parallel_pwls.relaxation = static_cast<float>(
            optional<double>(*pwls, "relaxation", 0.8));
        result.parallel_pwls.regularizer = optional<std::string>(*pwls,
            "regularizer", "quadratic");
        result.parallel_pwls.regularization = static_cast<float>(
            optional<double>(*pwls, "regularization", 1e-3));
        result.parallel_pwls.huber_delta = static_cast<float>(
            optional<double>(*pwls, "huber_delta", 3e-3));
        result.parallel_pwls.epsilon = static_cast<float>(
            optional<double>(*pwls, "epsilon", 1e-6));
        result.parallel_pwls.lower_bound = static_cast<float>(
            optional<double>(*pwls, "lower_bound", 0.0));
        result.parallel_pwls.upper_bound = static_cast<float>(
            optional<double>(*pwls, "upper_bound", 1e30));
        parsePwlsRegularizer(result.parallel_pwls.regularizer);
    }

    if (const auto* filter = table["filter"].as_table()) {
        p.desc.kind = parseFilterKernel(optional<std::string>(*filter, "kernel", "ramlak"));
        p.desc.source = parseFilterSource(optional<std::string>(*filter, "source", "discrete-fft"));
        p.desc.cutoff = static_cast<float>(optional<double>(*filter, "cutoff", 0.5));
        p.desc.gain = static_cast<float>(optional<double>(*filter, "gain", 1.0));
        p.desc.order = static_cast<float>(optional<double>(*filter, "order", 2.0));
        p.desc.beta = static_cast<float>(optional<double>(*filter, "beta", 8.6));
        p.desc.tukey_alpha = static_cast<float>(
            optional<double>(*filter, "tukey_alpha", 0.5));
    }
    const auto& output = requiredTable(table, "output", result.name);
    result.output = resolvePath(base,
        required<std::string>(output, "volume", "output"));
    result.preview = resolvePath(base,
        optional<std::string>(output, "preview", ""));

    if (p.iPU <= 0 || p.iPV <= 0 || p.iPAng <= 0 || p.iVX <= 0 || p.iVY <= 0 ||
        p.iVZ <= 0 || p.du_mm <= 0 || p.dv_mm <= 0 || p.vox_x_mm <= 0 ||
        p.vox_y_mm <= 0 || p.vox_z_mm <= 0 || p.SID <= 0 || p.SDD <= p.SID ||
        p.scan_range_rad <= 0 || (p.nDirSign != 1 && p.nDirSign != -1))
        throw std::runtime_error(result.name + " 的尺寸或几何参数无效");
    const bool uses_subsets = result.pipeline == Pipeline::Ossart ||
        result.pipeline == Pipeline::FdkOssart;
    if (result.chunk_views <= 0 || result.batch_views <= 0 ||
        result.input.minimum_ratio <= 0.f || result.input.minimum_ratio > 1.f ||
        result.algorithm.iterations <= 0 ||
        (uses_subsets && (result.algorithm.subsets <= 0 ||
            result.algorithm.subsets > p.iPAng)) ||
        result.algorithm.relaxation <= 0.f ||
        result.algorithm.relaxation_reduction <= 0.f ||
        result.algorithm.epsilon <= 0 ||
        result.algorithm.minimum > result.algorithm.maximum)
        throw std::runtime_error(result.name + " 的执行或迭代参数无效");
    parseWeightModel(result.algorithm.weight_model);
    parseSubsetOrder(result.algorithm.subset_order);
    const auto cgls_strategy = parseCglsStrategy(result.algorithm.cgls_strategy);
    const bool convergence_enabled =
        result.algorithm.relative_residual_tolerance > 0.f ||
        result.algorithm.relative_update_tolerance > 0.f ||
        result.algorithm.relative_improvement_tolerance > 0.f;
    if (result.algorithm.relative_residual_tolerance < 0.f ||
        result.algorithm.relative_update_tolerance < 0.f ||
        result.algorithm.relative_improvement_tolerance < 0.f ||
        result.algorithm.minimum_iterations < 0 ||
        result.algorithm.check_interval <= 0 || result.algorithm.patience <= 0 ||
        result.algorithm.regularization_strength < 0.f ||
        result.algorithm.regularization_iterations <= 0 ||
        result.algorithm.regularization_epsilon <= 0.f ||
        result.algorithm.regularization_reduction <= 0.f ||
        (cgls_strategy == Iter::ECglsStrategy::AstraClassic && convergence_enabled))
        throw std::runtime_error(result.name + " 的正则化或收敛参数无效");
    if (result.pipeline == Pipeline::TigreGradientLocal &&
        (result.tigre.block_size <= 0 || result.tigre.lambda <= 0.f ||
         result.tigre.lambda_reduction <= 0.f || result.tigre.tv_iterations <= 0 ||
         result.tigre.alpha < 0.f || result.tigre.alpha_reduction <= 0.f ||
         result.tigre.maximum_update_ratio <= 0.f || result.tigre.minimum_beta < 0.f ||
         result.tigre.tv_epsilon <= 0.f || result.tigre.bregman_beta <= 0.f ||
         result.tigre.bregman_beta_reduction <= 0.f ||
         result.tigre.bregman_interval <= 0))
        throw std::runtime_error(result.name + " 的 TIGRE 参数无效");
    if (result.pipeline == Pipeline::ParallelPwls &&
        (result.parallel_pwls.subsets <= 0 ||
         result.parallel_pwls.subsets > p.iPAng ||
         result.parallel_pwls.relaxation <= 0.f ||
         result.parallel_pwls.regularization < 0.f ||
         result.parallel_pwls.huber_delta <= 0.f ||
         result.parallel_pwls.epsilon <= 0.f ||
         result.parallel_pwls.lower_bound > result.parallel_pwls.upper_bound))
        throw std::runtime_error(result.name + " 的 PWLS 参数无效");
    if (!(p.desc.cutoff > 0.f && p.desc.cutoff <= 0.5f) ||
        !std::isfinite(p.desc.gain) || p.desc.order <= 0.f || p.desc.beta < 0.f ||
        p.desc.tukey_alpha < 0.f || p.desc.tukey_alpha > 1.f)
        throw std::runtime_error(result.name + " 的滤波参数无效");
    return result;
}

struct ValueStatistics {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    size_t count = 0;
    size_t invalid = 0;

    void add(float value)
    {
        if (!std::isfinite(value)) { ++invalid; return; }
        minimum = std::min(minimum, static_cast<double>(value));
        maximum = std::max(maximum, static_cast<double>(value));
        sum += value;
        ++count;
    }

    bool validNonzero() const
    {
        return invalid == 0 && count > 0 && maximum > minimum &&
            std::max(std::fabs(minimum), std::fabs(maximum)) > 1e-8;
    }
};

class ScopedCudaStream {
public:
    ~ScopedCudaStream()
    {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }
    bool create() { return cudaStreamCreate(&stream_) == cudaSuccess; }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

bool validateFloatFile(const std::filesystem::path& path, size_t element_count,
    const char* label)
{
    const uintmax_t expected = element_count * sizeof(float);
    std::error_code error;
    const uintmax_t actual = std::filesystem::file_size(path, error);
    if (!error && actual == expected) return true;
    YK_LOGE("{}文件大小错误: {}，期望 {} 字节，实际 {} 字节",
        label, path.string(), expected, error ? 0 : actual);
    return false;
}

bool readFloats(std::ifstream& stream, float* destination, size_t count)
{
    const auto bytes = static_cast<std::streamsize>(count * sizeof(float));
    stream.read(reinterpret_cast<char*>(destination), bytes);
    return stream.gcount() == bytes;
}

bool readFloatFile(const std::filesystem::path& path, size_t element_count,
    std::vector<float>& values)
{
    values.resize(element_count);
    std::ifstream stream(path, std::ios::binary);
    return stream && readFloats(stream, values.data(), element_count);
}

bool applyAirCorrection(const std::vector<float>& air, float* projection,
    size_t view_elements, int views, float minimum_ratio, ValueStatistics& stats)
{
    for (int view = 0; view < views; ++view) {
        float* current = projection + static_cast<size_t>(view) * view_elements;
        for (size_t pixel = 0; pixel < view_elements; ++pixel) {
            const float flat = air[pixel];
            const float measured = current[pixel];
            if (!(flat > 0.f) || !std::isfinite(flat) ||
                !(measured >= 0.f) || !std::isfinite(measured)) return false;
            const float value = -std::log(std::clamp(
                measured / flat, minimum_ratio, 1.f));
            current[pixel] = value;
            stats.add(value);
        }
    }
    return true;
}

bool writeVolume(const std::filesystem::path& path,
    const std::vector<float>& volume)
{
    std::error_code error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(volume.data()),
        static_cast<std::streamsize>(volume.size() * sizeof(float)));
    return stream.good();
}

bool writePreview(const ReconstructionCase& config,
    const std::vector<float>& volume, const ValueStatistics& stats)
{
    if (config.preview.empty()) return true;
    std::error_code error;
    std::filesystem::create_directories(config.preview.parent_path(), error);
    if (error) return false;
    const auto& p = config.params;
    const float display_max = static_cast<float>(std::max(
        std::fabs(stats.minimum), std::fabs(stats.maximum)));
    const std::vector<TestImage::GrayPanel> panels{
        { &volume, p.iVX, p.iVY, p.iVZ, p.iVZ / 2,
          1.f, 0.f, display_max, false }
    };
    return TestImage::writeGrayMontageBmp(config.preview, panels, 1, 0, 2);
}

bool runStreamingRawFdk(const ReconstructionCase& config,
    const std::vector<SConeProjGeomVec>& geometry, float* d_volume,
    cudaStream_t stream, Mem::MemoryController& memory)
{
    const auto& p = config.params;
    const size_t view_elements = static_cast<size_t>(p.iPU) * p.iPV;
    std::ifstream projection_input(config.input.projection, std::ios::binary);
    if (!projection_input) return false;

    std::vector<float> air;
    if (!config.input.air.empty() &&
        !readFloatFile(config.input.air, view_elements, air)) return false;

    std::array<Mem::HostPinnedBuffer3D<float>, 2> batches{
        memory.allocatePinnedCpu3D<float>(p.iPU, p.iPV, config.batch_views),
        memory.allocatePinnedCpu3D<float>(p.iPU, p.iPV, config.batch_views)
    };
    std::array<FdkBatchFence, 2> fences;
    FdkPipeline pipeline;
    bool ok = pipeline.prepareWithGeometry(p, geometry, config.chunk_views,
        stream, config.device);
    ValueStatistics projection_stats;

    for (int base = 0; ok && base < p.iPAng; base += config.batch_views) {
        const size_t slot = static_cast<size_t>(
            (base / config.batch_views) % batches.size());
        ok = fences[slot].wait();
        const int count = std::min(config.batch_views, p.iPAng - base);
        const size_t elements = view_elements * static_cast<size_t>(count);
        ok = ok && readFloats(projection_input, batches[slot].data(), elements);
        if (ok && !air.empty())
            ok = applyAirCorrection(air, batches[slot].data(), view_elements,
                count, config.input.minimum_ratio, projection_stats);
        if (!ok) {
            YK_LOGE("[{}] 读取或空气校正失败: base={}, count={}",
                config.name, base, count);
            break;
        }
        const FdkProjectionBatch batch{
            batches[slot].cdata(), nullptr, nullptr, count };
        ok = pipeline.enqueueBatch(batch, d_volume, base == 0, &fences[slot]);
    }
    ok = ok && pipeline.complete();
    for (const auto& fence : fences) ok = fence.wait() && ok;
    if (!air.empty() && projection_stats.count > 0)
        YK_LOGI("[{}] line integral: min={:.6e} max={:.6e} mean={:.6e}",
            config.name, projection_stats.minimum, projection_stats.maximum,
            projection_stats.sum / projection_stats.count);
    return ok;
}

bool runCase(const ReconstructionCase& config)
{
    const auto& p = config.params;
    const size_t volume_count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    const size_t view_elements = static_cast<size_t>(p.iPU) * p.iPV;
    const size_t projection_count = view_elements * p.iPAng;
    std::vector<SConeProjGeomVec> geometry;
    detail::buildCircularViews(p, geometry);

    // IO 配置错误必须在创建 stream 和分配大块显存前报告。
    if (config.input.mode == InputMode::ProjectionRaw &&
        (!validateFloatFile(config.input.projection, projection_count, "投影") ||
         (!config.input.air.empty() &&
          !validateFloatFile(config.input.air, view_elements, "空气场")))) return false;

    if (cudaSetDevice(config.device) != cudaSuccess) return false;
    ScopedCudaStream stream_owner;
    if (!stream_owner.create()) return false;
    const cudaStream_t stream = stream_owner.get();
    bool ok = true;
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> output(volume_count);
    {
        Mem::MemoryController memory;
        auto d_volume = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ,
            config.device);

        if (config.pipeline == Pipeline::Fdk &&
            config.input.mode == InputMode::ProjectionRaw) {
            ok = runStreamingRawFdk(config, geometry, d_volume.data(), stream, memory);
        }
        else {
            auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV,
                p.iPAng, config.device);
            std::vector<float> projection;
            if (config.input.mode == InputMode::ProjectionRaw) {
                ok = readFloatFile(config.input.projection, projection_count, projection);
                if (ok && !config.input.air.empty()) {
                    std::vector<float> air;
                    ValueStatistics unused;
                    ok = readFloatFile(config.input.air, view_elements, air) &&
                        applyAirCorrection(air, projection.data(), view_elements,
                            p.iPAng, config.input.minimum_ratio, unused);
                }
                ok = ok && cudaMemcpyAsync(d_projection.data(), projection.data(),
                    projection_count * sizeof(float), cudaMemcpyHostToDevice,
                    stream) == cudaSuccess;
            }
            else {
                const std::vector<float> phantom = config.input.phantom == "basic" ?
                    TestPhantom::makeBasic(p) : TestPhantom::makeCatphanLike(p);
                auto d_truth = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ,
                    config.device);
                ok = cudaMemcpyAsync(d_truth.data(), phantom.data(),
                    volume_count * sizeof(float), cudaMemcpyHostToDevice,
                    stream) == cudaSuccess;
                GeometryContext geometry_context;
                ResourceContext resources;
                ok = ok && geometry_context.initialize(p);
                resources.attach(stream, config.device);
                auto projector = makeForwardOperator(
                    parseForwardProjector(config.algorithm.forward_projector));
                ok = ok && projector->prepare(geometry_context, resources) &&
                    projector->apply(d_truth.data(), p, d_projection.data(), resources);
                projector->release();
            }
            ok = ok && cudaMemsetAsync(d_volume.data(), 0,
                volume_count * sizeof(float), stream) == cudaSuccess;

            const bool needs_fdk_initial = config.pipeline == Pipeline::Fdk ||
                config.pipeline == Pipeline::FdkOssart ||
                config.pipeline == Pipeline::FdkCgls;
            if (ok && needs_fdk_initial) {
                projection.resize(projection_count);
                ok = cudaMemcpyAsync(projection.data(), d_projection.data(),
                    projection_count * sizeof(float), cudaMemcpyDeviceToHost,
                    stream) == cudaSuccess && cudaStreamSynchronize(stream) == cudaSuccess;
                FdkPipeline pipeline;
                const FdkProjectionBatch batch{
                    projection.data(), nullptr, nullptr, p.iPAng };
                ok = ok && pipeline.prepareWithGeometry(p, geometry,
                    config.chunk_views, stream, config.device) &&
                    pipeline.processBatchSync(batch, d_volume.data(), true) &&
                    pipeline.complete();
            }
            if (ok && (config.pipeline == Pipeline::Cgls ||
                config.pipeline == Pipeline::FdkCgls)) {
                Iter::CglsReconstructionConfig algorithm{};
                algorithm.strategy = parseCglsStrategy(
                    config.algorithm.cgls_strategy);
                algorithm.iterations = config.algorithm.iterations;
                algorithm.epsilon = config.algorithm.epsilon;
                algorithm.restart_on_divergence =
                    config.algorithm.restart_on_divergence;
                algorithm.use_min = config.algorithm.use_min;
                algorithm.min_constraint = config.algorithm.minimum;
                algorithm.use_max = config.algorithm.use_max;
                algorithm.max_constraint = config.algorithm.maximum;
                algorithm.fp_task = parseForwardProjector(
                    config.algorithm.forward_projector);
                algorithm.bp_task = parseBackProjector(
                    config.algorithm.back_projector);
                algorithm.convergence.relative_residual_tolerance =
                    config.algorithm.relative_residual_tolerance;
                algorithm.convergence.relative_update_tolerance =
                    config.algorithm.relative_update_tolerance;
                algorithm.convergence.relative_improvement_tolerance =
                    config.algorithm.relative_improvement_tolerance;
                algorithm.convergence.minimum_iterations =
                    config.algorithm.minimum_iterations;
                algorithm.convergence.check_interval =
                    config.algorithm.check_interval;
                algorithm.convergence.patience = config.algorithm.patience;
                Iter::CglsReconstructorEx reconstructor;
                ok = reconstructor.prepare(p, geometry, algorithm, stream,
                    config.device) && reconstructor.reconstruct(
                        d_projection.data(), d_volume.data());
            }
            else if (ok && (config.pipeline == Pipeline::Sirt ||
                config.pipeline == Pipeline::Sart ||
                config.pipeline == Pipeline::Ossart ||
                config.pipeline == Pipeline::FdkOssart)) {
                Iter::AlgebraicReconstructionConfig algorithm{};
                algorithm.method = config.pipeline == Pipeline::Sirt ?
                    Iter::EAlgebraicMethod::Sirt : config.pipeline == Pipeline::Sart ?
                    Iter::EAlgebraicMethod::Sart : Iter::EAlgebraicMethod::Ossart;
                algorithm.iterations = config.algorithm.iterations;
                algorithm.subset_count = config.pipeline == Pipeline::Sirt ? 1 :
                    config.pipeline == Pipeline::Sart ? p.iPAng :
                    config.algorithm.subsets;
                algorithm.weight_model = parseWeightModel(
                    config.algorithm.weight_model);
                algorithm.subset_order = parseSubsetOrder(
                    config.algorithm.subset_order);
                algorithm.relaxation = config.algorithm.relaxation;
                algorithm.relaxation_reduction =
                    config.algorithm.relaxation_reduction;
                algorithm.epsilon = config.algorithm.epsilon;
                algorithm.use_min = config.algorithm.use_min;
                algorithm.min_constraint = config.algorithm.minimum;
                algorithm.use_max = config.algorithm.use_max;
                algorithm.max_constraint = config.algorithm.maximum;
                algorithm.fp_task = parseForwardProjector(
                    config.algorithm.forward_projector);
                algorithm.bp_task = parseBackProjector(
                    config.algorithm.back_projector);
                if (config.algorithm.regularizer == "smoothed-tv")
                    algorithm.regularization.type =
                        Iter::EAlgebraicRegularizer::SmoothedTv;
                else if (config.algorithm.regularizer != "none")
                    throw std::runtime_error("未知代数正则器: " +
                        config.algorithm.regularizer);
                algorithm.regularization.strength =
                    config.algorithm.regularization_strength;
                algorithm.regularization.inner_iterations =
                    config.algorithm.regularization_iterations;
                if (config.algorithm.tv_dimensionality == "2d")
                    algorithm.regularization.tv_dimensionality =
                        Iter::ETvDimensionality::Slice2D;
                else if (config.algorithm.tv_dimensionality == "3d")
                    algorithm.regularization.tv_dimensionality =
                        Iter::ETvDimensionality::Volume3D;
                else throw std::runtime_error("TV dimensionality 必须是 2d 或 3d");
                algorithm.regularization.epsilon =
                    config.algorithm.regularization_epsilon;
                algorithm.regularization.strength_reduction =
                    config.algorithm.regularization_reduction;
                algorithm.convergence.relative_residual_tolerance =
                    config.algorithm.relative_residual_tolerance;
                algorithm.convergence.relative_update_tolerance =
                    config.algorithm.relative_update_tolerance;
                algorithm.convergence.relative_improvement_tolerance =
                    config.algorithm.relative_improvement_tolerance;
                algorithm.convergence.minimum_iterations =
                    config.algorithm.minimum_iterations;
                algorithm.convergence.check_interval =
                    config.algorithm.check_interval;
                algorithm.convergence.patience = config.algorithm.patience;
                Iter::AlgebraicReconstructorEx reconstructor;
                ok = reconstructor.prepare(p, geometry, algorithm, stream,
                    config.device) && reconstructor.reconstruct(
                        d_projection.data(), d_volume.data());
            }
            else if (ok && config.pipeline == Pipeline::TigreGradientLocal) {
                Iter::TigreGradientConfig algorithm{};
                algorithm.algorithm = parseTigreMethod(config.tigre.method);
                algorithm.iterations = config.algorithm.iterations;
                algorithm.block_size = config.tigre.block_size;
                algorithm.lambda = config.tigre.lambda;
                algorithm.lambda_reduction = config.tigre.lambda_reduction;
                if (config.tigre.relaxation_mode == "scalar")
                    algorithm.relaxation_mode = Iter::ETigreRelaxationMode::Scalar;
                else if (config.tigre.relaxation_mode == "nesterov")
                    algorithm.relaxation_mode = Iter::ETigreRelaxationMode::Nesterov;
                else throw std::runtime_error(
                    "TIGRE relaxation_mode 必须是 scalar 或 nesterov");
                if (config.tigre.initialization == "zero")
                    algorithm.initialization = Iter::ETigreInitialization::Zero;
                else if (config.tigre.initialization == "fdk")
                    algorithm.initialization = Iter::ETigreInitialization::Fdk;
                else throw std::runtime_error(
                    "TIGRE initialization 必须是 zero 或 fdk");
                algorithm.non_negative = config.tigre.non_negative;
                algorithm.tv_iterations = config.tigre.tv_iterations;
                algorithm.alpha = config.tigre.alpha;
                algorithm.alpha_reduction = config.tigre.alpha_reduction;
                algorithm.maximum_update_ratio = config.tigre.maximum_update_ratio;
                algorithm.max_l2_error = config.tigre.max_l2_error;
                algorithm.minimum_beta = config.tigre.minimum_beta;
                algorithm.tv_epsilon = config.tigre.tv_epsilon;
                algorithm.adaptive_delta = config.tigre.adaptive_delta;
                algorithm.bregman_beta = config.tigre.bregman_beta;
                algorithm.bregman_beta_reduction =
                    config.tigre.bregman_beta_reduction;
                algorithm.bregman_interval = config.tigre.bregman_interval;
                algorithm.fp_task = parseForwardProjector(
                    config.algorithm.forward_projector);
                algorithm.bp_task = parseBackProjector(
                    config.algorithm.back_projector);
                Iter::TigreGradientReconstructorEx reconstructor;
                ok = reconstructor.prepare(p, geometry, algorithm, stream,
                    config.device) && reconstructor.reconstruct(
                        d_projection.data(), d_volume.data());
            }
            else if (ok && config.pipeline == Pipeline::ParallelPwls) {
                Iter::ParallelPwlsConfig algorithm{};
                algorithm.iterations = config.algorithm.iterations;
                algorithm.subset_count = config.parallel_pwls.subsets;
                algorithm.relaxation = config.parallel_pwls.relaxation;
                algorithm.regularizer = parsePwlsRegularizer(
                    config.parallel_pwls.regularizer);
                algorithm.regularization = config.parallel_pwls.regularization;
                algorithm.huber_delta = config.parallel_pwls.huber_delta;
                algorithm.epsilon = config.parallel_pwls.epsilon;
                algorithm.lower_bound = config.parallel_pwls.lower_bound;
                algorithm.upper_bound = config.parallel_pwls.upper_bound;
                algorithm.fp_task = parseForwardProjector(
                    config.algorithm.forward_projector);
                algorithm.bp_task = parseBackProjector(
                    config.algorithm.back_projector);
                Iter::ParallelPwlsReconstructor reconstructor;
                ok = reconstructor.prepare(p, geometry, algorithm, stream,
                    config.device) && reconstructor.reconstruct(
                        d_projection.data(), d_volume.data());
            }
        }

        ok = ok && cudaStreamSynchronize(stream) == cudaSuccess &&
            cudaMemcpy(output.data(), d_volume.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    ValueStatistics volume_stats;
    for (float value : output) volume_stats.add(value);
    ok = ok && volume_stats.validNonzero() && writeVolume(config.output, output) &&
        writePreview(config, output, volume_stats);
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (volume_stats.count > 0)
        YK_LOGI("[{}] volume: min={:.6e} max={:.6e} mean={:.6e} invalid={}",
            config.name, volume_stats.minimum, volume_stats.maximum,
            volume_stats.sum / volume_stats.count, volume_stats.invalid);
    YK_LOGI("[{}] {}, {:.3f} ms, output={}", config.name,
        ok ? "PASS" : "FAIL", milliseconds, config.output.string());
    return ok;
}

} // namespace

std::vector<ReconstructionCase> loadCases(const std::filesystem::path& file)
{
    const toml::table document = toml::parse_file(file.string());
    const auto* cases = document["case"].as_array();
    if (!cases || cases->empty())
        throw std::runtime_error("配置必须至少包含一个 [[case]]");
    std::vector<ReconstructionCase> result;
    std::unordered_set<std::string> names;
    for (size_t i = 0; i < cases->size(); ++i) {
        const auto* table = (*cases)[i].as_table();
        if (!table) throw std::runtime_error("case[" + std::to_string(i) + "] 必须是 table");
        auto item = parseCase(*table, std::filesystem::absolute(file).parent_path());
        if (!names.insert(item.name).second)
            throw std::runtime_error("重复的 case 名称: " + item.name);
        result.push_back(std::move(item));
    }
    return result;
}

int runConfiguredReconstruction(const std::filesystem::path& file,
    const std::string& case_name)
{
    try {
        const toml::table document = toml::parse_file(file.string());
        const std::string task = document["task"].value_or(std::string("reconstruction"));
        if (task == "forward-projection")
            return runConfiguredForwardProjection(file, case_name);
        if (task != "reconstruction") {
            YK_LOGE("未知配置任务类型: {}", task);
            return 2;
        }
        const auto cases = loadCases(file);
        int selected = 0;
        int failed = 0;
        for (const auto& item : cases) {
            if (!case_name.empty() && item.name != case_name) continue;
            ++selected;
            failed += !runCase(item);
        }
        if (selected == 0) {
            YK_LOGE("配置中没有名为 '{}' 的 case", case_name);
            return 2;
        }
        YK_LOGI("Configured summary: selected={} passed={} failed={}",
            selected, selected - failed, failed);
        return failed == 0 ? 0 : 1;
    }
    catch (const toml::parse_error& error) {
        YK_LOGE("TOML 解析失败: {}", error.what());
    }
    catch (const std::exception& error) {
        YK_LOGE("配置无效: {}", error.what());
    }
    return 2;
}

} // namespace YK::TestConfig
