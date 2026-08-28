#include "config/YkConfiguredWfbp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <toml.hpp>

#include "global/YkLog.h"

#if YKCBCT_TEST_HAS_HELICAL
#include "CylFpBp/YkCylFpBpGeometry.hpp"
#include "CylFpBp/YkCylForwardProjection.hpp"
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/wfbp/YkWfbpPipeline.hpp"
#include "Iter/YkAlgebraicReconstructorEx.hpp"
#include "Iter/YkPwlsReconstructor.hpp"
#include "YkTestImage.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"
#endif

namespace YK::TestConfig {

#if YKCBCT_TEST_HAS_HELICAL
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

std::filesystem::path resolvePath(const std::filesystem::path& base,
    const std::string& value)
{
    if (value.empty()) return {};
    const std::filesystem::path path(value);
    return path.is_absolute() ? path.lexically_normal() :
        (base / path).lexically_normal();
}

enum class InputMode { ProjectionRaw, VolumeRaw };
enum class Reconstructor { Wfbp, Pwls, Ossart };

struct Case {
    std::string name;
    int device = 0;
    SHeliCTParam params{};
    Helical::Wfbp::Config algorithm{};
    Reconstructor reconstructor = Reconstructor::Wfbp;
    Iter::PwlsConfig pwls{};
    Iter::AlgebraicReconstructionConfig ossart{};
    InputMode input_mode = InputMode::ProjectionRaw;
    std::filesystem::path input;
    std::string scalar_type = "float32";
    float input_scale = 1.f;
    int source_nx = 0;
    int source_ny = 0;
    int source_nz = 0;
    int crop_x = 0;
    int crop_y = 0;
    int crop_z = 0;
    float minimum_correlation = 0.8f;
    float minimum_slice_correlation = 0.8f;
    float maximum_absolute_nrmse = 0.75f;
    float maximum_mae = std::numeric_limits<float>::infinity();
    float maximum_absolute_bias = 0.01f;
    float maximum_material_bias = std::numeric_limits<float>::infinity();
    float maximum_background_mean = std::numeric_limits<float>::infinity();
    std::filesystem::path output;
    std::filesystem::path preview;
};

Helical::Wfbp::EInputDetector parseDetector(const std::string& value)
{
    using Detector = Helical::Wfbp::EInputDetector;
    if (value == "flat-panel") return Detector::FlatPanel;
    if (value == "equiangular-arc") return Detector::EquiangularArc;
    if (value == "cylindrical-arc") return Detector::CylindricalArc;
    throw std::runtime_error("未知 wFBP input_detector: " + value);
}

Helical::Wfbp::EFocalSpotMode parseFocalSpotMode(const std::string& value)
{
    using Mode = Helical::Wfbp::EFocalSpotMode;
    if (value == "none") return Mode::None;
    if (value == "phi") return Mode::Phi;
    if (value == "z") return Mode::Z;
    if (value == "phi-z") return Mode::PhiAndZ;
    throw std::runtime_error("未知 wFBP focal_spot_mode: " + value);
}

Case parseCase(const toml::table& table, const std::filesystem::path& base)
{
    Case result{};
    result.name = required<std::string>(table, "name", "case");
    result.device = static_cast<int>(optional<int64_t>(table, "device", 0));
    const std::string reconstructor = optional<std::string>(table,
        "reconstructor", "wfbp");
    if (reconstructor == "wfbp") result.reconstructor = Reconstructor::Wfbp;
    else if (reconstructor == "pwls") result.reconstructor = Reconstructor::Pwls;
    else if (reconstructor == "ossart") result.reconstructor = Reconstructor::Ossart;
    else throw std::runtime_error("reconstructor 必须是 wfbp、pwls 或 ossart");
    const auto& scan = requiredTable(table, "scan", result.name);
    auto& p = result.params;
    p.iPU = static_cast<int>(required<int64_t>(scan, "nu", "scan"));
    p.iPV = static_cast<int>(required<int64_t>(scan, "nv", "scan"));
    const int views = static_cast<int>(required<int64_t>(scan, "views", "scan"));
    p.du_mm = static_cast<float>(optional<double>(scan, "du_mm", 1.0));
    p.dv_mm = static_cast<float>(optional<double>(scan, "dv_mm", 1.0));
    p.offsetU_mm = static_cast<float>(optional<double>(scan, "offset_u_mm", 0.0));
    p.offsetV_mm = static_cast<float>(optional<double>(scan, "offset_v_mm", 0.0));
    p.sourceOffsetX_mm = static_cast<float>(optional<double>(scan, "source_offset_x_mm", 0.0));
    p.sourceOffsetY_mm = static_cast<float>(optional<double>(scan, "source_offset_y_mm", 0.0));
    p.sourceOffsetZ_mm = static_cast<float>(optional<double>(scan, "source_offset_z_mm", 0.0));
    p.SID = static_cast<float>(required<double>(scan, "sid_mm", "scan"));
    p.SDD = static_cast<float>(required<double>(scan, "sdd_mm", "scan"));
    const float start_angle = static_cast<float>(
        optional<double>(scan, "start_angle_deg", 0.0) * kPi / 180.0);
    const float scan_range = static_cast<float>(
        required<double>(scan, "scan_range_deg", "scan") * kPi / 180.0);

    const auto& volume = requiredTable(table, "volume", result.name);
    p.iVX = static_cast<int>(required<int64_t>(volume, "nx", "volume"));
    p.iVY = static_cast<int>(required<int64_t>(volume, "ny", "volume"));
    p.iVZ = static_cast<int>(required<int64_t>(volume, "nz", "volume"));
    const auto* voxel = volume["voxel_mm"].as_array();
    if (!voxel || voxel->size() != 3)
        throw std::runtime_error("volume.voxel_mm 必须包含 3 个数");
    p.vox_x_mm = static_cast<float>(required<double>(*voxel, 0, "voxel_mm"));
    p.vox_y_mm = static_cast<float>(required<double>(*voxel, 1, "voxel_mm"));
    p.vox_z_mm = static_cast<float>(required<double>(*voxel, 2, "voxel_mm"));
    if (const auto* offset = volume["offset_mm"].as_array()) {
        if (offset->size() != 3)
            throw std::runtime_error("volume.offset_mm 必须包含 3 个数");
        p.vol_offset_x_mm = static_cast<float>(required<double>(*offset, 0, "offset_mm"));
        p.vol_offset_y_mm = static_cast<float>(required<double>(*offset, 1, "offset_mm"));
        p.vol_offset_z_mm = static_cast<float>(required<double>(*offset, 2, "offset_mm"));
    }

    const auto& geometry = requiredTable(table, "geometry", result.name);
    p.pitch_mm = static_cast<float>(required<double>(geometry, "pitch_mm", "geometry"));
    p.start_z_mm = static_cast<float>(optional<double>(geometry, "start_z_mm", 0.0));
    p.views_per_rot = static_cast<int>(required<int64_t>(geometry,
        "views_per_rotation", "geometry"));
    p.angle_list.resize(views);
    for (int i = 0; i < views; ++i)
        p.angle_list[i] = start_angle + scan_range * static_cast<float>(i) / views;

    const auto& wfbp = requiredTable(table, "wfbp", result.name);
    auto& a = result.algorithm;
    a.input_detector = parseDetector(optional<std::string>(wfbp,
        "input_detector", "flat-panel"));
    a.focal_spot_mode = parseFocalSpotMode(optional<std::string>(wfbp,
        "focal_spot_mode", "none"));
    a.redundancy_flat = static_cast<float>(optional<double>(wfbp,
        "redundancy_flat", 0.6));
    a.angle_tolerance = static_cast<float>(optional<double>(wfbp,
        "angle_tolerance", 1e-3));
    a.arc_channel_angle_step_rad = static_cast<float>(optional<double>(wfbp,
        "arc_channel_angle_step_rad", 0.0));
    a.arc_principal_channel = static_cast<float>(optional<double>(wfbp,
        "arc_principal_channel", -1.0));
    a.arc_curvature_radius_mm = static_cast<float>(optional<double>(wfbp,
        "arc_curvature_radius_mm", 0.0));
    a.anode_angle_rad = static_cast<float>(optional<double>(wfbp,
        "anode_angle_deg", 0.0) * kPi / 180.0);
    a.reverse_row_interleave = optional<bool>(wfbp,
        "reverse_row_interleave", false);
    a.filter.cutoff_c = static_cast<float>(optional<double>(wfbp,
        "filter_cutoff", 1.0));
    a.filter.apodization_a = static_cast<float>(optional<double>(wfbp,
        "filter_apodization", 1.0));

    if (const auto* pwls = table["pwls"].as_table()) {
        result.pwls.iterations = static_cast<int>(optional<int64_t>(*pwls,
            "iterations", 5));
        result.pwls.subset_count = static_cast<int>(optional<int64_t>(*pwls,
            "subsets", 1));
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
        const std::string regularizer = optional<std::string>(*pwls,
            "regularizer", "huber");
        if (regularizer == "none")
            result.pwls.regularizer = Iter::EPwlsRegularizer::None;
        else if (regularizer == "quadratic")
            result.pwls.regularizer = Iter::EPwlsRegularizer::Quadratic;
        else if (regularizer == "huber")
            result.pwls.regularizer = Iter::EPwlsRegularizer::Huber;
        else throw std::runtime_error("pwls.regularizer 必须是 none、quadratic 或 huber");
    }

    if (const auto* ossart = table["ossart"].as_table()) {
        auto& o = result.ossart;
        o.method = Iter::EAlgebraicMethod::Ossart;
        o.weight_model = Iter::EAlgebraicWeightModel::DetailedSubset;
        o.iterations = static_cast<int>(optional<int64_t>(*ossart,
            "iterations", 20));
        o.subset_count = static_cast<int>(optional<int64_t>(*ossart,
            "subsets", 20));
        o.relaxation = static_cast<float>(optional<double>(*ossart,
            "relaxation", 0.2));
        o.relaxation_reduction = static_cast<float>(optional<double>(*ossart,
            "relaxation_reduction", 0.98));
        o.epsilon = static_cast<float>(optional<double>(*ossart,
            "epsilon", 1e-6));
        o.use_min = true;
        o.min_constraint = static_cast<float>(optional<double>(*ossart,
            "lower_bound", 0.0));
        o.use_max = true;
        o.max_constraint = static_cast<float>(optional<double>(*ossart,
            "upper_bound", 0.12));
        o.fp_task = ETask::FP_Joseph;
        const std::string back = optional<std::string>(*ossart,
            "back", "joseph-v3");
        if (back == "joseph-v3") o.bp_task = ETask::BP_Joseph_v3;
        else if (back == "joseph") o.bp_task = ETask::BP_Joseph;
        else throw std::runtime_error("ossart.back 必须是 joseph-v3 或 joseph");
    }

    const auto& input = requiredTable(table, "input", result.name);
    const std::string mode = required<std::string>(input, "mode", "input");
    if (mode == "projection-raw") {
        result.input_mode = InputMode::ProjectionRaw;
        result.input = resolvePath(base,
            required<std::string>(input, "projection", "input"));
        result.scalar_type = "float32";
    }
    else if (mode == "volume-raw") {
        result.input_mode = InputMode::VolumeRaw;
        result.input = resolvePath(base,
            required<std::string>(input, "volume", "input"));
        result.scalar_type = optional<std::string>(input, "scalar_type", "float32");
        result.input_scale = static_cast<float>(optional<double>(input, "scale", 1.0));
        result.source_nx = p.iVX;
        result.source_ny = p.iVY;
        result.source_nz = p.iVZ;
        if (const auto* dimensions = input["source_dimensions"].as_array()) {
            if (dimensions->size() != 3)
                throw std::runtime_error("input.source_dimensions 必须包含 3 个整数");
            result.source_nx = static_cast<int>(required<int64_t>(*dimensions, 0,
                "source_dimensions"));
            result.source_ny = static_cast<int>(required<int64_t>(*dimensions, 1,
                "source_dimensions"));
            result.source_nz = static_cast<int>(required<int64_t>(*dimensions, 2,
                "source_dimensions"));
        }
        if (const auto* crop = input["crop_origin"].as_array()) {
            if (crop->size() != 3)
                throw std::runtime_error("input.crop_origin 必须包含 3 个整数");
            result.crop_x = static_cast<int>(required<int64_t>(*crop, 0, "crop_origin"));
            result.crop_y = static_cast<int>(required<int64_t>(*crop, 1, "crop_origin"));
            result.crop_z = static_cast<int>(required<int64_t>(*crop, 2, "crop_origin"));
        }
    }
    else throw std::runtime_error("wFBP input.mode 必须是 projection-raw 或 volume-raw");

    const auto& output = requiredTable(table, "output", result.name);
    result.output = resolvePath(base, required<std::string>(output, "volume", "output"));
    result.preview = resolvePath(base, optional<std::string>(output, "preview", ""));
    if (const auto* validation = table["validation"].as_table()) {
        result.minimum_correlation = static_cast<float>(optional<double>(*validation,
            "minimum_correlation", 0.8));
        result.minimum_slice_correlation = static_cast<float>(optional<double>(*validation,
            "minimum_slice_correlation", 0.8));
        result.maximum_absolute_nrmse = static_cast<float>(optional<double>(*validation,
            "maximum_absolute_nrmse", 0.75));
        result.maximum_mae = static_cast<float>(optional<double>(*validation,
            "maximum_mae", std::numeric_limits<double>::infinity()));
        result.maximum_absolute_bias = static_cast<float>(optional<double>(*validation,
            "maximum_absolute_bias", 0.01));
        result.maximum_material_bias = static_cast<float>(optional<double>(*validation,
            "maximum_material_bias", std::numeric_limits<double>::infinity()));
        result.maximum_background_mean = static_cast<float>(optional<double>(*validation,
            "maximum_background_mean", std::numeric_limits<double>::infinity()));
    }

    const float configured_step = scan_range / views;
    const float expected_step = 2.f * kPi / p.views_per_rot;
    if (result.device < 0 || views < p.views_per_rot || p.iPU < 2 || p.iPV < 2 ||
        p.iVX <= 0 || p.iVY <= 0 || p.iVZ <= 0 || p.du_mm <= 0.f ||
        p.dv_mm <= 0.f || p.SID <= 0.f || p.SDD <= p.SID || p.pitch_mm <= 0.f ||
        p.views_per_rot < 4 || std::fabs(configured_step - expected_step) >
        a.angle_tolerance * expected_step)
        throw std::runtime_error(result.name + " 的尺寸、螺距或角度采样无效");
    if (result.input_mode == InputMode::VolumeRaw &&
        a.focal_spot_mode != Helical::Wfbp::EFocalSpotMode::None)
        throw std::runtime_error(result.name +
            "：volume-raw 自测不生成 FFS 交错焦点，请改用真实 projection-raw");
    if (result.scalar_type != "float32" && result.scalar_type != "uint8")
        throw std::runtime_error("volume-raw scalar_type 仅支持 float32 或 uint8");
    if (!(result.input_scale > 0.f))
        throw std::runtime_error("input.scale 必须大于 0");
    if (result.reconstructor != Reconstructor::Wfbp &&
        a.input_detector != Helical::Wfbp::EInputDetector::FlatPanel)
        throw std::runtime_error("普通迭代重建使用 SConeProjGeomVec，只支持 flat-panel");
    if (result.reconstructor == Reconstructor::Pwls &&
        (result.pwls.iterations <= 0 || result.pwls.subset_count != 1 ||
         result.pwls.relaxation <= 0.f || result.pwls.regularization < 0.f ||
         result.pwls.epsilon <= 0.f || result.pwls.lower_bound > result.pwls.upper_bound))
        throw std::runtime_error(
            "螺旋 PWLS 参数无效；当前定量对照只允许 subsets=1 的全批量更新");
    if (result.reconstructor == Reconstructor::Ossart &&
        (result.ossart.iterations <= 0 || result.ossart.subset_count <= 0 ||
         result.ossart.subset_count > views || result.ossart.relaxation <= 0.f ||
         result.ossart.relaxation_reduction <= 0.f || result.ossart.epsilon <= 0.f ||
         result.ossart.min_constraint > result.ossart.max_constraint))
        throw std::runtime_error("螺旋 OS-SART 参数无效");
    if (result.input_mode == InputMode::VolumeRaw &&
        (result.source_nx <= 0 || result.source_ny <= 0 || result.source_nz <= 0 ||
         result.crop_x < 0 || result.crop_y < 0 || result.crop_z < 0 ||
         result.crop_x + p.iVX > result.source_nx ||
         result.crop_y + p.iVY > result.source_ny ||
         result.crop_z + p.iVZ > result.source_nz))
        throw std::runtime_error(result.name + " 的 volume-raw 裁剪范围无效");
    if (result.minimum_correlation < -1.f || result.minimum_correlation > 1.f ||
        result.minimum_slice_correlation < -1.f ||
        result.minimum_slice_correlation > 1.f ||
        result.maximum_absolute_nrmse <= 0.f ||
        result.maximum_mae < 0.f || result.maximum_absolute_bias < 0.f ||
        result.maximum_material_bias < 0.f || result.maximum_background_mean < 0.f)
        throw std::runtime_error(result.name + " 的质量验证阈值无效");
    return result;
}

std::vector<float> readInput(const Case& config, size_t count)
{
    const bool volume = config.input_mode == InputMode::VolumeRaw;
    const size_t source_count = volume ? static_cast<size_t>(config.source_nx) *
        config.source_ny * config.source_nz : count;
    const size_t scalar_bytes = config.scalar_type == "uint8" ? 1 : sizeof(float);
    std::error_code error;
    const auto actual = std::filesystem::file_size(config.input, error);
    if (error || actual != source_count * scalar_bytes)
        throw std::runtime_error("输入文件大小错误: " + config.input.string());
    std::ifstream stream(config.input, std::ios::binary);
    std::vector<float> source(source_count);
    if (config.scalar_type == "uint8") {
        std::vector<uint8_t> raw(source_count);
        stream.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(raw.size()));
        if (!stream) throw std::runtime_error("无法读取 uint8 模体");
        std::transform(raw.begin(), raw.end(), source.begin(),
            [&](uint8_t value) { return config.input_scale * value; });
    }
    else {
        stream.read(reinterpret_cast<char*>(source.data()),
            static_cast<std::streamsize>(source_count * sizeof(float)));
        if (!stream) throw std::runtime_error("无法读取 float32 数据");
        if (config.input_scale != 1.f)
            for (float& value : source) value *= config.input_scale;
    }
    if (!volume || source_count == count) return source;
    std::vector<float> result(count);
    for (int z = 0; z < config.params.iVZ; ++z) {
        for (int y = 0; y < config.params.iVY; ++y) {
            const size_t source_offset =
                (static_cast<size_t>(z + config.crop_z) * config.source_ny +
                 y + config.crop_y) * config.source_nx + config.crop_x;
            const size_t target_offset =
                (static_cast<size_t>(z) * config.params.iVY + y) * config.params.iVX;
            std::copy_n(source.data() + source_offset, config.params.iVX,
                result.data() + target_offset);
        }
    }
    return result;
}

SReconstructionParams toCbct(const SHeliCTParam& h)
{
    SReconstructionParams p{};
    p.scan.Nu = h.iPU; p.scan.Nv = h.iPV;
    p.scan.NAng = static_cast<int>(h.angle_list.size());
    p.scan.totalViews = p.scan.NAng;
    p.scan.du_mm = h.du_mm; p.scan.dv_mm = h.dv_mm;
    p.scan.offsetU_mm = h.offsetU_mm; p.scan.offsetV_mm = h.offsetV_mm;
    p.scan.sid_mm = h.SID; p.scan.sdd_mm = h.SDD;
    p.scan.sourceOffsetX_mm = h.sourceOffsetX_mm;
    p.scan.sourceOffsetY_mm = h.sourceOffsetY_mm;
    p.scan.sourceOffsetZ_mm = h.sourceOffsetZ_mm;
    p.scan.angles = h.angle_list;
    p.volume.Nx = h.iVX; p.volume.Ny = h.iVY; p.volume.Nz = h.iVZ;
    p.volume.voxelX_mm = h.vox_x_mm;
    p.volume.voxelY_mm = h.vox_y_mm;
    p.volume.voxelZ_mm = h.vox_z_mm;
    p.volume.centerX_mm = h.vol_offset_x_mm;
    p.volume.centerY_mm = h.vol_offset_y_mm;
    p.volume.centerZ_mm = h.vol_offset_z_mm;
    return p;
}

bool writeRaw(const std::filesystem::path& path, const std::vector<float>& values)
{
    std::error_code error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return stream.good();
}

struct QualityMetrics {
    double correlation = 0.0;
    double slice_correlation = 0.0;
    double cosine_similarity = 0.0;
    double absolute_rmse = std::numeric_limits<double>::infinity();
    double absolute_nrmse = std::numeric_limits<double>::infinity();
    double mae = std::numeric_limits<double>::infinity();
    double bias = 0.0;
    double maximum_absolute_error = 0.0;
    double fitted_scale = 0.0;
    double fitted_nrmse = std::numeric_limits<double>::infinity();
    double truth_minimum = std::numeric_limits<double>::infinity();
    double truth_maximum = -std::numeric_limits<double>::infinity();
    double truth_mean = 0.0;
    double reconstruction_minimum = std::numeric_limits<double>::infinity();
    double reconstruction_maximum = -std::numeric_limits<double>::infinity();
    double reconstruction_mean = 0.0;
};

QualityMetrics compareWithTruth(const std::vector<float>& truth,
    const std::vector<float>& reconstruction, int nx, int ny, int nz)
{
    QualityMetrics result{};
    double dot = 0.0, truth_norm = 0.0, recon_norm = 0.0;
    double truth_sum = 0.0, recon_sum = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double a = truth[i];
        const double b = reconstruction[i];
        dot += a * b;
        truth_norm += a * a;
        recon_norm += b * b;
        truth_sum += a;
        recon_sum += b;
        result.truth_minimum = std::min(result.truth_minimum, a);
        result.truth_maximum = std::max(result.truth_maximum, a);
        result.reconstruction_minimum = std::min(result.reconstruction_minimum, b);
        result.reconstruction_maximum = std::max(result.reconstruction_maximum, b);
    }
    result.truth_mean = truth_sum / truth.size();
    result.reconstruction_mean = recon_sum / reconstruction.size();
    result.fitted_scale = recon_norm > 1e-30 ? dot / recon_norm : 0.0;
    result.cosine_similarity = dot /
        std::sqrt(std::max(truth_norm * recon_norm, 1e-30));
    double error_norm = 0.0, fitted_error_norm = 0.0, absolute_error_sum = 0.0;
    double centered_dot = 0.0, centered_truth_norm = 0.0, centered_recon_norm = 0.0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double error = static_cast<double>(reconstruction[i]) - truth[i];
        error_norm += error * error;
        absolute_error_sum += std::fabs(error);
        result.maximum_absolute_error = std::max(
            result.maximum_absolute_error, std::fabs(error));
        const double fitted_error = result.fitted_scale * reconstruction[i] - truth[i];
        fitted_error_norm += fitted_error * fitted_error;
        const double centered_truth = truth[i] - result.truth_mean;
        const double centered_recon = reconstruction[i] - result.reconstruction_mean;
        centered_dot += centered_truth * centered_recon;
        centered_truth_norm += centered_truth * centered_truth;
        centered_recon_norm += centered_recon * centered_recon;
    }
    result.correlation = centered_dot /
        std::sqrt(std::max(centered_truth_norm * centered_recon_norm, 1e-30));
    result.absolute_rmse = std::sqrt(error_norm / truth.size());
    result.absolute_nrmse = std::sqrt(error_norm / std::max(truth_norm, 1e-30));
    result.mae = absolute_error_sum / truth.size();
    result.bias = result.reconstruction_mean - result.truth_mean;
    result.fitted_nrmse = std::sqrt(fitted_error_norm /
        std::max(truth_norm, 1e-30));

    const size_t slice_begin = static_cast<size_t>(nz / 2) * nx * ny;
    double slice_dot = 0.0, slice_truth_norm = 0.0, slice_recon_norm = 0.0;
    for (size_t i = 0; i < static_cast<size_t>(nx) * ny; ++i) {
        const double a = truth[slice_begin + i];
        const double b = reconstruction[slice_begin + i];
        slice_dot += a * b;
        slice_truth_norm += a * a;
        slice_recon_norm += b * b;
    }
    result.slice_correlation = slice_dot /
        std::sqrt(std::max(slice_truth_norm * slice_recon_norm, 1e-30));
    return result;
}

struct MaterialMetrics {
    double maximum_bias = 0.0;
    double maximum_core_bias = 0.0;
    double background_mean = 0.0;
};

MaterialMetrics logMaterialValues(const Case& config, const char* method,
    const std::vector<float>& truth, const std::vector<float>& reconstruction)
{
    MaterialMetrics result{};
    if (config.scalar_type != "uint8" || !(config.input_scale > 0.f)) return result;
    struct Accumulator { size_t count = 0; double sum = 0.0; double sum2 = 0.0; };
    std::array<Accumulator, 256> values{};
    std::array<Accumulator, 256> core_values{};
    std::vector<uint8_t> labels(truth.size());
    for (size_t i = 0; i < truth.size(); ++i) {
        const int label = std::clamp(static_cast<int>(std::lround(
            truth[i] / config.input_scale)), 0, 255);
        labels[i] = static_cast<uint8_t>(label);
        auto& item = values[static_cast<size_t>(label)];
        ++item.count;
        item.sum += reconstruction[i];
        item.sum2 += static_cast<double>(reconstruction[i]) * reconstruction[i];
    }
    const int nx = config.params.iVX;
    const int ny = config.params.iVY;
    const int nz = config.params.iVZ;
    const size_t slice = static_cast<size_t>(nx) * ny;
    // 六邻域均属于同一标签的体素构成一层腐蚀后的核心 ROI。全标签均值
    // 反映包含部分容积效应的实际恢复，核心均值更接近材料内部定量值。
    for (int z = 1; z + 1 < nz; ++z) for (int y = 1; y + 1 < ny; ++y)
        for (int x = 1; x + 1 < nx; ++x) {
            const size_t i = (static_cast<size_t>(z) * ny + y) * nx + x;
            const uint8_t label = labels[i];
            if (labels[i - 1] != label || labels[i + 1] != label ||
                labels[i - nx] != label || labels[i + nx] != label ||
                labels[i - slice] != label || labels[i + slice] != label) continue;
            auto& item = core_values[label];
            ++item.count;
            item.sum += reconstruction[i];
            item.sum2 += static_cast<double>(reconstruction[i]) * reconstruction[i];
        }
    for (size_t label = 0; label < values.size(); ++label) {
        const auto& item = values[label];
        if (item.count == 0) continue;
        const double mean = item.sum / item.count;
        const double standard_deviation = std::sqrt(std::max(
            item.sum2 / item.count - mean * mean, 0.0));
        const double expected = label * config.input_scale;
        const auto& core = core_values[label];
        const double core_mean = core.count > 0 ? core.sum / core.count
                                                 : std::numeric_limits<double>::quiet_NaN();
        result.maximum_bias = std::max(result.maximum_bias,
            std::fabs(mean - expected));
        if (core.count > 0)
            result.maximum_core_bias = std::max(result.maximum_core_bias,
                std::fabs(core_mean - expected));
        if (label == 0) result.background_mean = mean;
        YK_LOGI("[{}:{}] material {}: expected={:.6e} mean={:.6e} "
            "std={:.6e} bias={:.6e} voxels={} core-mean={:.6e} "
            "core-bias={:.6e} core-voxels={}", method, config.name, label,
            expected, mean, standard_deviation, mean - expected, item.count,
            core_mean, core_mean - expected, core.count);
    }
    return result;
}

bool runCase(const Case& config)
{
    const auto& h = config.params;
    const char* method = config.reconstructor == Reconstructor::Wfbp ? "wFBP" :
        config.reconstructor == Reconstructor::Pwls ? "PWLS" : "OS-SART";
    const size_t volume_count = static_cast<size_t>(h.iVX) * h.iVY * h.iVZ;
    const size_t projection_count = static_cast<size_t>(h.iPU) * h.iPV *
        h.angle_list.size();
    YK_CUDA_CHECK(cudaSetDevice(config.device));
    cudaStream_t stream = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&stream));
    bool ok = true;
    std::vector<float> output(volume_count);
    std::vector<float> truth;
    const auto started = std::chrono::steady_clock::now();
    {
        Mem::MemoryController memory;
        auto d_projection = memory.allocateDevice3D<float>(h.iPU, h.iPV,
            static_cast<int>(h.angle_list.size()), config.device);
        auto d_volume = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ,
            config.device);
        if (config.input_mode == InputMode::ProjectionRaw) {
            const auto projection = readInput(config, projection_count);
            ok = cudaMemcpyAsync(d_projection.data(), projection.data(),
                projection_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
                cudaSuccess;
        }
        else {
            truth = readInput(config, volume_count);
            auto d_phantom = memory.allocateDevice3D<float>(h.iVX, h.iVY, h.iVZ,
                config.device);
            ok = cudaMemcpyAsync(d_phantom.data(), truth.data(),
                volume_count * sizeof(float), cudaMemcpyHostToDevice, stream) ==
                cudaSuccess;
            if (ok && config.algorithm.input_detector ==
                Helical::Wfbp::EInputDetector::FlatPanel) {
                const SReconstructionParams p = toCbct(h);
                std::vector<SConeProjGeomVec> geometry;
                build_helical_vec_geometry(geometry, h);
                ForwardOperatorAdapter fp;
                ok = fp.init(p, geometry, ETask::FP_Joseph, config.device, stream) &&
                    fp.run(d_phantom.data(), p, d_projection.data(), stream);
                fp.release();
            }
            else if (ok) {
                const float radius = config.algorithm.input_detector ==
                    Helical::Wfbp::EInputDetector::EquiangularArc ? h.SDD :
                    config.algorithm.arc_curvature_radius_mm;
                const float step = config.algorithm.input_detector ==
                    Helical::Wfbp::EInputDetector::EquiangularArc ?
                    config.algorithm.arc_channel_angle_step_rad : 0.f;
                SHeliCTParam geometry_params = h;
                if (config.algorithm.input_detector ==
                    Helical::Wfbp::EInputDetector::EquiangularArc) {
                    const float principal = config.algorithm.arc_principal_channel >= 0.f
                        ? config.algorithm.arc_principal_channel
                        : 0.5f * (h.iPU - 1);
                    geometry_params.offsetU_mm =
                        (0.5f * (h.iPU - 1) - principal) * h.du_mm;
                }
                auto geometry = CylFpBp::buildCylindricalArcGeometry(
                    geometry_params, radius, step);
                SVolGeom volume_geometry = SVolGeom::make_centered(h.iVX,
                    h.iVY, h.iVZ, h.vox_x_mm, h.vox_y_mm, h.vox_z_mm);
                volume_geometry.center = make_float3(h.vol_offset_x_mm,
                    h.vol_offset_y_mm, h.vol_offset_z_mm);
                // BrainWeb 标签边界包含大量单体素细节。使用两倍射线采样可
                // 将合成投影的离散积分误差与后续 wFBP 误差分开；真实
                // projection-raw 路径不经过此测试专用正投设置。
                CylFpBp::Config fp_config{};
                fp_config.samples_per_voxel = 2.f;
                ResourceContext resources;
                resources.attach(stream, config.device);
                auto fp = CylFpBp::makeForwardProjection(
                    CylFpBp::EForwardProjection::Joseph);
                ok = fp && fp->prepare(volume_geometry, h.iPU, h.iPV, geometry,
                        fp_config, resources) &&
                    fp->apply(d_phantom.data(), d_projection.data(), false,
                        resources);
                if (fp) fp->release();
                resources.release();
            }
        }
        ok = ok && cudaMemsetAsync(d_volume.data(), 0,
            volume_count * sizeof(float), stream) == cudaSuccess;
        if (ok && config.reconstructor == Reconstructor::Wfbp) {
            Helical::Wfbp::Pipeline pipeline;
            ok = pipeline.prepare(h, config.algorithm, stream, config.device) &&
                pipeline.reconstruct(d_projection.data(), d_volume.data());
        }
        else if (ok && config.reconstructor == Reconstructor::Pwls) {
            const SReconstructionParams p = toCbct(h);
            std::vector<SConeProjGeomVec> geometry;
            build_helical_vec_geometry(geometry, h);
            Iter::PwlsReconstructor reconstructor;
            ok = reconstructor.prepare(p, geometry, config.pwls, stream,
                config.device) && reconstructor.reconstruct(
                    d_projection.data(), d_volume.data());
        }
        else if (ok) {
            const SReconstructionParams p = toCbct(h);
            std::vector<SConeProjGeomVec> geometry;
            build_helical_vec_geometry(geometry, h);
            // 使用显式逐视图几何，螺旋床位和任意轨迹不会被圆扫描参数覆盖。
            Iter::AlgebraicReconstructorEx reconstructor;
            ok = reconstructor.prepare(p, geometry, config.ossart, stream,
                config.device) && reconstructor.reconstruct(
                    d_projection.data(), d_volume.data());
        }
        ok = ok && cudaStreamSynchronize(stream) == cudaSuccess &&
            cudaMemcpy(output.data(), d_volume.data(), volume_count * sizeof(float),
                cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaStreamDestroy(stream);

    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    for (float value : output) {
        if (!std::isfinite(value)) ok = false;
        minimum = std::min(minimum, static_cast<double>(value));
        maximum = std::max(maximum, static_cast<double>(value));
    }
    ok = ok && maximum > minimum &&
        std::max(std::fabs(minimum), std::fabs(maximum)) > 1e-8 &&
        writeRaw(config.output, output);
    QualityMetrics metrics{};
    if (!truth.empty()) {
        metrics = compareWithTruth(truth, output, h.iVX, h.iVY, h.iVZ);
        const MaterialMetrics materials = logMaterialValues(
            config, method, truth, output);
        const bool quality_ok = metrics.correlation >= config.minimum_correlation &&
            metrics.slice_correlation >= config.minimum_slice_correlation &&
            metrics.absolute_nrmse <= config.maximum_absolute_nrmse &&
            metrics.mae <= config.maximum_mae &&
            std::fabs(metrics.bias) <= config.maximum_absolute_bias &&
            materials.maximum_bias <= config.maximum_material_bias &&
            std::fabs(materials.background_mean) <= config.maximum_background_mean;
        ok = ok && quality_ok;
        YK_LOGI("[{}:{}] truth min/max/mean={:.6e}/{:.6e}/{:.6e}",
            method, config.name, metrics.truth_minimum, metrics.truth_maximum,
            metrics.truth_mean);
        YK_LOGI("[{}:{}] recon min/max/mean={:.6e}/{:.6e}/{:.6e}",
            method, config.name, metrics.reconstruction_minimum,
            metrics.reconstruction_maximum, metrics.reconstruction_mean);
        YK_LOGI("[{}:{}] abs RMSE={:.6e} NRMSE={:.6f} MAE={:.6e} "
            "bias={:.6e} max-error={:.6e}", method, config.name,
            metrics.absolute_rmse, metrics.absolute_nrmse, metrics.mae,
            metrics.bias, metrics.maximum_absolute_error);
        YK_LOGI("[{}:{}] Pearson={:.6f} cosine={:.6f} slice-cosine={:.6f} "
            "fit-scale={:.6f} fit-NRMSE={:.6f}，{}",
            method, config.name, metrics.correlation, metrics.cosine_similarity,
            metrics.slice_correlation, metrics.fitted_scale,
            metrics.fitted_nrmse, quality_ok ? "质量通过" : "质量不通过");
        YK_LOGI("[{}:{}] material max-bias={:.6e} core max-bias={:.6e} "
            "background-mean={:.6e}", method, config.name,
            materials.maximum_bias, materials.maximum_core_bias,
            materials.background_mean);
        YK_LOGI("[{}:{}] absolute gates: NRMSE {:.6f}<={:.6f}, "
            "MAE {:.6e}<={:.6e}, |bias| {:.6e}<={:.6e}, "
            "material-bias {:.6e}<={:.6e}, |background| {:.6e}<={:.6e}",
            method, config.name, metrics.absolute_nrmse,
            config.maximum_absolute_nrmse,
            metrics.mae, config.maximum_mae, std::fabs(metrics.bias),
            config.maximum_absolute_bias, materials.maximum_bias,
            config.maximum_material_bias, std::fabs(materials.background_mean),
            config.maximum_background_mean);
    }
    if (!config.preview.empty()) {
        std::error_code error;
        std::filesystem::create_directories(config.preview.parent_path(), error);
        const bool preview_directory_ok = !error;
        const float display_max = truth.empty()
            ? static_cast<float>(std::max(std::fabs(minimum), std::fabs(maximum)))
            : static_cast<float>(std::max(metrics.truth_maximum,
                metrics.reconstruction_maximum));
        std::vector<TestImage::GrayPanel> panels;
        if (!truth.empty()) {
            // 真值与重建使用同一窗宽，误差图单独使用最大绝对误差窗宽。
            // 这样既不会用视觉自动缩放掩盖幅值差，也能直接定位空间误差。
            panels.push_back({ &truth, h.iVX, h.iVY, h.iVZ,
                h.iVZ / 2, 1.f, 0.f, display_max, false });
        }
        panels.push_back({ &output, h.iVX, h.iVY, h.iVZ,
            h.iVZ / 2, 1.f, 0.f, display_max, false });
        if (!truth.empty()) {
            std::vector<float> error(volume_count);
            std::transform(output.begin(), output.end(), truth.begin(), error.begin(),
                [](float reconstruction, float reference) {
                    return std::fabs(reconstruction - reference);
                });
            panels.push_back({ &error, h.iVX, h.iVY, h.iVZ, h.iVZ / 2, 1.f,
                0.f, static_cast<float>(std::max(
                    metrics.maximum_absolute_error, 1e-12)), false });
            const bool preview_ok = preview_directory_ok &&
                TestImage::writeGrayMontageBmp(config.preview, panels,
                    static_cast<int>(panels.size()), 2, 2);
            ok = ok && preview_ok;
        }
        else {
            const bool preview_ok = preview_directory_ok &&
                TestImage::writeGrayMontageBmp(config.preview, panels, 1, 0, 2);
            ok = ok && preview_ok;
        }
    }
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    YK_LOGI("[{}:{}] min={:.6e} max={:.6e}，{}，{:.3f} ms，output={}",
        method, config.name, minimum, maximum, ok ? "PASS" : "FAIL", elapsed,
        config.output.string());
    return ok;
}

} // namespace
#endif

int runConfiguredWfbp(const std::filesystem::path& file,
    const std::string& case_name)
{
#if !YKCBCT_TEST_HAS_HELICAL
    (void)file;
    (void)case_name;
    YK_LOGE("wFBP 配置需要使用 -DYKCBCT_BUILD_HELICAL=ON 重新配置并编译 Yktest");
    return 2;
#else
    try {
        const toml::table document = toml::parse_file(file.string());
        const std::string task = document["task"].value_or(std::string{});
        if (task != "wfbp-reconstruction" && task != "helical-reconstruction")
            throw std::runtime_error(
                "螺旋配置的 task 必须是 wfbp-reconstruction 或 helical-reconstruction");
        const auto* cases = document["case"].as_array();
        if (!cases || cases->empty())
            throw std::runtime_error("配置必须至少包含一个 [[case]]");
        const auto base = std::filesystem::absolute(file).parent_path();
        std::unordered_set<std::string> names;
        int selected = 0;
        int failed = 0;
        for (const auto& node : *cases) {
            const auto* table = node.as_table();
            if (!table) throw std::runtime_error("case 数组元素必须是 table");
            const Case config = parseCase(*table, base);
            if (!names.insert(config.name).second)
                throw std::runtime_error("case.name 重复: " + config.name);
            if (!case_name.empty() && config.name != case_name) continue;
            ++selected;
            failed += !runCase(config);
        }
        if (selected == 0) {
            YK_LOGE("找不到配置 case: {}", case_name);
            return 2;
        }
        return failed == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        YK_LOGE("wFBP 配置执行失败: {}", error.what());
        return 1;
    }
#endif
}

} // namespace YK::TestConfig
