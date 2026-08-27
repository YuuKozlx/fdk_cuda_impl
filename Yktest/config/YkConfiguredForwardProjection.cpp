#include "config/YkConfiguredForwardProjection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <toml.hpp>

#include "YKCBCT/geometry/YkProjectionGeometryBuilders.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkMem3d.hpp"
#include "global/YkLog.h"
#include "CylFpBp/YkCylFpBpGeometry.hpp"
#include "CylFpBp/YkCylFpBpOperator.hpp"
#include "Heli/YkHelicalGeo.hpp"

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
T arrayValue(const toml::array& array, size_t index,
    std::string_view context)
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

enum class GeometryModel { Circular, Helical, Planar, PlanarEllipse, Cylindrical };
enum class ProjectorModel { Joseph, Siddon, CylindricalJoseph };
enum class InputModel { Phantom, VolumeRaw };

struct ForwardCase {
    std::string name;
    int device = 0;
    SCBCTParams params{};
    GeometryModel geometry = GeometryModel::Circular;
    ProjectorModel projector = ProjectorModel::Joseph;
    InputModel input = InputModel::Phantom;
    std::string phantom = "catphan";
    std::filesystem::path volume_input;
    std::filesystem::path projection_output;
    std::filesystem::path preview_output;
    int preview_view = 0;

    float pitch_mm = 0.f;
    float start_z_mm = 0.f;
    int views_per_rotation = 0;
    float source_lateral_mm = 0.f;
    float source_axis_x_mm = 0.f;
    float source_axis_z_mm = 0.f;
    float curvature_radius_mm = 0.f;
    float channel_angle_step_rad = 0.f;
    float samples_per_voxel = 1.f;
};

GeometryModel parseGeometry(const std::string& value)
{
    if (value == "circular") return GeometryModel::Circular;
    if (value == "helical") return GeometryModel::Helical;
    if (value == "planar") return GeometryModel::Planar;
    if (value == "planar-ellipse") return GeometryModel::PlanarEllipse;
    if (value == "cylindrical") return GeometryModel::Cylindrical;
    throw std::runtime_error("未知 geometry.model: " + value);
}

ProjectorModel parseProjector(const std::string& value)
{
    if (value == "joseph") return ProjectorModel::Joseph;
    if (value == "siddon") return ProjectorModel::Siddon;
    if (value == "cylindrical-joseph") return ProjectorModel::CylindricalJoseph;
    throw std::runtime_error("未知 projector.model: " + value);
}

ForwardCase parseCase(const toml::table& table, const std::filesystem::path& base)
{
    ForwardCase result{};
    result.name = required<std::string>(table, "name", "case");
    result.device = static_cast<int>(optional<int64_t>(table, "device", 0));

    auto& p = result.params;
    const auto& scan = requiredTable(table, "scan", result.name);
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
    p.nDirSign = static_cast<int>(optional<int64_t>(scan, "direction", 1));
    p.tiltu_angle_rad = static_cast<float>(
        optional<double>(scan, "tilt_u_deg", 0.0) * kPi / 180.0);
    p.tiltn_angle_rad = static_cast<float>(
        optional<double>(scan, "tilt_n_deg", 0.0) * kPi / 180.0);
    p.tiltv_angle_rad = static_cast<float>(
        optional<double>(scan, "tilt_v_deg", 0.0) * kPi / 180.0);
    p.angle_list.resize(std::max(p.iPAng, 0));
    for (int i = 0; i < p.iPAng; ++i) {
        // scan_range 是采集覆盖范围，均匀扫描不重复采集范围终点。
        p.angle_list[i] = p.scan_start_angle_rad + p.nDirSign *
            p.scan_range_rad * static_cast<float>(i) / p.iPAng;
    }

    const auto& volume = requiredTable(table, "volume", result.name);
    p.iVX = static_cast<int>(required<int64_t>(volume, "nx", "volume"));
    p.iVY = static_cast<int>(required<int64_t>(volume, "ny", "volume"));
    p.iVZ = static_cast<int>(required<int64_t>(volume, "nz", "volume"));
    const auto* voxel = volume["voxel_mm"].as_array();
    if (!voxel || voxel->size() != 3)
        throw std::runtime_error("volume.voxel_mm 必须包含 3 个数");
    p.vox_x_mm = static_cast<float>(arrayValue<double>(*voxel, 0, "volume.voxel_mm"));
    p.vox_y_mm = static_cast<float>(arrayValue<double>(*voxel, 1, "volume.voxel_mm"));
    p.vox_z_mm = static_cast<float>(arrayValue<double>(*voxel, 2, "volume.voxel_mm"));
    if (const auto* offset = volume["offset_mm"].as_array()) {
        if (offset->size() != 3)
            throw std::runtime_error("volume.offset_mm 必须包含 3 个数");
        p.vol_offset_x_mm = static_cast<float>(arrayValue<double>(*offset, 0, "volume.offset_mm"));
        p.vol_offset_y_mm = static_cast<float>(arrayValue<double>(*offset, 1, "volume.offset_mm"));
        p.vol_offset_z_mm = static_cast<float>(arrayValue<double>(*offset, 2, "volume.offset_mm"));
    }

    const auto& geometry = requiredTable(table, "geometry", result.name);
    result.geometry = parseGeometry(required<std::string>(geometry, "model", "geometry"));
    result.pitch_mm = static_cast<float>(optional<double>(geometry, "pitch_mm", 0.0));
    result.start_z_mm = static_cast<float>(optional<double>(geometry, "start_z_mm", 0.0));
    result.views_per_rotation = static_cast<int>(optional<int64_t>(geometry,
        "views_per_rotation", p.iPAng));
    result.source_lateral_mm = static_cast<float>(optional<double>(geometry,
        "source_lateral_mm", 0.0));
    result.source_axis_x_mm = static_cast<float>(optional<double>(geometry,
        "source_axis_x_mm", 0.0));
    result.source_axis_z_mm = static_cast<float>(optional<double>(geometry,
        "source_axis_z_mm", 0.0));
    result.curvature_radius_mm = static_cast<float>(optional<double>(geometry,
        "curvature_radius_mm", p.SDD));
    result.channel_angle_step_rad = static_cast<float>(optional<double>(geometry,
        "channel_angle_step_rad", 0.0));

    const auto& projector = requiredTable(table, "projector", result.name);
    result.projector = parseProjector(required<std::string>(projector, "model", "projector"));
    result.samples_per_voxel = static_cast<float>(optional<double>(projector,
        "samples_per_voxel", 1.0));

    const auto& input = requiredTable(table, "input", result.name);
    const std::string input_mode = optional<std::string>(input, "mode", "phantom");
    if (input_mode == "phantom") {
        result.input = InputModel::Phantom;
        result.phantom = optional<std::string>(input, "phantom", "catphan");
        if (result.phantom != "basic" && result.phantom != "catphan" &&
            result.phantom != "arrow")
            throw std::runtime_error("未知内置模体: " + result.phantom);
    } else if (input_mode == "volume-raw") {
        result.input = InputModel::VolumeRaw;
        result.volume_input = resolvePath(base,
            required<std::string>(input, "volume", "input"));
    } else {
        throw std::runtime_error("未知 input.mode: " + input_mode);
    }

    const auto& output = requiredTable(table, "output", result.name);
    result.projection_output = resolvePath(base,
        required<std::string>(output, "projection", "output"));
    result.preview_output = resolvePath(base,
        optional<std::string>(output, "preview", ""));
    result.preview_view = static_cast<int>(optional<int64_t>(output,
        "preview_view", 0));

    if (result.device < 0 || p.iPU <= 0 || p.iPV <= 0 || p.iPAng <= 0 ||
        p.iVX <= 0 || p.iVY <= 0 || p.iVZ <= 0 || p.du_mm <= 0.f ||
        p.dv_mm <= 0.f || p.vox_x_mm <= 0.f || p.vox_y_mm <= 0.f ||
        p.vox_z_mm <= 0.f || p.SID <= 0.f || p.SDD <= p.SID ||
        p.scan_range_rad <= 0.f || (p.nDirSign != 1 && p.nDirSign != -1))
        throw std::runtime_error(result.name + " 的尺寸或扫描参数无效");
    if (result.preview_view < 0 || result.preview_view >= p.iPAng)
        throw std::runtime_error(result.name + " 的 output.preview_view 超出视图范围");

    const bool cylindrical = result.geometry == GeometryModel::Cylindrical;
    if (cylindrical != (result.projector == ProjectorModel::CylindricalJoseph))
        throw std::runtime_error(result.name +
            "：cylindrical 几何必须与 cylindrical-joseph 算子配对");
    if ((result.geometry == GeometryModel::Helical || cylindrical) &&
        (result.views_per_rotation <= 0 || !std::isfinite(result.pitch_mm)))
        throw std::runtime_error(result.name + " 的螺旋参数无效");
    if (result.geometry == GeometryModel::Helical || cylindrical) {
        // 角度只由 scan 表生成；views_per_rotation 只校验采样密度，避免它
        // 悄悄成为第二份角度来源。允许少量浮点舍入误差。
        const float configured_step = p.scan_range_rad / p.iPAng;
        const float expected_step = 2.f * kPi / result.views_per_rotation;
        if (std::fabs(configured_step - expected_step) >
            1e-5f * std::max(configured_step, expected_step))
            throw std::runtime_error(result.name +
                "：scan 角步长与 geometry.views_per_rotation 不一致");
    }
    if (cylindrical && (!(result.curvature_radius_mm > 0.f) ||
        result.channel_angle_step_rad < 0.f || result.samples_per_voxel <= 0.f))
        throw std::runtime_error(result.name + " 的圆柱探测器参数无效");
    if (result.geometry == GeometryModel::Planar &&
        !std::isfinite(result.source_lateral_mm))
        throw std::runtime_error(result.name + " 的 source_lateral_mm 无效");
    if (result.geometry == GeometryModel::PlanarEllipse &&
        (!(result.source_axis_x_mm > 0.f) || !(result.source_axis_z_mm > 0.f)))
        throw std::runtime_error(result.name + " 的平面椭圆源轨迹轴长必须大于 0");
    if ((result.geometry == GeometryModel::Planar ||
         result.geometry == GeometryModel::PlanarEllipse) &&
        (p.offsetU_mm != 0.f || p.offsetV_mm != 0.f ||
         p.tiltu_angle_rad != 0.f || p.tiltn_angle_rad != 0.f ||
         p.tiltv_angle_rad != 0.f))
        throw std::runtime_error(result.name +
            "：当前平面 CT builder 不支持探测器 offset/tilt，不能静默忽略这些参数");
    return result;
}

std::vector<float> readRawVolume(const std::filesystem::path& path, size_t count)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    const size_t expected = count * sizeof(float);
    if (error || bytes != expected)
        throw std::runtime_error("volume raw 字节数错误: " + path.string() +
            "，期望 " + std::to_string(expected) + " 字节");
    std::vector<float> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input || !input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected)))
        throw std::runtime_error("无法读取 volume raw: " + path.string());
    return values;
}

std::vector<float> makeInputVolume(const ForwardCase& config)
{
    const auto& p = config.params;
    const size_t count = static_cast<size_t>(p.iVX) * p.iVY * p.iVZ;
    if (config.input == InputModel::VolumeRaw)
        return readRawVolume(config.volume_input, count);
    if (config.phantom == "basic") return TestPhantom::makeBasic(p);
    if (config.phantom == "arrow") return TestPhantom::makeArrowDirections(p);
    return TestPhantom::makeCatphanLike(p);
}

SHeliCTParam makeHelicalParams(const ForwardCase& config)
{
    const auto& p = config.params;
    SHeliCTParam h{};
    h.iPU = p.iPU; h.iPV = p.iPV;
    h.du_mm = p.du_mm; h.dv_mm = p.dv_mm;
    h.offsetU_mm = p.offsetU_mm; h.offsetV_mm = p.offsetV_mm;
    h.SID = p.SID; h.SDD = p.SDD;
    h.iVX = p.iVX; h.iVY = p.iVY; h.iVZ = p.iVZ;
    h.vox_x_mm = p.vox_x_mm; h.vox_y_mm = p.vox_y_mm;
    h.vox_z_mm = p.vox_z_mm;
    h.vol_offset_x_mm = p.vol_offset_x_mm;
    h.vol_offset_y_mm = p.vol_offset_y_mm;
    h.vol_offset_z_mm = p.vol_offset_z_mm;
    h.tiltu_angle_rad = p.tiltu_angle_rad;
    h.tiltn_angle_rad = p.tiltn_angle_rad;
    h.tiltv_angle_rad = p.tiltv_angle_rad;
    h.pitch_mm = config.pitch_mm;
    h.start_z_mm = config.start_z_mm;
    h.views_per_rot = config.views_per_rotation;
    h.angle_list = p.angle_list;
    return h;
}

std::vector<SConeProjGeomVec> buildPlanarGeometry(const ForwardCase& config)
{
    const auto& p = config.params;
    std::vector<SConeProjGeomVec> geometry;
    if (config.geometry == GeometryModel::Circular) {
        detail::buildCircularViews(p, geometry);
    } else if (config.geometry == GeometryModel::Planar) {
        build_planar_ct_vec_geometry(geometry, p.angle_list, p.iPAng,
            p.iPU, p.iPV, p.du_mm, p.dv_mm, p.SID, p.SDD - p.SID,
            config.source_lateral_mm);
    } else if (config.geometry == GeometryModel::PlanarEllipse) {
        build_planar_ct_vec_geometry_ellipse(geometry, p.angle_list, p.iPAng,
            p.iPU, p.iPV, p.du_mm, p.dv_mm, p.SID, p.SDD - p.SID,
            config.source_axis_x_mm, config.source_axis_z_mm);
    }
    else if (config.geometry == GeometryModel::Helical) {
        build_helical_vec_geometry(geometry, makeHelicalParams(config));
    }
    return geometry;
}

bool writeRaw(const std::filesystem::path& path, const std::vector<float>& values)
{
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

int runCase(const ForwardCase& config)
{
    const auto& p = config.params;
    const size_t projection_count = static_cast<size_t>(p.iPAng) * p.iPV * p.iPU;
    std::vector<float> volume = makeInputVolume(config);
    std::vector<float> projection(projection_count);

    for (float value : volume) {
        if (!std::isfinite(value))
            throw std::runtime_error(config.name + " 的输入体包含 NaN/Inf");
    }

    YK_CUDA_CHECK(cudaSetDevice(config.device));
    cudaStream_t stream = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&stream));
    Mem::MemoryController memory;
    auto h_volume = memory.allocateCpu3D<float>(p.iVX, p.iVY, p.iVZ, false);
    std::copy(volume.begin(), volume.end(), h_volume.data());
    auto d_volume = memory.allocateDevice3D<float>(p.iVX, p.iVY, p.iVZ,
        config.device, false);
    auto d_projection = memory.allocateDevice3D<float>(p.iPU, p.iPV, p.iPAng,
        config.device, true);
    memory.upload3D(d_volume, h_volume);

    const auto started = std::chrono::steady_clock::now();
    bool launched = false;
    if (config.geometry == GeometryModel::Cylindrical) {
        const SHeliCTParam h = makeHelicalParams(config);
        const auto geometry = CylFpBp::buildCylindricalArcGeometry(h,
            config.curvature_radius_mm, config.channel_angle_step_rad);
        CylFpBp::Config operator_config{};
        operator_config.samples_per_voxel = config.samples_per_voxel;
        CylFpBp::Operator op;
        SVolGeom volume_geometry = SVolGeom::make_centered(p.iVX, p.iVY, p.iVZ,
            p.vox_x_mm, p.vox_y_mm, p.vox_z_mm);
        volume_geometry.center = make_float3(p.vol_offset_x_mm,
            p.vol_offset_y_mm, p.vol_offset_z_mm);
        launched = op.prepare(volume_geometry, p.iPU, p.iPV, geometry,
            operator_config, config.device) &&
            op.forward(d_volume.data(), d_projection.data(), stream, false);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        op.release();
    } else {
        const auto geometry = buildPlanarGeometry(config);
        if (geometry.size() != static_cast<size_t>(p.iPAng)) {
            cudaStreamDestroy(stream);
            throw std::runtime_error("当前构建未提供所选几何模型");
        }
        GeometryContext geometry_context;
        ResourceContext resources;
        resources.attach(stream, config.device);
        const ETask task = config.projector == ProjectorModel::Joseph
            ? ETask::FP_Joseph : ETask::FP_Siddon;
        auto op = makeForwardOperator(task);
        launched = geometry_context.initialize(p, geometry) &&
            op->prepare(geometry_context, resources) &&
            op->apply(d_volume.data(), p, d_projection.data(), resources);
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        op->release();
        resources.release();
    }
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (!launched) {
        cudaStreamDestroy(stream);
        throw std::runtime_error(config.name + " 的前投算子执行失败");
    }

    auto h_projection = memory.allocateCpu3D<float>(p.iPU, p.iPV, p.iPAng, false);
    memory.download3D(h_projection, d_projection);
    std::copy(h_projection.cdata(), h_projection.cdata() + projection_count,
        projection.begin());
    YK_CUDA_CHECK(cudaStreamDestroy(stream));

    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (float value : projection) {
        if (!std::isfinite(value))
            throw std::runtime_error(config.name + " 的投影包含 NaN/Inf");
        minimum = std::min(minimum, static_cast<double>(value));
        maximum = std::max(maximum, static_cast<double>(value));
        sum += value;
    }
    if (!(maximum > minimum) || std::max(std::fabs(minimum), std::fabs(maximum)) <= 1e-8)
        throw std::runtime_error(config.name + " 的投影没有有效动态范围");
    if (!writeRaw(config.projection_output, projection))
        throw std::runtime_error("无法写出投影: " + config.projection_output.string());

    if (!config.preview_output.empty()) {
        // 投影数组本身就是 [view][v][u]，可直接复用体数据切片 BMP writer。
        const float window_max = static_cast<float>(maximum);
        const float window_min = static_cast<float>(minimum);
        const TestImage::GrayPanel panel{ &projection, p.iPU, p.iPV, p.iPAng,
            config.preview_view, 1.f, window_min, window_max, false };
        if (!TestImage::writeGrayMontageBmp(config.preview_output, { panel }, 1, 0, 4))
            throw std::runtime_error("无法写出投影预览: " +
                config.preview_output.string());
    }

    YK_LOGI("[FP] {:<28} min={:.6e} max={:.6e} mean={:.6e} time={:.3f} ms",
        config.name, minimum, maximum, sum / projection_count, elapsed_ms);
    YK_LOGI("[FP] projection: {} ({} floats)",
        config.projection_output.string(), projection_count);
    return 0;
}

} // namespace

int runConfiguredForwardProjection(const std::filesystem::path& file,
    const std::string& case_name)
{
    try {
        const toml::table document = toml::parse_file(file.string());
        if (document["task"].value_or(std::string{}) != "forward-projection")
            throw std::runtime_error("前投配置的 task 必须是 forward-projection");
        const auto* cases = document["case"].as_array();
        if (!cases || cases->empty())
            throw std::runtime_error("配置必须至少包含一个 [[case]]");

        std::unordered_set<std::string> names;
        std::vector<ForwardCase> parsed;
        parsed.reserve(cases->size());
        const auto base = std::filesystem::absolute(file).parent_path();
        for (const auto& node : *cases) {
            const auto* table = node.as_table();
            if (!table) throw std::runtime_error("case 数组元素必须是 table");
            auto config = parseCase(*table, base);
            if (!names.insert(config.name).second)
                throw std::runtime_error("case.name 重复: " + config.name);
            parsed.push_back(std::move(config));
        }

        bool matched = case_name.empty();
        int result = 0;
        for (const auto& config : parsed) {
            if (!case_name.empty() && config.name != case_name) continue;
            matched = true;
            result |= runCase(config);
        }
        if (!matched) {
            YK_LOGE("找不到配置 case: {}", case_name);
            return 2;
        }
        return result;
    } catch (const toml::parse_error& error) {
        YK_LOGE("TOML 解析失败: {}", error.what());
    } catch (const std::exception& error) {
        YK_LOGE("前投配置执行失败: {}", error.what());
    }
    return 1;
}

} // namespace YK::TestConfig
