#include "config/YkConfiguredForwardProjection.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <toml++/toml.hpp>

#include "YKCBCT/geometry/YkPlanarGeometryBuilder.hpp"
#include "YkTestImage.hpp"
#include "YkTestPhantoms.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMem3d.hpp"
#include "global/YkLog.h"
#include "CylFpBp/BP/YkCylBackProjection.hpp"
#include "YkTestGeometry.hpp"
#include "CylFpBp/FP/YkCylForwardProjection.hpp"


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
enum class ProjectorModel { Joseph, Siddon, CylindricalJoseph, CylindricalSiddon };
enum class InputModel { Phantom, VolumeRaw, ProjectionRaw };
enum class Operation { Forward, Backproject, ForwardBackproject };
enum class CylBackprojector {
    JosephV3, Siddon, SiddonV2, SiddonV3, SiddonRayDriven, Fdk, FdkMatched
};

struct ForwardCase {
    std::string name;
    int device = 0;
    SReconstructionParams params{};
    GeometryModel geometry = GeometryModel::Circular;
    ProjectorModel projector = ProjectorModel::Joseph;
    InputModel input = InputModel::Phantom;
    Operation operation = Operation::Forward;
    CylBackprojector backprojector = CylBackprojector::JosephV3;
    std::string phantom = "catphan";
    std::filesystem::path volume_input;
    std::filesystem::path projection_input;
    std::string scalar_type = "float32";
    float input_scale = 1.f;
    std::filesystem::path projection_output;
    std::filesystem::path volume_output;
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
    if (value == "cylindrical-siddon") return ProjectorModel::CylindricalSiddon;
    throw std::runtime_error("未知 projector.model: " + value);
}

CylBackprojector parseCylBackprojector(const std::string& value)
{
    if (value == "joseph-v3" || value == "v3")
        return CylBackprojector::JosephV3;
    if (value == "siddon") return CylBackprojector::Siddon;
    if (value == "siddon-v2") return CylBackprojector::SiddonV2;
    if (value == "siddon-v3") return CylBackprojector::SiddonV3;
    if (value == "siddon-ray-driven") return CylBackprojector::SiddonRayDriven;
    if (value == "fdk") return CylBackprojector::Fdk;
    if (value == "fdk-matched") return CylBackprojector::FdkMatched;
    throw std::runtime_error("未知 operator.backprojector: " + value);
}

ForwardCase parseCase(const toml::table& table, const std::filesystem::path& base)
{
    ForwardCase result{};
    result.name = required<std::string>(table, "name", "case");
    result.device = static_cast<int>(optional<int64_t>(table, "device", 0));

    auto& p = result.params;
    const auto& scan = requiredTable(table, "scan", result.name);
    p.scan.Nu = static_cast<int>(required<int64_t>(scan, "nu", "scan"));
    p.scan.Nv = static_cast<int>(required<int64_t>(scan, "nv", "scan"));
    p.scan.NAng = static_cast<int>(required<int64_t>(scan, "views", "scan"));
    p.scan.totalViews = p.scan.NAng;
    p.scan.du_mm = static_cast<float>(optional<double>(scan, "du_mm", 1.0));
    p.scan.dv_mm = static_cast<float>(optional<double>(scan, "dv_mm", 1.0));
    p.scan.offsetU_mm = static_cast<float>(optional<double>(scan, "offset_u_mm", 0.0));
    p.scan.offsetV_mm = static_cast<float>(optional<double>(scan, "offset_v_mm", 0.0));
    p.scan.sourceOffsetX_mm = static_cast<float>(optional<double>(scan, "source_offset_x_mm", 0.0));
    p.scan.sourceOffsetY_mm = static_cast<float>(optional<double>(scan, "source_offset_y_mm", 0.0));
    p.scan.sourceOffsetZ_mm = static_cast<float>(optional<double>(scan, "source_offset_z_mm", 0.0));
    p.scan.sid_mm = static_cast<float>(required<double>(scan, "sid_mm", "scan"));
    p.scan.sdd_mm = static_cast<float>(required<double>(scan, "sdd_mm", "scan"));
    p.scan.start_angle_rad = static_cast<float>(
        optional<double>(scan, "start_angle_deg", 0.0) * kPi / 180.0);
    p.scan.range_rad = static_cast<float>(
        optional<double>(scan, "scan_range_deg", 360.0) * kPi / 180.0);
    p.scan.direction = static_cast<int>(optional<int64_t>(scan, "direction", 1));
    p.scan.tiltU_rad = static_cast<float>(
        optional<double>(scan, "tilt_u_deg", 0.0) * kPi / 180.0);
    p.scan.tiltN_rad = static_cast<float>(
        optional<double>(scan, "tilt_n_deg", 0.0) * kPi / 180.0);
    p.scan.tiltV_rad = static_cast<float>(
        optional<double>(scan, "tilt_v_deg", 0.0) * kPi / 180.0);
    p.scan.angles.resize(std::max(p.scan.NAng, 0));
    for (int i = 0; i < p.scan.NAng; ++i) {
        // scan_range 是采集覆盖范围，均匀扫描不重复采集范围终点。
        p.scan.angles[i] = p.scan.start_angle_rad + p.scan.direction *
            p.scan.range_rad * static_cast<float>(i) / p.scan.NAng;
    }

    const auto& volume = requiredTable(table, "volume", result.name);
    p.volume.Nx = static_cast<int>(required<int64_t>(volume, "nx", "volume"));
    p.volume.Ny = static_cast<int>(required<int64_t>(volume, "ny", "volume"));
    p.volume.Nz = static_cast<int>(required<int64_t>(volume, "nz", "volume"));
    const auto* voxel = volume["voxel_mm"].as_array();
    if (!voxel || voxel->size() != 3)
        throw std::runtime_error("volume.voxel_mm 必须包含 3 个数");
    p.volume.voxelX_mm = static_cast<float>(arrayValue<double>(*voxel, 0, "volume.voxel_mm"));
    p.volume.voxelY_mm = static_cast<float>(arrayValue<double>(*voxel, 1, "volume.voxel_mm"));
    p.volume.voxelZ_mm = static_cast<float>(arrayValue<double>(*voxel, 2, "volume.voxel_mm"));
    if (const auto* offset = volume["offset_mm"].as_array()) {
        if (offset->size() != 3)
            throw std::runtime_error("volume.offset_mm 必须包含 3 个数");
        p.volume.centerX_mm = static_cast<float>(arrayValue<double>(*offset, 0, "volume.offset_mm"));
        p.volume.centerY_mm = static_cast<float>(arrayValue<double>(*offset, 1, "volume.offset_mm"));
        p.volume.centerZ_mm = static_cast<float>(arrayValue<double>(*offset, 2, "volume.offset_mm"));
    }

    const auto& geometry = requiredTable(table, "geometry", result.name);
    result.geometry = parseGeometry(required<std::string>(geometry, "model", "geometry"));
    result.pitch_mm = static_cast<float>(optional<double>(geometry, "pitch_mm", 0.0));
    result.start_z_mm = static_cast<float>(optional<double>(geometry, "start_z_mm", 0.0));
    result.views_per_rotation = static_cast<int>(optional<int64_t>(geometry,
        "views_per_rotation", p.scan.NAng));
    result.source_lateral_mm = static_cast<float>(optional<double>(geometry,
        "source_lateral_mm", 0.0));
    result.source_axis_x_mm = static_cast<float>(optional<double>(geometry,
        "source_axis_x_mm", 0.0));
    result.source_axis_z_mm = static_cast<float>(optional<double>(geometry,
        "source_axis_z_mm", 0.0));
    result.curvature_radius_mm = static_cast<float>(optional<double>(geometry,
        "curvature_radius_mm", p.scan.sdd_mm));
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
        result.scalar_type = optional<std::string>(input, "scalar_type", "float32");
        result.input_scale = static_cast<float>(optional<double>(input, "scale", 1.0));
    } else if (input_mode == "projection-raw") {
        result.input = InputModel::ProjectionRaw;
        result.projection_input = resolvePath(base,
            required<std::string>(input, "projection", "input"));
    } else {
        throw std::runtime_error("未知 input.mode: " + input_mode);
    }

    const auto& output = requiredTable(table, "output", result.name);
    result.projection_output = resolvePath(base,
        optional<std::string>(output, "projection", ""));
    result.preview_output = resolvePath(base,
        optional<std::string>(output, "preview", ""));
    result.preview_view = static_cast<int>(optional<int64_t>(output,
        "preview_view", 0));
    result.volume_output = resolvePath(base,
        optional<std::string>(output, "volume", ""));

    if (const auto* operation = table["operator"].as_table()) {
        const std::string mode = optional<std::string>(*operation, "mode", "forward");
        if (mode == "forward") result.operation = Operation::Forward;
        else if (mode == "backproject") result.operation = Operation::Backproject;
        else if (mode == "forward-backproject")
            result.operation = Operation::ForwardBackproject;
        else throw std::runtime_error(
            "operator.mode 必须是 forward、backproject 或 forward-backproject");
        result.backprojector = parseCylBackprojector(optional<std::string>(
            *operation, "backprojector", "v3"));
    }

    if (result.device < 0 || p.scan.Nu <= 0 || p.scan.Nv <= 0 || p.scan.NAng <= 0 ||
        p.volume.Nx <= 0 || p.volume.Ny <= 0 || p.volume.Nz <= 0 || p.scan.du_mm <= 0.f ||
        p.scan.dv_mm <= 0.f || p.volume.voxelX_mm <= 0.f || p.volume.voxelY_mm <= 0.f ||
        p.volume.voxelZ_mm <= 0.f || p.scan.sid_mm <= 0.f || p.scan.sdd_mm <= p.scan.sid_mm ||
        p.scan.range_rad <= 0.f || (p.scan.direction != 1 && p.scan.direction != -1))
        throw std::runtime_error(result.name + " 的尺寸或扫描参数无效");
    if (result.preview_view < 0 || result.preview_view >= p.scan.NAng)
        throw std::runtime_error(result.name + " 的 output.preview_view 超出视图范围");

    const bool cylindrical = result.geometry == GeometryModel::Cylindrical;
    const bool cylindrical_projector = result.projector == ProjectorModel::CylindricalJoseph ||
        result.projector == ProjectorModel::CylindricalSiddon;
    if (cylindrical != cylindrical_projector)
        throw std::runtime_error(result.name +
            "：cylindrical 几何必须与 cylindrical-joseph 或 cylindrical-siddon 算子配对");
    if ((result.geometry == GeometryModel::Helical || cylindrical) &&
        (result.views_per_rotation <= 0 || !std::isfinite(result.pitch_mm)))
        throw std::runtime_error(result.name + " 的螺旋参数无效");
    if (result.geometry == GeometryModel::Helical || cylindrical) {
        // 角度只由 scan 表生成；views_per_rotation 只校验采样密度，避免它
        // 悄悄成为第二份角度来源。允许少量浮点舍入误差。
        const float configured_step = p.scan.range_rad / p.scan.NAng;
        const float expected_step = 2.f * kPi / result.views_per_rotation;
        if (std::fabs(configured_step - expected_step) >
            1e-5f * std::max(configured_step, expected_step))
            throw std::runtime_error(result.name +
                "：scan 角步长与 geometry.views_per_rotation 不一致");
    }
    if (cylindrical && (!(result.curvature_radius_mm > 0.f) ||
        result.channel_angle_step_rad < 0.f || result.samples_per_voxel <= 0.f))
        throw std::runtime_error(result.name + " 的圆柱探测器参数无效");
    if (result.operation != Operation::Forward &&
        (!cylindrical || result.volume_output.empty()))
        throw std::runtime_error(result.name +
            "：BP 仅支持 cylindrical，且必须设置 output.volume");
    if (result.operation != Operation::Forward &&
        (result.backprojector == CylBackprojector::Fdk ||
         result.backprojector == CylBackprojector::FdkMatched)) {
        const float tolerance = std::max(1e-3f, 1e-5f * p.scan.sdd_mm);
        if (std::fabs(result.curvature_radius_mm - p.scan.sdd_mm) > tolerance)
            throw std::runtime_error(result.name +
                "：圆柱 FDK/FDK-matched BP 仅支持 R=SDD；请先将一般圆柱投影重映射到源中心等角弧面");
    }
    if (result.operation == Operation::Backproject &&
        result.input != InputModel::ProjectionRaw)
        throw std::runtime_error(result.name +
            "：backproject 必须使用 input.mode=projection-raw");
    if (result.operation != Operation::Backproject &&
        result.input == InputModel::ProjectionRaw)
        throw std::runtime_error(result.name +
            "：projection-raw 只用于 backproject");
    if (result.operation != Operation::Backproject &&
        result.projection_output.empty())
        throw std::runtime_error(result.name + "：FP 必须设置 output.projection");
    if (result.scalar_type != "float32" && result.scalar_type != "uint8")
        throw std::runtime_error("volume-raw scalar_type 仅支持 float32 或 uint8");
    if (!(result.input_scale > 0.f))
        throw std::runtime_error("input.scale 必须大于 0");
    if (result.geometry == GeometryModel::Planar &&
        !std::isfinite(result.source_lateral_mm))
        throw std::runtime_error(result.name + " 的 source_lateral_mm 无效");
    if (result.geometry == GeometryModel::PlanarEllipse &&
        (!(result.source_axis_x_mm > 0.f) || !(result.source_axis_z_mm > 0.f)))
        throw std::runtime_error(result.name + " 的平面椭圆源轨迹轴长必须大于 0");
    if ((result.geometry == GeometryModel::Planar ||
         result.geometry == GeometryModel::PlanarEllipse) &&
        (p.scan.offsetU_mm != 0.f || p.scan.offsetV_mm != 0.f ||
         p.scan.tiltU_rad != 0.f || p.scan.tiltN_rad != 0.f ||
         p.scan.tiltV_rad != 0.f))
        throw std::runtime_error(result.name +
            "：当前平面 CT builder 不支持探测器 offset/tilt，不能静默忽略这些参数");
    return result;
}

std::vector<float> readRawVolume(const ForwardCase& config, size_t count)
{
    const auto& path = config.volume_input;
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    const size_t scalar_bytes = config.scalar_type == "uint8" ? 1 : sizeof(float);
    const size_t expected = count * scalar_bytes;
    if (error || bytes != expected)
        throw std::runtime_error("volume raw 字节数错误: " + path.string() +
            "，期望 " + std::to_string(expected) + " 字节");
    std::ifstream input(path, std::ios::binary);
    std::vector<float> values(count);
    if (config.scalar_type == "uint8") {
        std::vector<uint8_t> raw(count);
        if (!input || !input.read(reinterpret_cast<char*>(raw.data()),
            static_cast<std::streamsize>(expected)))
            throw std::runtime_error("无法读取 uint8 volume raw: " + path.string());
        std::transform(raw.begin(), raw.end(), values.begin(),
            [&](uint8_t value) { return config.input_scale * value; });
    }
    else {
        if (!input || !input.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(expected)))
            throw std::runtime_error("无法读取 float32 volume raw: " + path.string());
        if (config.input_scale != 1.f)
            for (float& value : values) value *= config.input_scale;
    }
    return values;
}

std::vector<float> makeInputVolume(const ForwardCase& config)
{
    const auto& p = config.params;
    const size_t count = static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz;
    if (config.input == InputModel::VolumeRaw)
        return readRawVolume(config, count);
    if (config.phantom == "basic") return TestPhantom::makeBasic(p);
    if (config.phantom == "arrow") return TestPhantom::makeArrowDirections(p);
    return TestPhantom::makeCatphanLike(p);
}

SHeliCTParam makeHelicalParams(const ForwardCase& config)
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
    h.vol_offset_x_mm = p.volume.centerX_mm;
    h.vol_offset_y_mm = p.volume.centerY_mm;
    h.vol_offset_z_mm = p.volume.centerZ_mm;
    h.tiltu_angle_rad = p.scan.tiltU_rad;
    h.tiltn_angle_rad = p.scan.tiltN_rad;
    h.tiltv_angle_rad = p.scan.tiltV_rad;
    h.pitch_mm = config.pitch_mm;
    h.start_z_mm = config.start_z_mm;
    h.views_per_rot = config.views_per_rotation;
    h.angle_list = p.scan.angles;
    return h;
}

std::vector<SConeProjGeomVec> buildPlanarGeometry(const ForwardCase& config)
{
    const auto& p = config.params;
    std::vector<SConeProjGeomVec> geometry;
    if (config.geometry == GeometryModel::Circular) {
        detail::buildCircularViews(p, geometry);
    } else if (config.geometry == GeometryModel::Planar) {
        buildPlanarRotatingSourceGeometry(p.scan.angles, p.scan.Nu, p.scan.Nv,
            p.scan.du_mm, p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
            config.source_lateral_mm, geometry);
    } else if (config.geometry == GeometryModel::PlanarEllipse) {
        buildPlanarEllipticSourceGeometry(p.scan.angles, p.scan.Nu, p.scan.Nv,
            p.scan.du_mm, p.scan.dv_mm, p.scan.sid_mm, p.scan.sdd_mm - p.scan.sid_mm,
            config.source_axis_x_mm, config.source_axis_z_mm, geometry);
    }
    else if (config.geometry == GeometryModel::Helical) {
        geometry = TestGeometry::helicalFlat(makeHelicalParams(config));
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
    const size_t projection_count = static_cast<size_t>(p.scan.NAng) * p.scan.Nv * p.scan.Nu;
    std::vector<float> volume;
    std::vector<float> projection(projection_count);
    std::vector<float> backprojection;

    if (config.operation == Operation::Backproject) {
        std::error_code error;
        if (std::filesystem::file_size(config.projection_input, error) !=
            projection_count * sizeof(float) || error)
            throw std::runtime_error(config.name + " 的输入投影文件大小错误");
        std::ifstream input(config.projection_input, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(projection.data()),
            static_cast<std::streamsize>(projection_count * sizeof(float))))
            throw std::runtime_error(config.name + " 的输入投影读取失败");
    }
    else {
        volume = makeInputVolume(config);
        for (float value : volume) {
            if (!std::isfinite(value))
                throw std::runtime_error(config.name + " 的输入体包含 NaN/Inf");
        }
    }

    YK_CUDA_CHECK(cudaSetDevice(config.device));
    cudaStream_t stream = nullptr;
    YK_CUDA_CHECK(cudaStreamCreate(&stream));
    Mem::MemoryController memory;
    auto d_volume = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz,
        config.device, config.operation == Operation::Backproject);
    auto d_projection = memory.allocateDevice3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng,
        config.device, true);
    if (config.operation == Operation::Backproject) {
        YK_CUDA_CHECK(cudaMemcpyAsync(d_projection.data(), projection.data(),
            projection_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }
    else {
        auto h_volume = memory.allocateCpu3D<float>(p.volume.Nx, p.volume.Ny, p.volume.Nz, false);
        std::copy(volume.begin(), volume.end(), h_volume.data());
        memory.upload3D(d_volume, h_volume);
    }

    const auto started = std::chrono::steady_clock::now();
    bool launched = false;
    if (config.geometry == GeometryModel::Cylindrical) {
        const SHeliCTParam h = makeHelicalParams(config);
        const auto geometry = config.pitch_mm == 0.f
            ? TestGeometry::staticCyl(h, config.curvature_radius_mm,
                config.channel_angle_step_rad)
            : TestGeometry::helicalCyl(h, config.curvature_radius_mm,
                config.channel_angle_step_rad);
        CylFpBp::Config operator_config{};
        operator_config.samples_per_voxel = config.samples_per_voxel;
        SVolGeom volume_geometry = SVolGeom::make_centered(p.volume.Nx, p.volume.Ny, p.volume.Nz,
            p.volume.voxelX_mm, p.volume.voxelY_mm, p.volume.voxelZ_mm);
        volume_geometry.center = make_float3(p.volume.centerX_mm,
            p.volume.centerY_mm, p.volume.centerZ_mm);
        ResourceContext resources;
        resources.attach(stream, config.device);
        const auto forward_model =
            config.projector == ProjectorModel::CylindricalSiddon
            ? CylFpBp::EForwardProjection::Siddon
            : CylFpBp::EForwardProjection::Joseph;
        auto forward = CylFpBp::makeForwardProjection(forward_model);
        const auto back_model = [&] {
            switch (config.backprojector) {
            case CylBackprojector::JosephV3:
                return CylFpBp::EBackProjection::JosephV3;
            case CylBackprojector::Siddon:
                return CylFpBp::EBackProjection::Siddon;
            case CylBackprojector::SiddonV2:
                return CylFpBp::EBackProjection::SiddonV2;
            case CylBackprojector::SiddonV3:
                return CylFpBp::EBackProjection::SiddonV3;
            case CylBackprojector::SiddonRayDriven:
                return CylFpBp::EBackProjection::SiddonRayDriven;
            case CylBackprojector::Fdk:
                return CylFpBp::EBackProjection::Fdk;
            case CylBackprojector::FdkMatched:
                return CylFpBp::EBackProjection::FdkMatched;
            }
            return CylFpBp::EBackProjection::JosephV3;
        };
        auto back = CylFpBp::makeBackProjection(back_model());
        const bool needs_forward = config.operation != Operation::Backproject;
        const bool needs_back = config.operation != Operation::Forward;
        launched = (!needs_forward ||
                (forward && forward->prepare(volume_geometry, p.scan.Nu, p.scan.Nv,
                    geometry, operator_config, resources))) &&
            (!needs_back ||
                (back && back->prepare(volume_geometry, p.scan.Nu, p.scan.Nv,
                    geometry, operator_config, resources)));
        if (launched && needs_forward) {
            launched = forward->apply(d_volume.data(), d_projection.data(),
                false, resources);
        }
        if (launched && config.operation != Operation::Forward) {
            float* d_result = d_volume.data();
            Mem::DeviceLinearBuffer3D<float> d_roundtrip{};
            if (config.operation == Operation::ForwardBackproject) {
                d_roundtrip = memory.allocateDevice3D<float>(p.volume.Nx, p.volume.Ny,
                    p.volume.Nz, config.device, true);
                d_result = d_roundtrip.data();
            }
            launched = back->apply(d_projection.data(), d_result, false,
                resources);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            backprojection.resize(static_cast<size_t>(p.volume.Nx) * p.volume.Ny * p.volume.Nz);
            YK_CUDA_CHECK(cudaMemcpy(backprojection.data(), d_result,
                backprojection.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (back) back->release();
        if (forward) forward->release();
        resources.release();
    } else {
        const auto geometry = buildPlanarGeometry(config);
        if (geometry.size() != static_cast<size_t>(p.scan.NAng)) {
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

    if (config.operation != Operation::Backproject) {
        auto h_projection = memory.allocateCpu3D<float>(p.scan.Nu, p.scan.Nv, p.scan.NAng,
            false);
        memory.download3D(h_projection, d_projection);
        std::copy(h_projection.cdata(), h_projection.cdata() + projection_count,
            projection.begin());
    }
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
    if (config.operation != Operation::Backproject &&
        !writeRaw(config.projection_output, projection))
        throw std::runtime_error("无法写出投影: " + config.projection_output.string());

    if (config.operation != Operation::Backproject &&
        !config.preview_output.empty()) {
        // 投影数组本身就是 [view][v][u]，可直接复用体数据切片 BMP writer。
        const float window_max = static_cast<float>(maximum);
        const float window_min = static_cast<float>(minimum);
        const TestImage::GrayPanel panel{ &projection, p.scan.Nu, p.scan.Nv, p.scan.NAng,
            config.preview_view, 1.f, window_min, window_max, false };
        if (!TestImage::writeGrayMontageBmp(config.preview_output, { panel }, 1, 0, 4))
            throw std::runtime_error("无法写出投影预览: " +
                config.preview_output.string());
    }
    if (config.operation != Operation::Forward) {
        const auto finite_nonzero = std::all_of(backprojection.begin(),
            backprojection.end(), [](float value) { return std::isfinite(value); }) &&
            std::any_of(backprojection.begin(), backprojection.end(),
                [](float value) { return std::fabs(value) > 1e-8f; });
        if (!finite_nonzero || !writeRaw(config.volume_output, backprojection))
            throw std::runtime_error("圆柱 BP 输出无效或写出失败: " +
                config.volume_output.string());
        YK_LOGI("[CylFpBp] backprojection: {} ({} floats)",
            config.volume_output.string(), backprojection.size());
    }

    if (config.operation != Operation::Backproject) {
        YK_LOGI("[FP] {:<28} min={:.6e} max={:.6e} mean={:.6e} time={:.3f} ms",
            config.name, minimum, maximum, sum / projection_count, elapsed_ms);
        YK_LOGI("[FP] projection: {} ({} floats)",
            config.projection_output.string(), projection_count);
    }
    return 0;
}

} // namespace

int runConfiguredForwardProjection(const std::filesystem::path& file,
    const std::string& case_name)
{
    try {
        const toml::table document = toml::parse_file(file.string());
        const std::string task = document["task"].value_or(std::string{});
        if (task != "forward-projection" && task != "cyl-fp-bp")
            throw std::runtime_error(
                "算子配置的 task 必须是 forward-projection 或 cyl-fp-bp");
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
            if (task == "cyl-fp-bp" &&
                config.geometry != GeometryModel::Cylindrical)
                throw std::runtime_error(
                    "cyl-fp-bp 任务只接受 cylindrical geometry");
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
