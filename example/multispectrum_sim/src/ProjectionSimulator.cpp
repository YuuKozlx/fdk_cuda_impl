#include "ProjectionSimulator.hpp"
#if defined(YK_MULTISPECTRUM_CUDA_SPECTRAL)
#include "CudaSpectralIntegrator.hpp"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace yk::spectral {
struct Vec3 { double x,y,z; };

namespace {
bool intersectBox(const Vec3& source, const Vec3& direction,
                  const Vec3& minimum, const Vec3& maximum,
                  double& entry, double& exit)
{
    entry = 0.0;
    exit = std::numeric_limits<double>::max();
    const std::array<double, 3> origin{source.x, source.y, source.z};
    const std::array<double, 3> ray{direction.x, direction.y, direction.z};
    const std::array<double, 3> lo{minimum.x, minimum.y, minimum.z};
    const std::array<double, 3> hi{maximum.x, maximum.y, maximum.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(ray[axis]) < 1.0e-12) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) return false;
            continue;
        }
        double a = (lo[axis] - origin[axis]) / ray[axis];
        double b = (hi[axis] - origin[axis]) / ray[axis];
        if (a > b) std::swap(a, b);
        entry = std::max(entry, a);
        exit = std::min(exit, b);
        if (entry >= exit) return false;
    }
    return exit > 0.0;
}

void validate(const SimulationConfig& config, const std::vector<std::uint8_t>& labels)
{
    const auto& g = config.geometry_config;
    if (g.views <= 0 || g.detector_u <= 0 || g.detector_v <= 0 ||
        g.volume_x <= 0 || g.volume_y <= 0 || g.volume_z <= 0 ||
        g.voxel_x_mm <= 0 || g.voxel_y_mm <= 0 || g.voxel_z_mm <= 0 ||
        g.sid_mm <= 0 || g.sdd_mm <= 0)
        throw std::runtime_error("geometry_config 的尺寸和距离必须为正");
    const std::size_t volume = static_cast<std::size_t>(g.volume_x) * g.volume_y * g.volume_z;
    if (labels.size() != volume) throw std::runtime_error("标签体素数量与配置尺寸不一致");
}

std::vector<float> calculateViewPaths(const SimulationConfig& config,
                                      const std::vector<std::uint8_t>& labels,
                                      int view)
{
    const auto& g = config.geometry_config;
    const std::size_t pixels = static_cast<std::size_t>(g.detector_u) * g.detector_v;
    std::vector<float> paths(config.materials.size() * pixels, 0.0f);
    std::array<int, 256> material_index{};
    material_index.fill(-1);
    for (std::size_t i = 0; i < config.materials.size(); ++i)
        material_index[config.materials[i].label] = static_cast<int>(i);

    const double pi = std::acos(-1.0);
    const double angle = g.start_angle_rad + 2.0 * pi * view / g.views;
    const double source_z = g.start_z_mm + g.pitch_mm_per_turn * view / g.views;
    const double ca = std::cos(angle), sa = std::sin(angle);
    const Vec3 source{g.sid_mm * ca, g.sid_mm * sa, source_z};
    const Vec3 box_min{-0.5 * g.volume_x * g.voxel_x_mm,
                       -0.5 * g.volume_y * g.voxel_y_mm,
                       -0.5 * g.volume_z * g.voxel_z_mm};
    const Vec3 box_max{-box_min.x, -box_min.y, -box_min.z};
    const double step = 0.5 * std::min({g.voxel_x_mm, g.voxel_y_mm, g.voxel_z_mm});

    for (int v = 0; v < g.detector_v; ++v) for (int u = 0; u < g.detector_u; ++u) {
        const double uu = (u - (g.detector_u - 1) * 0.5) * g.pixel_u_mm + g.offset_u_mm;
        const double vv = (v - (g.detector_v - 1) * 0.5) * g.pixel_v_mm + g.offset_v_mm;
        Vec3 detector{};
        if (config.geometry == GeometryKind::CylCbct || config.geometry == GeometryKind::CylHelical) {
            // 等角柱面以光源为圆心，通道坐标 uu 为弧长，因此 gamma=uu/SDD。
            const double gamma = uu / g.sdd_mm;
            detector = {source.x - g.sdd_mm * std::cos(angle + gamma),
                        source.y - g.sdd_mm * std::sin(angle + gamma), source_z + vv};
        } else {
            detector = {source.x - g.sdd_mm * ca - uu * sa,
                        source.y - g.sdd_mm * sa + uu * ca, source_z + vv};
        }
        Vec3 ray{detector.x - source.x, detector.y - source.y, detector.z - source.z};
        const double length = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
        ray.x /= length; ray.y /= length; ray.z /= length;
        double entry = 0.0, exit = 0.0;
        if (!intersectBox(source, ray, box_min, box_max, entry, exit)) continue;
        const int samples = std::max(1, static_cast<int>(std::ceil((exit - entry) / step)));
        const double actual_step = (exit - entry) / samples;
        const std::size_t pixel = static_cast<std::size_t>(v) * g.detector_u + u;
        for (int k = 0; k < samples; ++k) {
            const double t = entry + (k + 0.5) * actual_step;
            const int ix = static_cast<int>(std::floor((source.x + ray.x * t - box_min.x) / g.voxel_x_mm));
            const int iy = static_cast<int>(std::floor((source.y + ray.y * t - box_min.y) / g.voxel_y_mm));
            const int iz = static_cast<int>(std::floor((source.z + ray.z * t - box_min.z) / g.voxel_z_mm));
            if (ix < 0 || iy < 0 || iz < 0 || ix >= g.volume_x || iy >= g.volume_y || iz >= g.volume_z) continue;
            const auto label = labels[(static_cast<std::size_t>(iz) * g.volume_y + iy) * g.volume_x + ix];
            const int material = material_index[label];
            if (material >= 0) paths[static_cast<std::size_t>(material) * pixels + pixel] += static_cast<float>(actual_step * 0.1);
        }
    }
    return paths;
}
}

std::vector<float> ProjectionSimulator::run(const std::vector<std::uint8_t>& labels) const {
    validate(config_, labels);
    const auto& g=config_.geometry_config; const std::size_t pixels=static_cast<std::size_t>(g.detector_u)*g.detector_v;
    std::vector<float> output(static_cast<std::size_t>(g.views)*pixels,0.0f); std::vector<double> ray_paths(config_.materials.size());
    for(int view=0;view<g.views;++view){const auto paths=calculateViewPaths(config_,labels,view);for(std::size_t p=0;p<pixels;++p){for(std::size_t m=0;m<ray_paths.size();++m)ray_paths[m]=paths[m*pixels+p];output[static_cast<std::size_t>(view)*pixels+p]=static_cast<float>(model_.simulate(ray_paths).line_integral);}}
    return output;
}

bool ProjectionSimulator::runToFile(const std::vector<std::uint8_t>& labels,
                                    const std::filesystem::path& output_file,
                                    std::string& error) const {
    if (config_.use_library_fp) {
        if (!generatePathCacheWithLibraryFp(config_, labels, config_.path_cache_file, error)) return false;
    } else if (!generatePathCache(labels, config_.path_cache_file, error)) return false;
    return runFromPathCache(config_.path_cache_file, output_file, error);
}

bool ProjectionSimulator::generatePathCache(const std::vector<std::uint8_t>& labels,
                                            const std::filesystem::path& cache_file,
                                            std::string& error) const {
    try { validate(config_, labels); const auto& g=config_.geometry_config; PathCacheWriter writer;
        if(!writer.open(cache_file,g.views,g.detector_u,g.detector_v,static_cast<std::uint32_t>(config_.materials.size()),error))return false;
        for(int view=0;view<g.views;++view)if(!writer.writeView(calculateViewPaths(config_,labels,view),error))return false;
        writer.close(); return true;
    } catch(const std::exception& e){error=e.what();return false;}
}

bool ProjectionSimulator::runFromPathCache(const std::filesystem::path& cache_file,
                                           const std::filesystem::path& output_file,
                                           std::string& error) const {
    try {
        PathCacheReader reader;
        PathCacheHeader header;
        if (!reader.open(cache_file, header, error)) return false;
        const auto& g = config_.geometry_config;
        if (header.views != static_cast<std::uint32_t>(g.views) ||
            header.detector_u != static_cast<std::uint32_t>(g.detector_u) ||
            header.detector_v != static_cast<std::uint32_t>(g.detector_v) ||
            header.material_count != config_.materials.size())
            throw std::runtime_error("路径积分缓存与当前配置不匹配");
        std::ofstream out(output_file, std::ios::binary);
        if (!out) throw std::runtime_error("无法创建投影输出文件: " + output_file.string());
        const std::size_t pixels = static_cast<std::size_t>(g.detector_u) *
            g.detector_v;
        std::vector<float> paths, projection(pixels);
        std::vector<double> ray_paths(config_.materials.size());
#if defined(YK_MULTISPECTRUM_CUDA_SPECTRAL)
        CudaSpectralIntegrator cuda_integrator;
        if (config_.use_cuda_spectral) {
            std::vector<float> weights(model_.spectrum().size());
            for (std::size_t energy = 0; energy < weights.size(); ++energy)
                weights[energy] = static_cast<float>(
                    model_.spectrum()[energy].relative_photons);
            std::vector<float> table(config_.materials.size() * weights.size());
            for (std::size_t material = 0;
                material < config_.materials.size(); ++material) {
                const auto& attenuation = model_.massAttenuation(
                    config_.materials[material].label);
                if (attenuation.size() != weights.size())
                    throw std::runtime_error("材料衰减表长度与能谱长度不一致");
                for (std::size_t energy = 0; energy < weights.size(); ++energy)
                    table[material * weights.size() + energy] =
                        static_cast<float>(config_.materials[material].density_g_cm3 *
                            attenuation[energy]);
            }
            if (!cuda_integrator.initialize(weights, table,
                    static_cast<int>(config_.materials.size()), pixels, error))
                return false;
        }
#else
        if (config_.use_cuda_spectral) {
            error = "当前构建未启用 CUDA 多能谱积分，请关闭 use_cuda_spectral 或使用顶层 DLL 构建";
            return false;
        }
#endif
        for (int view = 0; view < g.views; ++view) {
            if (!reader.readView(paths, error)) return false;
#if defined(YK_MULTISPECTRUM_CUDA_SPECTRAL)
            if (config_.use_cuda_spectral) {
                if (!cuda_integrator.integrate(paths, projection, error))
                    return false;
            } else
#endif
            {
                for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                    for (std::size_t material = 0;
                        material < ray_paths.size(); ++material)
                        ray_paths[material] = paths[material * pixels + pixel];
                    projection[pixel] = static_cast<float>(
                        model_.simulate(ray_paths).line_integral);
                }
            }
            out.write(reinterpret_cast<const char*>(projection.data()),
                static_cast<std::streamsize>(projection.size() * sizeof(float)));
            if (!out) throw std::runtime_error("写入投影文件失败");
        }
        reader.close();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
}
