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
struct GaussLegendreRule { std::vector<double> nodes; std::vector<double> weights; };

GaussLegendreRule gaussLegendreRule(int count)
{
    switch (count) {
    case 1: return {{0.0}, {2.0}};
    case 2: return {{-0.5773502691896258, 0.5773502691896258}, {1.0, 1.0}};
    case 3: return {{-0.7745966692414834, 0.0, 0.7745966692414834},
        {0.5555555555555556, 0.8888888888888888, 0.5555555555555556}};
    case 4: return {{-0.8611363115940526, -0.3399810435848563,
        0.3399810435848563, 0.8611363115940526},
        {0.3478548451374539, 0.6521451548625461,
         0.6521451548625461, 0.3478548451374539}};
    case 5: return {{-0.9061798459386640, -0.5384693101056831, 0.0,
        0.5384693101056831, 0.9061798459386640},
        {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
         0.4786286704993665, 0.2369268850561891}};
    case 6: return {{-0.9324695142031521, -0.6612093864662645,
        -0.2386191860831969, 0.2386191860831969,
        0.6612093864662645, 0.9324695142031521},
        {0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
         0.4679139345726910, 0.3607615730481386, 0.1713244923791704}};
    case 7: return {{-0.9491079123427585, -0.7415311855993945,
        -0.4058451513773972, 0.0, 0.4058451513773972,
        0.7415311855993945, 0.9491079123427585},
        {0.1294849661688697, 0.2797053914892766, 0.3818300505051189,
         0.4179591836734694, 0.3818300505051189, 0.2797053914892766,
         0.1294849661688697}};
    default: throw std::runtime_error("矩形焦点 Gauss-Legendre 采样点数仅支持 1..7");
    }
}

void writeProjectionMetadata(const std::filesystem::path& raw_file,
    int columns, int rows, int frames, const char* signal)
{
    auto metadata_file = raw_file;
    metadata_file += ".json";
    std::ofstream metadata(metadata_file);
    if (!metadata)
        throw std::runtime_error("无法创建投影元数据文件: " + metadata_file.string());
    metadata << "{\n"
        << "  \"columns\": " << columns << ",\n"
        << "  \"rows\": " << rows << ",\n"
        << "  \"frames\": " << frames << ",\n"
        << "  \"data_type\": \"float32\",\n"
        << "  \"byte_order\": \"little_endian\",\n"
        << "  \"layout\": \"frame_row_column\",\n"
        << "  \"signal\": \"" << signal << "\"\n"
        << "}\n";
    if (!metadata)
        throw std::runtime_error("写入投影元数据文件失败: " + metadata_file.string());
}

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
    const auto& g = config.geometry.parameters;
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
    const auto& g = config.geometry.parameters;
    const std::size_t pixels = static_cast<std::size_t>(g.detector_u) * g.detector_v;
    std::vector<float> paths(config.projection.materials.size() * pixels, 0.0f);
    std::array<int, 256> material_index{};
    material_index.fill(-1);
    for (std::size_t i = 0; i < config.projection.materials.size(); ++i)
        material_index[config.projection.materials[i].label] = static_cast<int>(i);

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
        if (config.geometry.kind == GeometryKind::CylCbct || config.geometry.kind == GeometryKind::CylHelical) {
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

std::vector<float> flatGeometryFlux(const SimulationConfig& config, int view)
{
    const auto& g = config.geometry.parameters;
    const std::size_t pixels = static_cast<std::size_t>(g.detector_u) * g.detector_v;
    std::vector<float> result(pixels, 1.f);
    if (!config.projection.apply_geometry_flux ||
        (config.geometry.kind != GeometryKind::FlatCbct && config.geometry.kind != GeometryKind::FlatHelical))
        return result;
    const double pi = std::acos(-1.0);
    const double angle = g.start_angle_rad + 2.0 * pi * view / g.views;
    const double z = g.start_z_mm + g.pitch_mm_per_turn * view / g.views;
    const double ca = std::cos(angle), sa = std::sin(angle);
    auto rotate = [&](double x, double y, double zz) { return Vec3{ca*x-sa*y, sa*x+ca*y, zz+z}; };
    const Vec3 source = rotate(g.source_offset_x_mm, -g.sid_mm + g.source_offset_y_mm,
        g.source_offset_z_mm);
    const Vec3 center = rotate(0.0, g.sdd_mm - g.sid_mm, 0.0);
    const Vec3 u_axis{ca, sa, 0.0};
    const Vec3 v_axis{0.0, 0.0, 1.0};
    const Vec3 n_axis{-sa, ca, 0.0};
    const Vec3 reference_ray{center.x + g.offset_n_mm*n_axis.x - source.x,
                             center.y + g.offset_n_mm*n_axis.y - source.y,
                             center.z - source.z};
    const double reference_distance = std::sqrt(
        reference_ray.x*reference_ray.x + reference_ray.y*reference_ray.y +
        reference_ray.z*reference_ray.z);
    const double reference_cosine = std::max(1e-9,
        (reference_ray.x*n_axis.x + reference_ray.y*n_axis.y +
         reference_ray.z*n_axis.z) / std::max(reference_distance, 1e-9));
    const double reference = reference_cosine /
        std::max(reference_distance*reference_distance, 1e-18);
    for (int v = 0; v < g.detector_v; ++v) for (int u = 0; u < g.detector_u; ++u) {
        const double du = (u - (g.detector_u - 1) * .5) * g.pixel_u_mm + g.offset_u_mm;
        const double dv = (v - (g.detector_v - 1) * .5) * g.pixel_v_mm + g.offset_v_mm;
        const Vec3 p{center.x + du*u_axis.x + g.offset_n_mm*n_axis.x,
                     center.y + du*u_axis.y + g.offset_n_mm*n_axis.y,
                     center.z + dv};
        const Vec3 d{p.x-source.x, p.y-source.y, p.z-source.z};
        const double r = std::sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
        const double inv = 1.0 / std::max(r, 1e-9);
        const double cosine = std::max(0.0, (d.x*n_axis.x+d.y*n_axis.y+d.z*n_axis.z)*inv);
        result[static_cast<std::size_t>(v)*g.detector_u+u] = static_cast<float>((cosine/(r*r))/reference);
    }
    return result;
}
}

std::vector<float> ProjectionSimulator::run(const std::vector<std::uint8_t>& labels) const {
    validate(config_, labels);
    const auto& g=config_.geometry.parameters; const std::size_t pixels=static_cast<std::size_t>(g.detector_u)*g.detector_v;
    std::vector<float> output(static_cast<std::size_t>(g.views)*pixels,0.0f); std::vector<double> ray_paths(config_.projection.materials.size());
    for(int view=0;view<g.views;++view){const auto paths=calculateViewPaths(config_,labels,view);for(std::size_t p=0;p<pixels;++p){for(std::size_t m=0;m<ray_paths.size();++m)ray_paths[m]=paths[m*pixels+p];output[static_cast<std::size_t>(view)*pixels+p]=static_cast<float>(model_.simulate(ray_paths).line_integral);}}
    return output;
}

bool ProjectionSimulator::runToFile(const std::vector<std::uint8_t>& labels,
                                    const std::filesystem::path& output_file,
                                    std::string& error) const {
    const auto& spot = config_.projection.focal_spot;
    if (spot.enabled && spot.samples_u * spot.samples_v > 1) {
        try {
            const auto rule_u = gaussLegendreRule(spot.samples_u);
            const auto rule_v = gaussLegendreRule(spot.samples_v);
            const int sample_count = spot.samples_u * spot.samples_v;
            std::vector<std::filesystem::path> projections, energies, caches;
            for (int v = 0; v < spot.samples_v; ++v) for (int u = 0; u < spot.samples_u; ++u) {
                const int index = v * spot.samples_u + u;
                auto sub = config_;
                sub.projection.focal_spot.enabled = false;
                sub.geometry.parameters.source_offset_x_mm +=
                    0.5 * rule_u.nodes[u] * spot.size_u_mm;
                sub.geometry.parameters.source_offset_z_mm +=
                    0.5 * rule_v.nodes[v] * spot.size_v_mm;
                auto suffix = std::string(".focal-") + std::to_string(index);
                auto projection = output_file; projection += suffix;
                auto energy = config_.projection.energy_output_file; energy += suffix;
                auto cache = config_.projection.path_cache_file; cache += suffix;
                sub.projection.energy_output_file = energy;
                sub.projection.path_cache_file = cache;
                ProjectionSimulator simulator(sub, model_);
                if (!simulator.runToFile(labels, projection, error)) return false;
                projections.push_back(projection); energies.push_back(energy); caches.push_back(cache);
            }
            std::vector<std::ifstream> projection_inputs, energy_inputs;
            for (int i = 0; i < sample_count; ++i) {
                projection_inputs.emplace_back(projections[i], std::ios::binary);
                energy_inputs.emplace_back(energies[i], std::ios::binary);
                if (!projection_inputs.back() || !energy_inputs.back())
                    throw std::runtime_error("无法读取子焦点投影");
            }
            std::ofstream projection_out(output_file, std::ios::binary);
            std::ofstream energy_out(config_.projection.energy_output_file, std::ios::binary);
            const std::size_t pixels = static_cast<std::size_t>(config_.geometry.parameters.detector_u) *
                config_.geometry.parameters.detector_v;
            std::vector<float> input(pixels), transmission(pixels), energy_sum(pixels), output(pixels);
            for (int view = 0; view < config_.geometry.parameters.views; ++view) {
                std::fill(transmission.begin(), transmission.end(), 0.f);
                std::fill(energy_sum.begin(), energy_sum.end(), 0.f);
                for (int sample = 0; sample < sample_count; ++sample) {
                    projection_inputs[sample].read(reinterpret_cast<char*>(input.data()), pixels * sizeof(float));
                    const int u = sample % spot.samples_u;
                    const int v = sample / spot.samples_u;
                    const float weight = static_cast<float>(rule_u.weights[u] * rule_v.weights[v] / 4.0);
                    for (std::size_t i = 0; i < pixels; ++i) transmission[i] += weight * std::exp(-input[i]);
                    energy_inputs[sample].read(reinterpret_cast<char*>(input.data()), pixels * sizeof(float));
                    for (std::size_t i = 0; i < pixels; ++i) energy_sum[i] += weight * input[i];
                }
                for (std::size_t i = 0; i < pixels; ++i) {
                    output[i] = -std::log(std::max(transmission[i], 1e-30f));
                }
                projection_out.write(reinterpret_cast<const char*>(output.data()), pixels * sizeof(float));
                energy_out.write(reinterpret_cast<const char*>(energy_sum.data()), pixels * sizeof(float));
            }
            writeProjectionMetadata(output_file, config_.geometry.parameters.detector_u,
                config_.geometry.parameters.detector_v, config_.geometry.parameters.views,
                "negative_log_transmission");
            writeProjectionMetadata(config_.projection.energy_output_file,
                config_.geometry.parameters.detector_u, config_.geometry.parameters.detector_v,
                config_.geometry.parameters.views, "transmitted_energy_keV_per_incident_photon");
            projection_inputs.clear();
            energy_inputs.clear();
            for (int i = 0; i < sample_count; ++i) {
                std::error_code ignored;
                std::filesystem::remove(projections[i], ignored);
                std::filesystem::remove(projections[i].string() + ".json", ignored);
                std::filesystem::remove(energies[i], ignored);
                std::filesystem::remove(energies[i].string() + ".json", ignored);
                std::filesystem::remove(caches[i], ignored);
            }
            return true;
        } catch (const std::exception& e) { error = e.what(); return false; }
    }
    if (config_.projection.use_library_fp) {
        if (!generatePathCacheWithLibraryFp(config_, labels, config_.projection.path_cache_file, error)) return false;
    } else if (!generatePathCache(labels, config_.projection.path_cache_file, error)) return false;
    return runFromPathCache(config_.projection.path_cache_file, output_file, error);
}

bool ProjectionSimulator::generatePathCache(const std::vector<std::uint8_t>& labels,
                                            const std::filesystem::path& cache_file,
                                            std::string& error) const {
    try { validate(config_, labels); const auto& g=config_.geometry.parameters; PathCacheWriter writer;
        if(!writer.open(cache_file,g.views,g.detector_u,g.detector_v,static_cast<std::uint32_t>(config_.projection.materials.size()),error))return false;
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
        const auto& g = config_.geometry.parameters;
        if (header.views != static_cast<std::uint32_t>(g.views) ||
            header.detector_u != static_cast<std::uint32_t>(g.detector_u) ||
            header.detector_v != static_cast<std::uint32_t>(g.detector_v) ||
            header.material_count != config_.projection.materials.size())
            throw std::runtime_error("路径积分缓存与当前配置不匹配");
        std::ofstream out(output_file, std::ios::binary);
        if (!out) throw std::runtime_error("无法创建投影输出文件: " + output_file.string());
        std::ofstream energy_out;
        if (!config_.projection.energy_output_file.empty()) {
            energy_out.open(config_.projection.energy_output_file, std::ios::binary);
            if (!energy_out) throw std::runtime_error("无法创建能量积分输出文件: " + config_.projection.energy_output_file.string());
        }
        const std::size_t pixels = static_cast<std::size_t>(g.detector_u) *
            g.detector_v;
        std::vector<float> paths, projection(pixels), energy_signal(pixels);
        std::vector<double> ray_paths(config_.projection.materials.size());
#if defined(YK_MULTISPECTRUM_CUDA_SPECTRAL)
        CudaSpectralIntegrator cuda_integrator;
        if (config_.projection.use_cuda_spectral) {
            std::vector<float> weights(model_.spectrum().size());
            for (std::size_t energy = 0; energy < weights.size(); ++energy)
                weights[energy] = static_cast<float>(
                    model_.spectrum()[energy].relative_photons);
            std::vector<float> energies(model_.spectrum().size());
            for (std::size_t energy = 0; energy < energies.size(); ++energy)
                energies[energy] = static_cast<float>(model_.spectrum()[energy].energy_keV);
            std::vector<float> table(config_.projection.materials.size() * weights.size());
            for (std::size_t material = 0;
                material < config_.projection.materials.size(); ++material) {
                const auto& attenuation = model_.massAttenuation(
                    config_.projection.materials[material].label);
                if (attenuation.size() != weights.size())
                    throw std::runtime_error("材料衰减表长度与能谱长度不一致");
                for (std::size_t energy = 0; energy < weights.size(); ++energy)
                    table[material * weights.size() + energy] =
                        static_cast<float>(config_.projection.materials[material].density_g_cm3 *
                            attenuation[energy]);
            }
            if (!cuda_integrator.initialize(weights, energies, table,
                    static_cast<int>(config_.projection.materials.size()), pixels, error))
                return false;
        }
#else
        if (config_.projection.use_cuda_spectral) {
            error = "当前构建未启用 CUDA 多能谱积分，请关闭 use_cuda_spectral 或使用顶层 DLL 构建";
            return false;
        }
#endif
        if (!config_.projection.energy_output_file.empty() && !config_.projection.use_cuda_spectral) {
            error = "Energy-integrating detector output requires use_cuda_spectral=true";
            return false;
        }
        for (int view = 0; view < g.views; ++view) {
            if (!reader.readView(paths, error)) return false;
#if defined(YK_MULTISPECTRUM_CUDA_SPECTRAL)
            if (config_.projection.use_cuda_spectral) {
                const auto flux = config_.projection.apply_geometry_flux ? flatGeometryFlux(config_, view) : std::vector<float>{};
                if (!cuda_integrator.integrate(paths, flux, projection, energy_signal, error))
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
            if (energy_out) {
                energy_out.write(reinterpret_cast<const char*>(energy_signal.data()),
                    static_cast<std::streamsize>(energy_signal.size() * sizeof(float)));
                if (!energy_out) throw std::runtime_error("写入能量积分文件失败");
            }
        }
        reader.close();
        out.close();
        writeProjectionMetadata(output_file, g.detector_u, g.detector_v,
            g.views, "negative_log_transmission");
        if (energy_out) {
            energy_out.close();
            writeProjectionMetadata(config_.projection.energy_output_file,
                g.detector_u, g.detector_v, g.views,
                "transmitted_energy_keV_per_incident_photon");
        }
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}
}
