#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "primary/IdealBoneCorrection.cuh"
#include "recon/preprocess/BeamHardeningFit.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

using namespace yk::spectral;
namespace detail = yk::spectral::primary_detail;
namespace pool = yk::spectral::cuda_resources;
using Json = nlohmann::json;
void checkCuda(cudaError_t error)
{
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
template <class T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { checkCuda(pool::poolAllocate(&data, count * sizeof(T))); }
    ~DeviceBuffer() { pool::poolRelease(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    void upload(const T* source, size_t count) { checkCuda(cudaMemcpy(data, source, count * sizeof(T), cudaMemcpyHostToDevice)); }
};

#include "recon/preprocess/CudaPixelOps.hpp"
#include "recon/preprocess/correction/TwoMaterialBhcStage.hpp"
__global__ void bonePaths(detail::DeviceGeometry g, const unsigned char* labels, const int* map, int view, float* out, float scale)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nu * g.nv)
        return;
    const auto ray = detail::scannerRay(g, view, float(i % g.nu), float(i / g.nu));
    float3 d = detail::add3(ray.detector, detail::scale3(ray.source, -1));
    d = detail::scale3(d, rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z));
    const float3 lo = make_float3(-.5f * g.nx * g.vx, -.5f * g.ny * g.vy, -.5f * g.nz * g.vz), hi = detail::scale3(lo, -1);
    float paths[2] = {};
    detail::accumulatePaths<2>(g, labels, map, detail::phantomInverse(g, ray.source, false), detail::phantomInverse(g, d, true), lo, hi, paths);
    out[i] = 10 * paths[1] * scale; // Equivalent second-material thickness in mm.
}
class BoneProvider final : public preprocess::IBoneThicknessProvider {
    detail::DeviceGeometry g_;
    const unsigned char* labels_;
    const int* map_;
    float scale_;

public:
    BoneProvider(detail::DeviceGeometry g, const unsigned char* labels, const int* map, float scale = 1)
        : g_(g)
        , labels_(labels)
        , map_(map)
        , scale_(scale)
    {
    }
    void writeDevice(int view, int width, int height, float* out) override
    {
        bonePaths<<<(width * height + 127) / 128, 128>>>(g_, labels_, map_, view, out, scale_);
        checkCuda(cudaGetLastError());
    }
};

#include "recon/preprocess/correction/EmpiricalBhcStage.hpp"
__global__ void basis(float* p, const float* paths, size_t n, bool cross)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float q = paths[i] / 100.f;
        p[i] = cross ? p[i] * q : q * q;
    }
}
int main(int argc, char** argv)
{
    try {
        if (argc != 6 && argc != 7)
            throw std::runtime_error("Usage: basis|apply|apply-candidate config.toml segmentation.raw baseline-log.raw output-dir [coefficients.json]");
        const bool candidate = std::string(argv[1]) == "apply-candidate";
        const bool apply = std::string(argv[1]) == "apply" || candidate;
        if ((apply && argc != 7) || (!apply && (argc != 6 || std::string(argv[1]) != "basis")))
            throw std::runtime_error("Invalid mode");
        auto config = loadConfig(argv[2]);
        auto& g = config.geometry.parameters;
        if (config.geometry.kind != GeometryKind::FlatHelical && config.geometry.kind != GeometryKind::CylHelical && config.geometry.kind != GeometryKind::FlatCbct)
            throw std::runtime_error("Unsupported EBHC geometry");
        g.volume_x = g.reconstruction_volume_x;
        g.volume_y = g.reconstruction_volume_y;
        g.volume_z = g.reconstruction_volume_z;
        g.voxel_x_mm = g.reconstruction_voxel_x_mm;
        g.voxel_y_mm = g.reconstruction_voxel_y_mm;
        g.voxel_z_mm = g.reconstruction_voxel_z_mm;
        g.phantom_offset_x_mm = g.reconstruction_offset_x_mm;
        g.phantom_offset_y_mm = g.reconstruction_offset_y_mm;
        g.phantom_offset_z_mm = g.reconstruction_offset_z_mm;
        g.phantom_rotation_x_rad = g.phantom_rotation_y_rad = g.phantom_rotation_z_rad = 0;
        config.projection.focal_spot.enabled = config.projection.focal_spot.defocus_enabled = false;
        const size_t voxels = size_t(g.volume_x) * g.volume_y * g.volume_z, pixels = size_t(g.detector_u) * g.detector_v;
        if (std::filesystem::file_size(argv[3]) != voxels || std::filesystem::file_size(argv[4]) != pixels * g.views * 4)
            throw std::runtime_error("Grid size mismatch");
        const std::filesystem::path out = argv[5];
        std::filesystem::create_directories(out);
        const auto first = out / (apply ? "corrected.raw" : "square.raw");
        if (std::filesystem::exists(first) || (!apply && std::filesystem::exists(out / "cross.raw")))
            throw std::runtime_error("Outputs exist");
        auto labels = loadLabelVolume(argv[3]);
        for (auto v : labels)
            if (v > 2)
                throw std::runtime_error("Expected labels 0 air, 1 tissue, 2 bone");
        DeviceBuffer<unsigned char> dl(voxels);
        dl.upload(labels.data(), voxels);
        std::vector<int> map(256, -1);
        map[0] = map[1] = 0;
        map[2] = 1;
        DeviceBuffer<int> dm(256);
        dm.upload(map.data(), 256);
        auto provider = std::make_shared<BoneProvider>(detail::makeDeviceGeometry(config, 2, 1), dl.data, dm.data);
        preprocess::EmpiricalBhcParameters parameters;
        if (apply) {
            std::ifstream f(argv[6]);
            Json j;
            f >> j;
            if (j.at("schema") != "ebhc-path-v1")
                throw std::runtime_error("Wrong coefficient schema");
            if (!candidate && !j.value("accepted", false))
                throw std::runtime_error("EBHC fit rejected or unvalidated; use apply-candidate only for explicit diagnostic experiments");
            parameters.square = j.at("square");
            parameters.cross = j.at("cross");
            parameters.path_scale_mm = j.at("path_scale_mm");
        }
        auto stage = preprocess::makeEmpiricalBhcStage(parameters, provider);
        preprocess::CudaPixelOps ops(1);
        TaskProgressCallback progress;
        preprocess::Context context { config, ops, progress, first };
        if (apply)
            stage->prepare(context);
        const int capacity = std::min(g.views, std::max(1, int((256ULL << 20) / (pixels * 4))));
        DeviceBuffer<float> d(pixels * capacity), paths(pixels);
        std::vector<float> host(pixels * capacity), original;
        std::ifstream input(argv[4], std::ios::binary);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        std::ofstream o1(first, std::ios::binary), o2;
        o1.exceptions(std::ios::badbit | std::ios::failbit);
        if (!apply) {
            o2.open(out / "cross.raw", std::ios::binary);
            o2.exceptions(std::ios::badbit | std::ios::failbit);
        }
        const auto start = std::chrono::steady_clock::now();
        for (int v = 0; v < g.views; v += capacity) {
            const int count = std::min(capacity, g.views - v);
            const size_t n = pixels * count;
            input.read(reinterpret_cast<char*>(host.data()), n * 4);
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(host[i]))
                    throw std::runtime_error("Nonfinite baseline projection");
            if (!apply)
                original = host;
            for (int pass = 0; pass < (apply ? 1 : 2); ++pass) {
                d.upload(apply ? host.data() : original.data(), n);
                for (int j = 0; j < count; ++j) {
                    if (apply) {
                        preprocess::Frame f { v + j, g.detector_u, g.detector_v, pixels, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, d.data + j * pixels, nullptr };
                        stage->process(f, context);
                    } else {
                        provider->writeDevice(v + j, g.detector_u, g.detector_v, paths.data);
                        basis<<<unsigned((pixels + 255) / 256), 256>>>(d.data + j * pixels, paths.data, pixels, pass == 1);
                        checkCuda(cudaGetLastError());
                    }
                }
                checkCuda(cudaMemcpy(host.data(), d.data, n * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < n; ++i)
                    if (!std::isfinite(host[i]))
                        throw std::runtime_error("Nonfinite EBHC output");
                (pass == 0 ? o1 : o2).write(reinterpret_cast<const char*>(host.data()), n * 4);
            }
            std::cout << "EBHC " << v + count << '/' << g.views << std::endl;
        }
        Json report = { { "mode", argv[1] }, { "seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() },
            { "baseline_log", std::filesystem::absolute(argv[4]).generic_string() }, { "segmentation", std::filesystem::absolute(argv[3]).generic_string() },
            { "path_scale_mm", parameters.path_scale_mm }, { "square", parameters.square }, { "cross", parameters.cross }, { "random_rays", false } };
        std::ofstream(out / (apply ? "apply.json" : "basis.json")) << report.dump(2);
        std::cout << report.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
