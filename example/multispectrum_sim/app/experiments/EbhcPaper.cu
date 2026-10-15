#include "config/SimConfig.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "primary/JosephScalar.cuh"
#include "recon/postprocess/stages/correction/EmpiricalBoneCorrection.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace yk::spectral;
namespace detail = yk::spectral::primary_detail;
namespace pp = yk::spectral::postprocess;
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    float* p = nullptr;
    explicit Buffer(size_t n) { check(cuda_resources::poolAllocate(&p, n * 4)); }
    ~Buffer() { cuda_resources::poolRelease(p); }
    void upload(const float* v, size_t n) { check(cudaMemcpy(p, v, n * 4, cudaMemcpyHostToDevice)); }
};
std::vector<float> read(const fs::path& p, size_t n)
{
    if (fs::file_size(p) != n * 4)
        throw std::runtime_error("Size mismatch: " + p.string());
    std::ifstream in(p, std::ios::binary);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    std::vector<float> v(n);
    in.read(reinterpret_cast<char*>(v.data()), n * 4);
    for (float x : v)
        if (!std::isfinite(x))
            throw std::runtime_error("Nonfinite input");
    return v;
}
void write(const fs::path& p, const std::vector<float>& v)
{
    std::ofstream o(p, std::ios::binary);
    o.exceptions(std::ios::badbit | std::ios::failbit);
    o.write(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}
__global__ void project(detail::DeviceGeometry g, const float* bone, int view, const float* p0, float* cross, float* square)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nu * g.nv)
        return;
    auto ray = detail::scannerRay(g, view, float(i % g.nu), float(i / g.nu));
    auto d = detail::add3(ray.detector, detail::scale3(ray.source, -1));
    d = detail::scale3(d, rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z));
    float p2 = detail::josephScalar(bone, g.nx, g.ny, g.nz, make_float3(g.vx, g.vy, g.vz), detail::phantomInverse(g, ray.source, false), detail::phantomInverse(g, d, true));
    cross[i] = p0[i] * p2;
    square[i] = p2 * p2;
}
int main(int argc, char** argv)
{
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: multispectrum_ebhc_paper basis|fit experiment.json");
        std::ifstream input(argv[2]);
        Json job;
        input >> job;
        const fs::path out = job.at("output_directory").get<std::string>();
        fs::create_directories(out);
        const auto start = std::chrono::steady_clock::now();
        auto dims = job.at("dimensions_xyz").get<std::array<int, 3>>();
        size_t n = 1;
        for (int v : dims) {
            if (v < 2)
                throw std::runtime_error("Invalid dimension");
            n *= v;
        }
        const std::string mode = argv[1];
        if (mode == "basis") {
            if (fs::exists(out / "bone.raw") || fs::exists(out / "cross.raw") || fs::exists(out / "square.raw"))
                throw std::runtime_error("Basis output exists");
            auto config = loadConfig(job.at("config").get<std::string>());
            auto& g = config.geometry.parameters;
            if (config.geometry.kind != GeometryKind::FlatHelical && config.geometry.kind != GeometryKind::CylHelical && config.geometry.kind != GeometryKind::FlatCbct)
                throw std::runtime_error("Unsupported geometry");
            if (dims != std::array<int, 3> { g.reconstruction_volume_x, g.reconstruction_volume_y, g.reconstruction_volume_z })
                throw std::runtime_error("Grid mismatch");
            const double mu = job.at("water_mu_mm_inv"), low = job.at("threshold_low_hu"), high = job.at("threshold_high_hu");
            if (!std::isfinite(mu) || mu <= 0 || !std::isfinite(low) || !std::isfinite(high) || high <= low)
                throw std::runtime_error("Invalid segmentation parameters");
            auto bone = read(job.at("baseline").get<std::string>(), n);
            for (auto& v : bone) {
                const double hu = v * 1000 / mu - 1000;
                v = float(v * std::clamp((hu - low) / (high - low), 0., 1.));
            }
            write(out / "bone.raw", bone);
            Buffer db(n);
            db.upload(bone.data(), n);
            g.volume_x = dims[0];
            g.volume_y = dims[1];
            g.volume_z = dims[2];
            g.voxel_x_mm = g.reconstruction_voxel_x_mm;
            g.voxel_y_mm = g.reconstruction_voxel_y_mm;
            g.voxel_z_mm = g.reconstruction_voxel_z_mm;
            g.phantom_offset_x_mm = g.reconstruction_offset_x_mm;
            g.phantom_offset_y_mm = g.reconstruction_offset_y_mm;
            g.phantom_offset_z_mm = g.reconstruction_offset_z_mm;
            g.phantom_rotation_x_rad = g.phantom_rotation_y_rad = g.phantom_rotation_z_rad = 0;
            config.projection.focal_spot.enabled = config.projection.focal_spot.defocus_enabled = false;
            const size_t pixels = size_t(g.detector_u) * g.detector_v;
            const fs::path projection = job.at("baseline_projection").get<std::string>();
            if (fs::file_size(projection) != pixels * g.views * 4)
                throw std::runtime_error("Projection size mismatch");
            const int batch = std::min(g.views, std::max(1, int((256ULL << 20) / (3 * pixels * 4))));
            Buffer dp(pixels * batch), dc(pixels * batch), ds(pixels * batch);
            std::vector<float> host(pixels * batch);
            std::ifstream pin(projection, std::ios::binary);
            pin.exceptions(std::ios::badbit | std::ios::failbit);
            std::ofstream co(out / "cross.raw", std::ios::binary), so(out / "square.raw", std::ios::binary);
            co.exceptions(std::ios::badbit | std::ios::failbit);
            so.exceptions(std::ios::badbit | std::ios::failbit);
            auto dg = detail::makeDeviceGeometry(config, 1, 1);
            for (int view = 0; view < g.views; view += batch) {
                int count = std::min(batch, g.views - view);
                size_t elements = count * pixels;
                pin.read(reinterpret_cast<char*>(host.data()), elements * 4);
                dp.upload(host.data(), elements);
                for (int j = 0; j < count; ++j)
                    project<<<unsigned((pixels + 255) / 256), 256>>>(dg, db.p, view + j, dp.p + j * pixels, dc.p + j * pixels, ds.p + j * pixels);
                check(cudaGetLastError());
                check(cudaMemcpy(host.data(), dc.p, elements * 4, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < elements; ++i)
                    if (!std::isfinite(host[i]))
                        throw std::runtime_error("Nonfinite projection");
                co.write(reinterpret_cast<char*>(host.data()), elements * 4);
                check(cudaMemcpy(host.data(), ds.p, elements * 4, cudaMemcpyDeviceToHost));
                so.write(reinterpret_cast<char*>(host.data()), elements * 4);
                std::cout << "basis " << view + count << '/' << g.views << std::endl;
            }
            Json report = { { "method", "soft attenuation bone; dominant-axis bilinear Joseph" }, { "inputs", job }, { "seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() } };
            std::ofstream(out / "basis.json") << report.dump(2);
            return 0;
        }
        if (mode != "fit")
            throw std::runtime_error("Invalid mode");
        if (fs::exists(out / "reconstruction.raw"))
            throw std::runtime_error("Fit output exists");
        auto baseline = read(job.at("baseline").get<std::string>(), n);
        pp::EbhcBasis b;
        const std::array<std::string, 3> keys = { "bone", "cross", "square" };
        for (int k = 0; k < 3; ++k)
            b[k] = std::make_shared<const std::vector<float>>(read(job.at(keys[k]).get<std::string>(), n));
        pp::Grid grid;
        grid.dimensions = dims;
        grid.spacing_mm = job.at("spacing_mm").get<std::array<double, 3>>();
        auto fit = pp::fitEbhcFullVolumeTv(baseline, b, grid, [](int i, double cost) { std::cout << "TV evaluation=" << i << " cost=" << cost << std::endl; });
        // Apply the exact paper image formula, including the original segmented f01.
        // R(A(f01)) is not substituted for f01, so this is an image-domain Stage.
        Buffer dv(n);
        dv.upload(baseline.data(), n);
        auto stage = pp::makeEbhcVolumeStage(fit.coefficients, b, grid);
        pp::Context ctx;
        ctx.grid = grid;
        stage->prepare(ctx);
        pp::Volume v;
        v.grid = grid;
        v.elements = n;
        v.device_data = dv.p;
        v.location = pp::Location::Device;
        stage->process(v, ctx);
        check(cudaMemcpy(baseline.data(), dv.p, n * 4, cudaMemcpyDeviceToHost));
        stage->reset();
        write(out / "reconstruction.raw", baseline);
        Json report = { { "schema", "ebhc-three-term-tv-v1" }, { "linear", fit.coefficients.linear }, { "cross", fit.coefficients.cross }, { "square", fit.coefficients.square }, { "converged", fit.converged }, { "evaluations", fit.evaluations }, { "initial_tv", fit.initial_tv }, { "final_tv", fit.final_tv }, { "trace", fit.trace }, { "inputs", job }, { "seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() } };
        std::ofstream(out / "fit.json") << report.dump(2);
        std::cout << "Final coefficients " << fit.coefficients.linear << ' ' << fit.coefficients.cross << ' ' << fit.coefficients.square << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
