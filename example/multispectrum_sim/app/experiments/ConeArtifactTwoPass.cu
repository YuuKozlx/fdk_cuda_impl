// Experimental Hsieh-style two-pass cone artifact correction. Not a default stage.
#include "config/SimConfig.hpp"
#include "image/ImageProcessing.hpp"
#include "primary/JosephScalar.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
using namespace yk::spectral;
namespace pd = yk::spectral::primary_detail;
namespace fs = std::filesystem;
using Json = nlohmann::json;
void check(cudaError_t e)
{
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
std::vector<float> readVolume(const fs::path& p, size_t n)
{
    if (fs::file_size(p) != n * sizeof(float))
        throw std::runtime_error("RAW size mismatch: " + p.string());
    std::vector<float> v(n);
    std::ifstream f(p, std::ios::binary);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    for (float x : v)
        if (!std::isfinite(x))
            throw std::runtime_error("Nonfinite RAW input");
    return v;
}
void writeVolume(const fs::path& p, const std::vector<float>& v)
{
    if (fs::exists(p))
        throw std::runtime_error("Output already exists: " + p.string());
    std::ofstream f(p, std::ios::binary);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}
__global__ void project(pd::DeviceGeometry g, const float* volume, int view, float* output)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.nu * g.nv)
        return;
    // Deterministic 2x2 pixel quadrature; no noise, spectrum or focal randomization.
    float sum = 0;
    for (int v = 0; v < 2; ++v)
        for (int u = 0; u < 2; ++u) {
            auto ray = pd::scannerRay(g, view, float(i % g.nu) + u * .5f - .25f, float(i / g.nu) + v * .5f - .25f);
            auto d = pd::add3(ray.detector, pd::scale3(ray.source, -1));
            d = pd::scale3(d, rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z));
            sum += pd::josephScalar(volume, g.nx, g.ny, g.nz, make_float3(g.vx, g.vy, g.vz), pd::phantomInverse(g, ray.source, false), pd::phantomInverse(g, d, true));
        }
    output[i] = sum * .25f;
}
int main(int argc, char** argv)
{
    try {
        if (argc != 3)
            throw std::runtime_error("Usage: multispectrum_cone_two_pass project|combine job.json");
        Json j;
        std::ifstream input(argv[2]);
        input >> j;
        auto config = loadConfig(j.at("config").get<std::string>());
        if (config.geometry.kind != GeometryKind::FlatCbct)
            throw std::runtime_error("Experiment requires flat circular CBCT");
        auto& g = config.geometry.parameters;
        image::Grid grid;
        grid.dimensions = { g.reconstruction_volume_x, g.reconstruction_volume_y, g.reconstruction_volume_z };
        grid.spacing_mm = { g.reconstruction_voxel_x_mm, g.reconstruction_voxel_y_mm, g.reconstruction_voxel_z_mm };
        size_t n = 1;
        for (int a = 0; a < 3; ++a) {
            if (grid.dimensions[a] < 1 || grid.spacing_mm[a] <= 0)
                throw std::runtime_error("Invalid grid");
            n *= grid.dimensions[a];
        }
        fs::path out = j.at("output_directory").get<std::string>();
        fs::create_directories(out);
        const auto start = std::chrono::steady_clock::now();
        std::string mode = argv[1];
        if (mode == "project") {
            if (fs::exists(out / "projection.raw") || fs::exists(out / "bone.raw"))
                throw std::runtime_error("Projection output exists");
            auto volume = readVolume(j.at("input_volume").get<std::string>(), n);
            if (j.value("segment", true)) {
                const double mu = j.at("water_mu_mm_inv"), threshold = j.value("threshold_hu", 225.);
                if (!std::isfinite(mu) || mu <= 0 || !std::isfinite(threshold))
                    throw std::runtime_error("Invalid segmentation");
                for (auto& x : volume)
                    if (x * 1000. / mu - 1000. <= threshold)
                        x = 0;
            }
            writeVolume(out / "bone.raw", volume);
            auto device = image::allocate(image::Location::Device, n);
            image::copy(device.get(), image::Location::Device, volume.data(), image::Location::Host, n);
            g.volume_x = grid.dimensions[0];
            g.volume_y = grid.dimensions[1];
            g.volume_z = grid.dimensions[2];
            g.voxel_x_mm = grid.spacing_mm[0];
            g.voxel_y_mm = grid.spacing_mm[1];
            g.voxel_z_mm = grid.spacing_mm[2];
            g.phantom_offset_x_mm = g.reconstruction_offset_x_mm;
            g.phantom_offset_y_mm = g.reconstruction_offset_y_mm;
            g.phantom_offset_z_mm = g.reconstruction_offset_z_mm;
            g.phantom_rotation_x_rad = g.phantom_rotation_y_rad = g.phantom_rotation_z_rad = 0;
            config.projection.focal_spot.enabled = config.projection.focal_spot.defocus_enabled = false;
            const size_t pixels = size_t(g.detector_u) * g.detector_v;
            const int batch = std::min(g.views, std::max(1, int((256ull << 20) / (pixels * sizeof(float)))));
            auto buffer = image::allocate(image::Location::Device, pixels * batch);
            std::vector<float> host(pixels * batch);
            std::ofstream f(out / "projection.raw", std::ios::binary);
            f.exceptions(std::ios::badbit | std::ios::failbit);
            auto dg = pd::makeDeviceGeometry(config, 1, 1);
            for (int view = 0; view < g.views; view += batch) {
                int count = std::min(batch, g.views - view);
                for (int k = 0; k < count; ++k)
                    project<<<unsigned((pixels + 255) / 256), 256>>>(dg, device.get(), view + k, buffer.get() + k * pixels);
                check(cudaGetLastError());
                image::copy(host.data(), image::Location::Host, buffer.get(), image::Location::Device, pixels * count);
                for (size_t k = 0; k < pixels * count; ++k)
                    if (!std::isfinite(host[k]))
                        throw std::runtime_error("Nonfinite projection");
                f.write(reinterpret_cast<char*>(host.data()), pixels * count * sizeof(float));
                std::cout << "projection " << view + count << '/' << g.views << std::endl;
            }
        } else if (mode == "combine") {
            if (fs::exists(out / "reconstruction.raw"))
                throw std::runtime_error("Correction output exists");
            auto baseline = readVolume(j.at("baseline").get<std::string>(), n);
            auto bone = readVolume(j.at("bone").get<std::string>(), n);
            auto reconstructed = readVolume(j.at("bone_reconstruction").get<std::string>(), n);
            auto smooth = [&](std::vector<float>& data, std::array<double, 3> sigma) {
                image::Options o;
                o.operation = image::Operation::Gaussian;
                o.sigma_mm = sigma;
                for (int a = 0; a < 3; ++a) {
                    if (!std::isfinite(sigma[a]) || sigma[a] <= 0)
                        throw std::runtime_error("Gaussian sigma must be positive");
                    o.radius[a] = int(std::ceil(3 * sigma[a] / grid.spacing_mm[a]));
                }
                image::Image im;
                im.grid = grid;
                im.location = image::Location::Host;
                im.data = data.data();
                image::ensureLocation(im, image::Location::Device);
                image::Workspace w(image::Location::Device, 2 * n * sizeof(float));
                image::process(im, o, w);
                image::copy(data.data(), image::Location::Host, im.data, im.location, n);
            };
            smooth(bone, j.at("psf_sigma_mm").get<std::array<double, 3>>());
            for (size_t k = 0; k < n; ++k)
                reconstructed[k] -= bone[k]; // estimated artifact = R A bone - matched bone
            if (j.contains("error_sigma_mm"))
                smooth(reconstructed, j.at("error_sigma_mm").get<std::array<double, 3>>());
            writeVolume(out / "estimated-artifact.raw", reconstructed);
            for (size_t k = 0; k < n; ++k) {
                baseline[k] -= reconstructed[k];
                if (!std::isfinite(baseline[k]))
                    throw std::runtime_error("Nonfinite correction");
            }
            writeVolume(out / "reconstruction.raw", baseline);
        } else
            throw std::runtime_error("Unknown mode");
        std::ofstream(out / (mode + ".json")) << Json { { "inputs", j }, { "seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() }, { "status", "experimental; quality not yet validated" } }.dump(2);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
