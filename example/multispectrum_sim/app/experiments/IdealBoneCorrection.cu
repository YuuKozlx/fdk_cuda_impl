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
void selfTest()
{
    detail::DeviceGeometry g {};
    g.nu = 3;
    g.nv = 1;
    g.views = 1;
    g.views_per_turn = 360;
    g.rotation_direction = 1;
    g.nx = g.ny = g.nz = 4;
    g.vx = g.vy = g.vz = 1;
    g.pu = g.pv = .5f;
    g.sid = 709;
    g.sdd = 1143;
    g.nm = 3;
    g.ne = 2;
    std::vector<int> map(256, -1);
    map[0] = 0;
    map[1] = 1;
    map[2] = 2;
    const double mu[] = { 0, 0, .1, .2, .3, .5 }, mono[] = { 0, .15, .4 }, weights[] = { .3, .7 };
    const double coefficients[] = { .05, 1.1, .02 };
    DeviceBuffer<unsigned char> labels(64);
    DeviceBuffer<int> device_map(256);
    DeviceBuffer<double> device_mu(6), device_mono(3), device_weights(2), device_coefficients(3);
    DeviceBuffer<float> measured(3), corrected(3), residual(3);
    device_map.upload(map.data(), 256);
    device_mu.upload(mu, 6);
    device_mono.upload(mono, 3);
    device_weights.upload(weights, 2);
    device_coefficients.upload(coefficients, 3);
    const float input[] = { 5, 5, 5 };
    measured.upload(input, 3);
    for (int test = 0; test < 4; ++test) {
        std::vector<unsigned char> source(64, static_cast<unsigned char>(std::min(test, 2)));
        if (test == 3)
            for (int z = 0; z < 4; ++z)
                for (int y = 0; y < 4; ++y)
                    for (int x = 0; x < 4; ++x)
                        source[(z * 4 + y) * 4 + x] = y < 2 ? 1 : 2;
        labels.upload(source.data(), 64);
        detail::idealBoneCorrectionKernel<<<1, 32>>>(g, labels.data, device_map.data, device_mu.data,
            device_weights.data, device_mono.data, device_coefficients.data, 3, -100, 100, 0,
            measured.data, corrected.data, residual.data);
        checkCuda(cudaGetLastError());
        float result[3], delta[3];
        checkCuda(cudaMemcpy(result, corrected.data, sizeof(result), cudaMemcpyDeviceToHost));
        checkCuda(cudaMemcpy(delta, residual.data, sizeof(delta), cudaMemcpyDeviceToHost));
        const double water_path = test == 1 ? .4 : test == 3 ? .2
                                                             : 0;
        const double bone_path = test == 2 ? .4 : test == 3 ? .2
                                                            : 0;
        const double p = -std::log(.3 * std::exp(-.1 * water_path - .3 * bone_path) + .7 * std::exp(-.2 * water_path - .5 * bone_path));
        const double expected = .15 * water_path + .4 * bone_path - (.05 + 1.1 * p + .02 * p * p);
        if (std::abs(delta[1] - expected) > 2e-6 || std::abs(result[1] - (5 + expected)) > 2e-6)
            throw std::runtime_error("Ideal bone correction CUDA/analytic mismatch, test=" + std::to_string(test));
    }
    std::cout << "singlepass_tests=4 passed (air, water, bone, mixed; nonlinear water polynomial)\n";
}
int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            selfTest();
            return 0;
        }
        if (argc != 5)
            throw std::runtime_error("Usage: multispectrum_bone_bhc_singlepass baseline.toml segmented-u8.raw corrected-water-input.raw output-directory");
        auto config = loadConfig(argv[1]);
        if (config.geometry.kind != GeometryKind::FlatHelical && config.geometry.kind != GeometryKind::CylHelical)
            throw std::runtime_error("Single-pass experiment currently supports flat/cyl helical geometry");
        const auto bhc_input = config.reconstruction.preprocessing.beam_hardening;
        if (!bhc_input.enabled)
            throw std::runtime_error("Baseline water BHC must be enabled");
        auto bhc = bhc_input.fit_from_water_plate ? preprocess::fitWaterPlate(config, bhc_input, {}) : bhc_input;
        const auto reference = resolveWaterReference(config, bhc_input);
        const auto spectrum = bhc_input.spectrum_file.empty() ? loadProjectionSpectrum(config.projection) : loadSpectrum(bhc_input.spectrum_file);
        auto& geom = config.geometry.parameters;
        geom.volume_x = geom.reconstruction_volume_x;
        geom.volume_y = geom.reconstruction_volume_y;
        geom.volume_z = geom.reconstruction_volume_z;
        geom.voxel_x_mm = geom.reconstruction_voxel_x_mm;
        geom.voxel_y_mm = geom.reconstruction_voxel_y_mm;
        geom.voxel_z_mm = geom.reconstruction_voxel_z_mm;
        geom.phantom_offset_x_mm = geom.reconstruction_offset_x_mm;
        geom.phantom_offset_y_mm = geom.reconstruction_offset_y_mm;
        geom.phantom_offset_z_mm = geom.reconstruction_offset_z_mm;
        geom.phantom_rotation_x_rad = geom.phantom_rotation_y_rad = geom.phantom_rotation_z_rad = 0;
        config.projection.focal_spot.enabled = false;
        config.projection.focal_spot.defocus_enabled = false;
        config.projection.materials.clear();
        MaterialSpec air;
        air.label = 0;
        air.name = "air";
        air.preset = "mixture.air";
        air.density_g_cm3 = .001205;
        MaterialSpec water;
        water.label = 1;
        water.name = "soft-water";
        water.preset = "compound.water";
        water.density_g_cm3 = 1.03;
        MaterialSpec bone;
        bone.label = 2;
        bone.name = "bone";
        bone.preset = "tissue.ncat_skull";
        bone.density_g_cm3 = 1.61;
        config.projection.materials = { air, water, bone };
        const int ne = int(spectrum.size());
        if (ne <= 0 || ne > 256 || bhc.coefficients.empty() || bhc.coefficients.size() > 17)
            throw std::runtime_error("Invalid spectrum or BHC coefficient count");
        std::vector<double> energies, weights(ne), mu(3 * ne), mono_mu(3);
        double total_weight = 0;
        for (int e = 0; e < ne; ++e) {
            energies.push_back(spectrum[e].energy_keV);
            weights[e] = spectrum[e].fluence_per_keV_cm2_mAs_at_1m * spectrum[e].bin_width_keV * spectrum[e].energy_keV;
            total_weight += weights[e];
        }
        if (!std::isfinite(total_weight) || total_weight <= 0)
            throw std::runtime_error("Invalid spectral energy weights");
        for (auto& w : weights)
            w /= total_weight;
        XcomAttenuationProvider attenuation({});
        for (int m = 0; m < 3; ++m) {
            std::vector<double> mass, reference_mass;
            std::string error;
            if (!attenuation.query(config.projection.materials[m], energies, mass, error) || !attenuation.query(config.projection.materials[m], { reference.energy_keV }, reference_mass, error))
                throw std::runtime_error(error);
            for (int e = 0; e < ne; ++e)
                mu[m * ne + e] = mass[e] * config.projection.materials[m].density_g_cm3;
            mono_mu[m] = reference_mass[0] * config.projection.materials[m].density_g_cm3;
        }
        std::vector<int> map(256, -1);
        for (int m = 0; m < 3; ++m)
            map[m] = m;
        const size_t voxels = size_t(geom.volume_x) * geom.volume_y * geom.volume_z;
        if (std::filesystem::file_size(argv[2]) != voxels)
            throw std::runtime_error("Segmented grid size mismatch");
        auto labels = loadLabelVolume(argv[2]);
        for (auto label : labels)
            if (label > 2)
                throw std::runtime_error("Unexpected segmented material label");
        const size_t pixels = size_t(geom.detector_u) * geom.detector_v;
        if (std::filesystem::file_size(argv[3]) != pixels * geom.views * sizeof(float))
            throw std::runtime_error("Measured projection size mismatch");
        const int capacity = std::min(geom.views, std::max(1, int((256ULL << 20) / (3 * pixels * sizeof(float)))));
        const size_t elements = pixels * capacity;
        std::vector<float> input(elements), output(elements), delta(elements);
        DeviceBuffer<unsigned char> dlabels(voxels);
        dlabels.upload(labels.data(), voxels);
        DeviceBuffer<int> dmap(256);
        dmap.upload(map.data(), 256);
        DeviceBuffer<double> dmu(mu.size()), dmono(3), dw(ne), dcoeff(bhc.coefficients.size());
        dmu.upload(mu.data(), mu.size());
        dmono.upload(mono_mu.data(), 3);
        dw.upload(weights.data(), ne);
        dcoeff.upload(bhc.coefficients.data(), bhc.coefficients.size());
        DeviceBuffer<float> din(elements), dout(elements), ddelta(elements);
        const std::filesystem::path directory(argv[4]);
        std::filesystem::create_directories(directory);
        std::ifstream measured(argv[3], std::ios::binary);
        measured.exceptions(std::ios::failbit | std::ios::badbit);
        std::ofstream corrected(directory / "bone-corrected.raw", std::ios::binary), residual(directory / "model-residual.raw", std::ios::binary);
        corrected.exceptions(std::ios::failbit | std::ios::badbit);
        residual.exceptions(std::ios::failbit | std::ios::badbit);
        const auto dg = detail::makeDeviceGeometry(config, 3, ne);
        const auto start = std::chrono::steady_clock::now();
        double kernel_ms = 0;
        float minimum = 1e30f, maximum = -1e30f;
        cudaEvent_t before, after;
        checkCuda(cudaEventCreate(&before));
        checkCuda(cudaEventCreate(&after));
        for (int view = 0; view < geom.views; view += capacity) {
            const int count = std::min(capacity, geom.views - view);
            const size_t n = pixels * count;
            measured.read(reinterpret_cast<char*>(input.data()), n * sizeof(float));
            din.upload(input.data(), n);
            checkCuda(cudaEventRecord(before));
            detail::idealBoneCorrectionKernel<<<dim3(unsigned((pixels + 127) / 128), unsigned(count)), 128>>>(dg, dlabels.data, dmap.data, dmu.data, dw.data, dmono.data, dcoeff.data,
                int(bhc.coefficients.size()), bhc.clamp_min, bhc.clamp_max, view, din.data, dout.data, ddelta.data);
            checkCuda(cudaGetLastError());
            checkCuda(cudaEventRecord(after));
            checkCuda(cudaEventSynchronize(after));
            float elapsed = 0;
            checkCuda(cudaEventElapsedTime(&elapsed, before, after));
            kernel_ms += elapsed;
            checkCuda(cudaMemcpy(output.data(), dout.data, n * sizeof(float), cudaMemcpyDeviceToHost));
            checkCuda(cudaMemcpy(delta.data(), ddelta.data, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(output[i]) || !std::isfinite(delta[i]))
                    throw std::runtime_error("Nonfinite bone correction output");
                minimum = std::min(minimum, delta[i]);
                maximum = std::max(maximum, delta[i]);
            }
            corrected.write(reinterpret_cast<const char*>(output.data()), n * sizeof(float));
            residual.write(reinterpret_cast<const char*>(delta.data()), n * sizeof(float));
            std::cout << "bone_singlepass views=" << view + count << '/' << geom.views << std::endl;
        }
        corrected.close();
        residual.close();
        cudaEventDestroy(before);
        cudaEventDestroy(after);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Json info = { { "method", "one ideal central ray; one Siddon material traversal; shared poly/mono paths" },
            { "reference_energy_keV", reference.energy_keV }, { "water_bhc_coefficients", bhc.coefficients },
            { "segmentation", std::filesystem::absolute(argv[2]).generic_string() }, { "measured_water_corrected", std::filesystem::absolute(argv[3]).generic_string() },
            { "correction_formula", "measured + mono(path) - Q(poly(path))" }, { "nonideal_model_effects", false },
            { "batch_views", capacity }, { "kernel_ms", kernel_ms }, { "processing_seconds", seconds },
            { "delta_min", minimum }, { "delta_max", maximum }, { "source_grid_xyz", { geom.volume_x, geom.volume_y, geom.volume_z } },
            { "projection_shape_vvu", { geom.views, geom.detector_v, geom.detector_u } } };
        std::ofstream(directory / "singlepass.json") << info.dump(2);
        std::cout << info.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
