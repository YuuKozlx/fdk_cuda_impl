#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "primary/IdealBoneCorrection.cuh"
#include "primary/QuantumSampling.cuh"
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
__global__ void phantomPaths(detail::DeviceGeometry g, const unsigned char* labels, const int* map, int view, unsigned seed, float2* out)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= g.nu)
        return;
    // Reproduce the single simulated detector sample exactly; correction remains central-ray.
    const unsigned pixel = unsigned(g.nv / 2) * unsigned(g.nu) + unsigned(u);
    const unsigned base = seed ^ (unsigned(view) * unsigned(g.nu * g.nv) + pixel) * 0x9e3779b9U;
    const auto ray = detail::scannerRay(g, view, float(u), float(g.nv / 2), 0, 0, detail::uniform(base + 3) - .5f, detail::uniform(base + 4) - .5f);
    float3 d = detail::add3(ray.detector, detail::scale3(ray.source, -1));
    d = detail::scale3(d, rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z));
    const float3 lo = make_float3(-.5f * g.nx * g.vx, -.5f * g.ny * g.vy, -.5f * g.nz * g.vz);
    float p[3] = {};
    detail::accumulatePaths<3>(g, labels, map, detail::phantomInverse(g, ray.source, false), detail::phantomInverse(g, d, true), lo, detail::scale3(lo, -1), p);
    out[u] = make_float2(10 * p[1], 10 * p[2]);
}
int collectPhantom(const char* configPath, const char* output)
{
    const auto config = loadConfig(configPath);
    const auto& g = config.geometry.parameters;
    if (config.projection.sampling.samples_per_pixel != 1 || config.projection.focal_spot.enabled || config.projection.focal_spot.defocus_enabled)
        throw std::runtime_error("CAD sample collector requires one detector sample and no focal sampling");
    auto labels = loadLabelVolume(config.projection.label_volume);
    if (labels.size() != size_t(g.volume_x) * g.volume_y * g.volume_z)
        throw std::runtime_error("Phantom size mismatch");
    for (auto x : labels)
        if (x > 2)
            throw std::runtime_error("Calibration requires labels 0,1,2");
    DeviceBuffer<unsigned char> dl(labels.size());
    dl.upload(labels.data(), labels.size());
    std::vector<int> map(256, -1);
    map[0] = 0;
    map[1] = 1;
    map[2] = 2;
    DeviceBuffer<int> dm(256);
    dm.upload(map.data(), 256);
    DeviceBuffer<float2> dp(g.detector_u);
    std::vector<float2> paths(g.detector_u);
    std::vector<float> pixels(g.detector_u);
    std::ifstream in(config.projection.output_file, std::ios::binary);
    in.exceptions(std::ios::failbit | std::ios::badbit);
    if (std::filesystem::exists(output))
        throw std::runtime_error("Sample output exists");
    std::ofstream out(output);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.precision(9);
    out << "view,first_mm,second_mm,measured_log\n";
    const auto dg = detail::makeDeviceGeometry(config, 3, 1);
    for (int v = 0; v < g.views; ++v) {
        phantomPaths<<<(g.detector_u + 127) / 128, 128>>>(dg, dl.data, dm.data, v, config.projection.sampling.seed, dp.data);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(paths.data(), dp.data, g.detector_u * sizeof(float2), cudaMemcpyDeviceToHost));
        in.seekg((size_t(v) * g.detector_u * g.detector_v + size_t(g.detector_v / 2) * g.detector_u) * sizeof(float));
        in.read(reinterpret_cast<char*>(pixels.data()), pixels.size() * sizeof(float));
        for (int u = 0; u < g.detector_u; ++u)
            if (paths[u].x + paths[u].y > .5f)
                out << v << ',' << paths[u].x << ',' << paths[u].y << ',' << pixels[u] << '\n';
    }
    Json metadata = { { "label_volume", config.projection.label_volume.generic_string() }, { "measured_projection", config.projection.output_file.generic_string() },
        { "thickness_source", "provided calibration-phantom label volume" }, { "ray_matching", "same single detector sample as simulated measurement; no focal sampling" } };
    std::ofstream(std::string(output) + ".json") << metadata.dump(2);
    return 0;
}
int exportMaterialReference(const char* configPath, const char* output)
{
    const auto config=loadConfig(configPath);
    if(config.projection.detector_response.enabled)throw std::runtime_error("Reference export requires energy-integrating detector without added response");
    const auto spectrum=loadProjectionSpectrum(config.projection);
    std::vector<double> energies,weights;
    double sum=0;
    for(const auto& e:spectrum){energies.push_back(e.energy_keV);weights.push_back(e.energy_keV*e.bin_width_keV*e.fluence_per_keV_cm2_mAs_at_1m);sum+=weights.back();}
    if(!(sum>0))throw std::runtime_error("Invalid spectrum");
    for(auto& w:weights)w/=sum;
    Json result={{"purpose","analytic evaluation only; not input to empirical fitter"},{"energy_keV",energies},{"energy_weights",weights}};
    XcomAttenuationProvider provider({});
    for(int label=1;label<=2;++label){
        const auto it=std::find_if(config.projection.materials.begin(),config.projection.materials.end(),[&](const auto& m){return m.label==label;});
        if(it==config.projection.materials.end())throw std::runtime_error("Missing material label");
        std::vector<double> mu;std::string error;
        if(!provider.query(*it,energies,mu,error))throw std::runtime_error(error);
        for(auto& m:mu)m*=it->density_g_cm3/10;
        result[label==1?"first":"second"]={{"preset",it->preset},{"density_g_cm3",it->density_g_cm3},{"mu_mm",mu}};
    }
    std::ofstream out(output);out.exceptions(std::ios::failbit|std::ios::badbit);out<<result.dump(2);
    return 0;
}
int main(int argc, char** argv)
{
    try {
        if(argc==4 && std::string(argv[1])=="--export-material-reference")return exportMaterialReference(argv[2],argv[3]);
        if (argc == 4 && std::string(argv[1]) == "--collect-phantom")
            return collectPhantom(argv[2], argv[3]);
        const bool external = argc == 8 && std::string(argv[1]) == "--apply-table";
        if (argc != 5 && !external)
            throw std::runtime_error("Usage: baseline.toml segmented-u8.raw log.raw output-dir OR --apply-table config.toml labels.raw log.raw calibration.json second-material-scale output-dir OR --collect-phantom config.toml samples.csv");
        const char* configPath = argv[external ? 2 : 1];
        const char* segmentationPath = argv[external ? 3 : 2];
        const char* measuredPath = argv[external ? 4 : 3];
        const char* outputDirectory = argv[external ? 7 : 4];
        const float thicknessScale = external ? std::stof(argv[6]) : 1.f;
        if (!std::isfinite(thicknessScale) || thicknessScale <= 0)
            throw std::runtime_error("Invalid equivalent thickness scale");
        auto config = loadConfig(configPath);
        if (config.geometry.kind != GeometryKind::FlatHelical && config.geometry.kind != GeometryKind::CylHelical && config.geometry.kind != GeometryKind::FlatCbct)
            throw std::runtime_error("Experiment supports flat CBCT and flat/cyl helical scans");
        preprocess::TwoMaterialCalibration model;
        Json calibrationReport;
        if (external) {
            model = preprocess::TwoMaterialCalibration::load(argv[5]);
            calibrationReport = { { "loaded_table", std::filesystem::absolute(argv[5]).generic_string() }, { "equivalent_second_material_scale", thicknessScale } };
        } else {
            const auto reference = resolveWaterReference(config);
            if (config.projection.detector_response.enabled)
                throw std::runtime_error("Experiment spectral calibration requires energy-integrating detector without additional response");
            auto spectrum = loadProjectionSpectrum(config.projection);
            MaterialSpec water;
            water.name = "soft-water";
            water.preset = "compound.water";
            water.density_g_cm3 = 1.03;
            MaterialSpec bone;
            bone.name = "skull";
            bone.preset = "tissue.ncat_skull";
            bone.density_g_cm3 = 1.61;
            XcomAttenuationProvider attenuation({});
            std::vector<double> energies, weights, mw, mb, rw, rb;
            double sum = 0;
            for (const auto& e : spectrum) {
                energies.push_back(e.energy_keV);
                weights.push_back(e.fluence_per_keV_cm2_mAs_at_1m * e.bin_width_keV * e.energy_keV);
                sum += weights.back();
            }
            if (!(sum > 0))
                throw std::runtime_error("Invalid spectrum");
            for (auto& w : weights)
                w /= sum;
            std::string error;
            if (!attenuation.query(water, energies, mw, error) || !attenuation.query(bone, energies, mb, error) || !attenuation.query(water, { reference.energy_keV }, rw, error) || !attenuation.query(bone, { reference.energy_keV }, rb, error))
                throw std::runtime_error(error);
            for (auto& x : mw)
                x *= water.density_g_cm3 / 10;
            for (auto& x : mb)
                x *= bone.density_g_cm3 / 10;
            preprocess::TwoMaterialSamples samples;
            const auto& grid = config.geometry.parameters;
            const double bone_limit = std::ceil(std::sqrt(std::pow(grid.reconstruction_volume_x * grid.reconstruction_voxel_x_mm, 2) + std::pow(grid.reconstruction_volume_y * grid.reconstruction_voxel_y_mm, 2) + std::pow(grid.reconstruction_volume_z * grid.reconstruction_voxel_z_mm, 2)));
            samples.bone_rows = int(std::ceil(bone_limit / .25)) + 1;
            samples.water_columns = 351;
            samples.bone_step_mm = .25;
            samples.water_step_mm = 1;
            samples.target_water_mu_mm = rw[0] * water.density_g_cm3 / 10;
            samples.target_bone_mu_mm = rb[0] * bone.density_g_cm3 / 10;
            samples.reference_energy_keV = reference.energy_keV;
            samples.provenance = "synthetic spectrum-weighted water(1.03g/cc)/ncat_skull(1.61g/cc), no scatter/noise/focal jitter; monotone LUT variant, not original five-parameter empirical fit";
            for (int b = 0; b < samples.bone_rows; ++b)
                for (int w = 0; w < samples.water_columns; ++w) {
                    double t = 0;
                    for (size_t e = 0; e < weights.size(); ++e)
                        t += weights[e] * std::exp(-mw[e] * w * samples.water_step_mm - mb[e] * b * samples.bone_step_mm);
                    samples.log_values.push_back(-std::log(t));
                }
            model = preprocess::calibrateTwoMaterial(samples);
            double max_error = 0, sum_error = 0;
            size_t holdouts = 0;
            for (double b = .125; b < bone_limit; b += 3.75)
                for (double w = .37; w < 350; w += 3.71) {
                    double t = 0;
                    for (size_t e = 0; e < weights.size(); ++e)
                        t += weights[e] * std::exp(-mw[e] * w - mb[e] * b);
                    const double d = model.correct(float(-std::log(t)), float(b)) - (w * samples.target_water_mu_mm + b * samples.target_bone_mu_mm);
                    max_error = std::max(max_error, std::abs(d));
                    sum_error += d * d;
                    ++holdouts;
                }
            calibrationReport = { { "holdout_max_projection_error", max_error }, { "holdout_rmse", std::sqrt(sum_error / holdouts) } };
        }
        const std::filesystem::path directory = outputDirectory;
        std::filesystem::create_directories(directory);
        if (std::filesystem::exists(directory / "bone-corrected.raw"))
            throw std::runtime_error("Output exists; choose a fresh directory");
        model.save(directory / "calibration.json");
        model = preprocess::TwoMaterialCalibration::load(directory / "calibration.json");
        auto& g = config.geometry.parameters;
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
        config.reconstruction.preprocessing.beam_hardening.enabled = false;
        const size_t voxels = size_t(g.volume_x) * g.volume_y * g.volume_z, pixels = size_t(g.detector_u) * g.detector_v;
        if (std::filesystem::file_size(segmentationPath) != voxels || std::filesystem::file_size(measuredPath) != pixels * g.views * sizeof(float))
            throw std::runtime_error("Input grid size mismatch");
        auto labels = loadLabelVolume(segmentationPath);
        for (auto x : labels)
            if (x > 2)
                throw std::runtime_error("Unknown segmentation label");
        DeviceBuffer<unsigned char> dlabels(voxels);
        dlabels.upload(labels.data(), voxels);
        std::vector<int> map(256, -1);
        map[0] = map[1] = 0;
        map[2] = 1;
        DeviceBuffer<int> dmap(256);
        dmap.upload(map.data(), 256);
        auto provider = std::make_shared<BoneProvider>(detail::makeDeviceGeometry(config, 2, 1), dlabels.data, dmap.data, thicknessScale);
        auto stage = preprocess::makeTwoMaterialBhcStage(model, provider);
        const auto output_path = directory / "bone-corrected.raw";
        preprocess::CudaPixelOps pixel_ops(1);
        TaskProgressCallback progress;
        preprocess::Context context { config, pixel_ops, progress, output_path };
        stage->prepare(context);
        const int capacity = std::min(g.views, std::max(1, int((256ULL << 20) / (pixels * sizeof(float)))));
        DeviceBuffer<float> buffer(pixels * capacity);
        std::vector<float> host(pixels * capacity);
        std::ifstream input(measuredPath, std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        std::ofstream output(output_path, std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        const auto start = std::chrono::steady_clock::now();
        double kernel_ms = 0;
        cudaEvent_t before, after;
        checkCuda(cudaEventCreate(&before));
        checkCuda(cudaEventCreate(&after));
        for (int view = 0; view < g.views; view += capacity) {
            const int count = std::min(capacity, g.views - view);
            const size_t n = pixels * count;
            input.read(reinterpret_cast<char*>(host.data()), n * sizeof(float));
            buffer.upload(host.data(), n);
            checkCuda(cudaEventRecord(before));
            for (int j = 0; j < count; ++j) {
                preprocess::Frame f { view + j, g.detector_u, g.detector_v, pixels, preprocess::Domain::LogAttenuation, preprocess::Location::Device, nullptr, buffer.data + j * pixels, nullptr };
                pipeline::execute(stage->info().name.c_str(), stage->contract(), &f, 1, [](auto) { throw std::runtime_error("unexpected transfer"); }, [&](auto& frame) { stage->process(frame, context); }, [] {});
            }
            checkCuda(cudaEventRecord(after));
            checkCuda(cudaEventSynchronize(after));
            float ms = 0;
            checkCuda(cudaEventElapsedTime(&ms, before, after));
            kernel_ms += ms;
            checkCuda(cudaMemcpy(host.data(), buffer.data, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(host[i])) {
                    DeviceBuffer<float> diagnostic(pixels);
                    provider->writeDevice(view + int(i / pixels), g.detector_u, g.detector_v, diagnostic.data);
                    std::vector<float> paths(pixels);
                    checkCuda(cudaMemcpy(paths.data(), diagnostic.data, pixels * 4, cudaMemcpyDeviceToHost));
                    throw std::runtime_error("Invalid corrected pixel: view=" + std::to_string(view + i / pixels) + " bone_mm=" + std::to_string(paths[i % pixels]) + " max_frame_bone_mm=" + std::to_string(*std::max_element(paths.begin(), paths.end())));
                }
            output.write(reinterpret_cast<char*>(host.data()), n * sizeof(float));
            std::cout << "2dcalbh views=" << view + count << '/' << g.views << std::endl;
        }
        output.close();
        stage->reset();
        cudaEventDestroy(before);
        cudaEventDestroy(after);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Json info = { { "method", "two-material table correction" }, { "bone_only", true }, { "random_rays", false }, { "reference_energy_keV", model.reference_energy_keV },
            { "calibration_provenance", model.provenance }, { "calibration", calibrationReport },
            { "kernel_ms", kernel_ms }, { "processing_seconds", seconds }, { "input", std::filesystem::absolute(measuredPath).generic_string() }, { "segmentation", std::filesystem::absolute(segmentationPath).generic_string() }, { "batch_views", capacity } };
        std::ofstream(directory / "2dcalbh.json") << info.dump(2);
        std::cout << info.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
