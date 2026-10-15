#include "BoneBhcIo.hpp"
#include "WaterReference.hpp"
#include "XcomAttenuationProvider.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include "primary/IdealBoneCorrection.cuh"
#include "recon/bhc/BoneBhcStage.hpp"
#include "recon/preprocess/CudaPixelOps.hpp"
#include "recon/preprocess/correction/TwoMaterialBhcStage.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>

namespace yk::spectral::bone_bhc {
namespace {
    using Json = nlohmann::json;
    namespace pd = primary_detail;
    void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    template <class T>
    struct Buffer {
        T* p = nullptr;
        explicit Buffer(size_t n) { check(cuda_resources::poolAllocate(&p, n * sizeof(T))); }
        ~Buffer() { cuda_resources::poolRelease(p); }
        Buffer(const Buffer&) = delete;
        void upload(const T* x, size_t n) { check(cudaMemcpy(p, x, n * sizeof(T), cudaMemcpyHostToDevice)); }
        void download(T* x, size_t n) { check(cudaMemcpy(x, p, n * sizeof(T), cudaMemcpyDeviceToHost)); }
    };
    using detail::openInput;
    using detail::openOutput;
    using detail::product;
    std::vector<unsigned char> segmentVolume(const std::filesystem::path& volume, size_t voxels, double water_mu,
        SegmentationOptions segmentation, std::array<size_t, 3>& counts)
    {
        // Segment in bounded CPU chunks; only byte labels occupy a whole host volume.
        auto vf = openInput(volume, voxels * sizeof(float));
        std::vector<unsigned char> labels(voxels);
        std::vector<float> chunk(std::min(voxels, size_t(1 << 20)));

        for (size_t first = 0; first < voxels; first += chunk.size()) {
            const size_t n = std::min(chunk.size(), voxels - first);
            vf.read(reinterpret_cast<char*>(chunk.data()), n * sizeof(float));
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(chunk[i]))
                    throw std::runtime_error("nonfinite iBHC volume");
                const double hu = 1000 * (chunk[i] / water_mu - 1);
                const int label = hu < segmentation.air_hu ? 0 : hu < segmentation.bone_hu ? 1
                                                                                           : 2;
                labels[first + i] = static_cast<unsigned char>(label);
                ++counts[label];
            }
        }
        return labels;
    }
}
struct PhysicalBoneBhc::Impl {
    SimulationConfig config_;
    WaterReference reference_;
    double air_hu_, bone_hu_;
    std::vector<double> weights_, mu_, mono_;

public:
    size_t pixels, elements, voxels;
    Domain domain_;
    BeamHardeningConfig fit_;
    Impl(SimulationConfig config, Domain domain, BeamHardeningConfig fit, SegmentationOptions segmentation)
        : config_(std::move(config))
        , reference_(resolveWaterReference(config_))
        , air_hu_(segmentation.air_hu)
        , bone_hu_(segmentation.bone_hu)
        , domain_(domain)
        , fit_(std::move(fit))
    {
        if (domain_ != Domain::ScatterCorrectedLog && domain_ != Domain::WaterCorrectedLog)
            throw std::invalid_argument("invalid physical BHC input domain");
        if (domain_ == Domain::WaterCorrectedLog && (!fit_.enabled || fit_.coefficients.empty()))
            throw std::invalid_argument("water-corrected input requires its water polynomial");
        if (!std::isfinite(air_hu_) || !std::isfinite(bone_hu_) || air_hu_ >= bone_hu_)
            throw std::invalid_argument("invalid iBHC segmentation thresholds");
        if (config_.projection.detector_response.enabled)
            throw std::invalid_argument("iBHC currently requires an ideal energy-integrating detector response");
        if (config_.geometry.kind != GeometryKind::FlatHelical && config_.geometry.kind != GeometryKind::CylHelical && config_.geometry.kind != GeometryKind::FlatCbct)
            throw std::invalid_argument("iBHC supports flat circular and flat/cylindrical helical geometry");
        auto& g = config_.geometry.parameters;
        pixels = product(g.detector_u, g.detector_v, 1);
        elements = product(g.detector_u, g.detector_v, g.views);
        voxels = product(g.reconstruction_volume_x, g.reconstruction_volume_y, g.reconstruction_volume_z);
        if (voxels > size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("iBHC label grid exceeds kernel index range");
        if (pixels > size_t(std::numeric_limits<int>::max()) || pixels * 4 * sizeof(float) > (256ULL << 20))
            throw std::invalid_argument("iBHC detector exceeds batch limit");
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
        config_.projection.focal_spot.enabled = false;
        config_.projection.focal_spot.defocus_enabled = false;
        const auto spectrum = loadProjectionSpectrum(config_.projection);
        if (spectrum.empty() || spectrum.size() > 256)
            throw std::invalid_argument("invalid iBHC spectrum");
        std::vector<double> energies;
        double total = 0;
        for (const auto& s : spectrum) {
            energies.push_back(s.energy_keV);
            const double w = s.fluence_per_keV_cm2_mAs_at_1m * s.bin_width_keV * s.energy_keV;
            if (!std::isfinite(w) || w < 0)
                throw std::invalid_argument("invalid iBHC energy weight");
            weights_.push_back(w);
            total += w;
        }
        if (!(total > 0) || !std::isfinite(total))
            throw std::invalid_argument("empty iBHC spectrum");
        for (auto& w : weights_)
            w /= total;
        // Explicit material assumptions; background is vacuum. No truth-label input.
        mu_.resize(3 * energies.size(), 0);
        mono_.resize(3, 0);
        XcomAttenuationProvider attenuation({});
        for (int m = 1; m < 3; ++m) {
            const auto it = std::find_if(config_.projection.materials.begin(), config_.projection.materials.end(),
                [m](const auto& material) { return material.label == m; });
            if (it == config_.projection.materials.end())
                throw std::invalid_argument("physical BHC requires materials labelled 1=soft, 2=bone");
            const MaterialSpec& material = *it;
            if (!std::isfinite(material.density_g_cm3) || material.density_g_cm3 <= 0)
                throw std::invalid_argument("invalid iBHC density");
            std::vector<double> mass, ref;
            std::string error;
            if (!attenuation.query(material, energies, mass, error) || !attenuation.query(material, { reference_.energy_keV }, ref, error))
                throw std::runtime_error(error);
            for (size_t e = 0; e < energies.size(); ++e)
                mu_[m * energies.size() + e] = mass[e] * material.density_g_cm3;
            mono_[m] = ref[0] * material.density_g_cm3;
        }
    }
    double run(const std::filesystem::path& volume, const std::filesystem::path& original, const std::filesystem::path& output)
    {
        std::array<size_t, 3> counts {};
        auto labels = segmentVolume(volume, voxels, reference_.mu_water_mm_inv, { air_hu_, bone_hu_ }, counts);
        const auto& g = config_.geometry.parameters;
        const int capacity = std::min(g.views, int((256ULL << 20) / (4 * pixels * sizeof(float))));
        const size_t batch = pixels * capacity;
        Buffer<unsigned char> dl(voxels);
        dl.upload(labels.data(), voxels);
        std::vector<int> map(256, -1);
        map[0] = 0;
        map[1] = 1;
        map[2] = 2;
        Buffer<int> dm(256);
        dm.upload(map.data(), 256);
        Buffer<double> mu(mu_.size()), mono(3), weights(weights_.size());
        mu.upload(mu_.data(), mu_.size());
        mono.upload(mono_.data(), 3);
        weights.upload(weights_.data(), weights_.size());
        const bool water = domain_ == Domain::WaterCorrectedLog;
        Buffer<double> coefficients(std::max(size_t(1), fit_.coefficients.size()));
        if (water)
            coefficients.upload(fit_.coefficients.data(), fit_.coefficients.size());
        Buffer<float> measured(batch), corrected(batch), delta(batch), poly(batch);
        std::vector<float> input(batch), result(batch), model(batch);
        auto in = openInput(original, elements * sizeof(float));
        auto out = openOutput(output);
        const auto dg = pd::makeDeviceGeometry(config_, 3, int(weights_.size()));
        double sum = 0;
        for (int view = 0; view < g.views; view += capacity) {
            const int nv = std::min(capacity, g.views - view);
            const size_t n = pixels * nv;
            in.read(reinterpret_cast<char*>(input.data()), n * sizeof(float));
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(input[i]) || input[i] < -80)
                    throw std::runtime_error("invalid pre-water-BHC log projection");
            measured.upload(input.data(), n);
            pd::idealBoneCorrectionKernel<<<dim3(unsigned((pixels + 127) / 128), unsigned(nv)), 128>>>(dg, dl.p, dm.p, mu.p, weights.p, mono.p, water ? coefficients.p : nullptr, water ? int(fit_.coefficients.size()) : 0,
                water ? fit_.clamp_min : -1e30, water ? fit_.clamp_max : 1e30, view, measured.p, corrected.p, delta.p, poly.p);
            check(cudaGetLastError());
            corrected.download(result.data(), n);
            poly.download(model.data(), n);
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(result[i]) || !std::isfinite(model[i]))
                    throw std::runtime_error("nonfinite iBHC model output");
                double model_value = model[i];
                if (water) {
                    model_value = 0;
                    for (auto it = fit_.coefficients.rbegin(); it != fit_.coefficients.rend(); ++it)
                        model_value = model_value * model[i] + *it;
                    model_value = std::clamp(model_value, fit_.clamp_min, fit_.clamp_max);
                }
                const double d = water ? double(input[i]) - model_value : std::exp(-double(input[i])) - std::exp(-model_value);
                sum += d * d;
            }
            out.write(reinterpret_cast<const char*>(result.data()), n * sizeof(float));
            std::cout << "ibhc model views=" << view + nv << '/' << g.views << std::endl;
        }
        Json report = { { "segmentation_counts_air_soft_bone", counts }, { "batch_views", capacity },
            { "model_error", sum / elements }, { "formula", water ? "water_log + mono - W(poly)" : "original_log + mono - poly" },
            { "water_mu_mm_inv", reference_.mu_water_mm_inv }, { "reference_energy_keV", reference_.energy_keV } };
        auto log = openOutput(output.parent_path() / "model.json");
        log << report.dump(2);
        return sum / elements;
    }
};
PhysicalBoneBhc::PhysicalBoneBhc(SimulationConfig c, Domain d, BeamHardeningConfig b, SegmentationOptions s)
    : impl_(std::make_unique<Impl>(std::move(c), d, std::move(b), s))
{
}
PhysicalBoneBhc::~PhysicalBoneBhc() = default;
Domain PhysicalBoneBhc::inputDomain() const noexcept { return impl_->domain_; }
const char* PhysicalBoneBhc::errorMetric() const noexcept { return inputDomain() == Domain::WaterCorrectedLog ? "water_log_mse" : "transmission_mse"; }
std::optional<double> PhysicalBoneBhc::correct(const std::filesystem::path& v, const std::filesystem::path& p, const std::filesystem::path& o) { return impl_->run(v, p, o); }

namespace {
    __global__ void bonePaths(pd::DeviceGeometry g, const unsigned char* labels, const int* map, int view, float* out, float scale)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= g.nu * g.nv)
            return;
        const auto ray = pd::scannerRay(g, view, float(i % g.nu), float(i / g.nu));
        float3 d = pd::add3(ray.detector, pd::scale3(ray.source, -1));
        d = pd::scale3(d, rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z));
        const float3 lo = make_float3(-.5f * g.nx * g.vx, -.5f * g.ny * g.vy, -.5f * g.nz * g.vz);
        float paths[2] = {};
        pd::accumulatePaths<2>(g, labels, map, pd::phantomInverse(g, ray.source, false), pd::phantomInverse(g, d, true), lo, pd::scale3(lo, -1), paths);
        out[i] = 10 * paths[1] * scale;
    }
    class BoneProvider final : public preprocess::IBoneThicknessProvider {
        pd::DeviceGeometry g_;
        const unsigned char* labels_;
        const int* map_;
        float scale_;

    public:
        BoneProvider(pd::DeviceGeometry g, const unsigned char* l, const int* m, float s)
            : g_(g)
            , labels_(l)
            , map_(m)
            , scale_(s)
        {
        }
        void writeDevice(int view, int width, int height, float* out) override
        {
            bonePaths<<<(width * height + 127) / 128, 128>>>(g_, labels_, map_, view, out, scale_);
            check(cudaGetLastError());
        }
    };
}
struct TwoMaterialLutBhc::Impl {
    SimulationConfig config;
    preprocess::TwoMaterialCalibration calibration;
    double scale;
    SegmentationOptions segmentation;
    WaterReference reference;
    size_t pixels, elements, voxels;
    Impl(SimulationConfig c, preprocess::TwoMaterialCalibration table, double factor, SegmentationOptions seg)
        : config(std::move(c))
        , calibration(std::move(table))
        , scale(factor)
        , segmentation(seg)
        , reference(resolveWaterReference(config))
    {
        calibration.validate();
        if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(seg.air_hu) || !std::isfinite(seg.bone_hu) || seg.air_hu >= seg.bone_hu)
            throw std::invalid_argument("invalid LUT thickness scale/segmentation");
        // Automatic HU re-segmentation needs an assigned target reference. Existing
        // frame LUT stage remains available for empirical tables without one.
        if (calibration.reference_energy_keV <= 0 || std::abs(calibration.reference_energy_keV - reference.energy_keV) > 1e-6)
            throw std::invalid_argument("BoneBhcStage LUT target energy must match reconstruction HU reference");
        if (config.geometry.kind != GeometryKind::FlatCbct && config.geometry.kind != GeometryKind::FlatHelical && config.geometry.kind != GeometryKind::CylHelical)
            throw std::invalid_argument("unsupported bone BHC geometry");
        auto& g = config.geometry.parameters;
        pixels = product(g.detector_u, g.detector_v, 1);
        elements = product(g.detector_u, g.detector_v, g.views);
        voxels = product(g.reconstruction_volume_x, g.reconstruction_volume_y, g.reconstruction_volume_z);
        if (voxels > size_t(std::numeric_limits<int>::max()) || pixels * 2 * sizeof(float) > (256ULL << 20))
            throw std::invalid_argument("LUT grid exceeds buffer/index capacity");
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
        config.reconstruction.preprocessing.beam_hardening.enabled = false;
    }
    void run(const std::filesystem::path& volume, const std::filesystem::path& original, const std::filesystem::path& output)
    {
        std::array<size_t, 3> counts {};
        auto labels = segmentVolume(volume, voxels, reference.mu_water_mm_inv, segmentation, counts);
        Buffer<unsigned char> dl(voxels);
        dl.upload(labels.data(), voxels);
        std::vector<int> map(256, -1);
        map[2] = 1;
        Buffer<int> dm(256);
        dm.upload(map.data(), 256);
        const auto& g = config.geometry.parameters;
        auto provider = std::make_shared<BoneProvider>(pd::makeDeviceGeometry(config, 2, 1), dl.p, dm.p, float(scale));
        auto stage = preprocess::makeTwoMaterialBhcStage(calibration, provider);
        preprocess::CudaPixelOps ops(1);
        TaskProgressCallback progress;
        preprocess::Context context { config, ops, progress, output };
        auto cleanup = pipeline::onExit([&] { stage->reset(); });
        stage->prepare(context);
        // Reserve one frame for the LUT stage's bone path buffer.
        const int capacity = std::min(g.views, int((256ULL << 20) / (pixels * sizeof(float))) - 1);
        const size_t batch = pixels * capacity;
        Buffer<float> buffer(batch);
        std::vector<float> host(batch);
        auto in = openInput(original, elements * sizeof(float));
        auto out = openOutput(output);
        for (int first = 0; first < g.views; first += capacity) {
            const int count = std::min(capacity, g.views - first);
            const size_t n = pixels * count;
            in.read(reinterpret_cast<char*>(host.data()), n * sizeof(float));
            buffer.upload(host.data(), n);
            for (int v = 0; v < count; ++v) {
                preprocess::Frame frame;
                frame.view = first + v;
                frame.width = g.detector_u;
                frame.height = g.detector_v;
                frame.pixels = pixels;
                frame.location = pipeline::Location::Device;
                frame.device_data = buffer.p + size_t(v) * pixels;
                stage->process(frame, context);
            }
            buffer.download(host.data(), n);
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(host[i]))
                    throw std::runtime_error("LUT output invalid: check input and calibrated bone thickness coverage");
            out.write(reinterpret_cast<const char*>(host.data()), n * sizeof(float));
            std::cout << "bone_bhc lut views=" << first + count << '/' << g.views << std::endl;
        }
        auto log = openOutput(output.parent_path() / "model.json");
        log << Json { { "algorithm", "two_material_lut" }, { "segmentation_counts_air_soft_bone", counts },
            { "bone_thickness_scale", scale }, { "model_error", nullptr }, { "reference_energy_keV", reference.energy_keV } }
                   .dump(2);
    }
};
TwoMaterialLutBhc::TwoMaterialLutBhc(SimulationConfig c, preprocess::TwoMaterialCalibration t, double s, SegmentationOptions seg)
    : impl_(std::make_unique<Impl>(std::move(c), std::move(t), s, seg))
{
}
TwoMaterialLutBhc::~TwoMaterialLutBhc() = default;
std::optional<double> TwoMaterialLutBhc::correct(const std::filesystem::path& v, const std::filesystem::path& p, const std::filesystem::path& o)
{
    impl_->run(v, p, o);
    return std::nullopt;
}

}
