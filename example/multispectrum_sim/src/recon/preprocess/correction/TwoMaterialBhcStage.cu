#include "cuda/DeviceMemoryPool.hpp"
#include "recon/preprocess/correction/TwoMaterialBhcStage.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
namespace yk::spectral::preprocess {
namespace {
    void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    __device__ float row(int b, float p, const float* minimum, const float* step, const float* table, int columns)
    {
        const float x = (p - minimum[b]) / step[b];
        const int i = int(fminf(fmaxf(x, 0), float(columns - 2)));
        const float f = x - i;
        const size_t base = size_t(b) * columns + i;
        return table[base] + f * (table[base + 1] - table[base]);
    }
    __global__ void apply(float* projection, const float* bone, size_t n, const float* minimum, const float* step, const float* table, int rows, int columns, float bone_step)
    {
        const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= n)
            return;
        const float t = bone[i], p = projection[i];
        if (!isfinite(t) || !isfinite(p) || t < 0 || t > (rows - 1) * bone_step) {
            projection[i] = nanf("");
            return;
        }
        const float x = t / bone_step;
        const int b = min(int(x), rows - 2);
        const float f = x - b;
        projection[i] = row(b, p, minimum, step, table, columns) * (1 - f) + row(b + 1, p, minimum, step, table, columns) * f;
    }
    class Stage final : public IStage {
        TwoMaterialCalibration model_;
        std::shared_ptr<IBoneThicknessProvider> provider_;
        float *bone_ = nullptr, *minimum_ = nullptr, *step_ = nullptr, *table_ = nullptr;
        size_t pixels_ = 0;
        int width_ = 0, height_ = 0, views_ = 0;
        void alloc(float*& p, size_t n) { check(cuda_resources::poolAllocate(&p, n * sizeof(float))); }

    public:
        Stage(TwoMaterialCalibration model, std::shared_ptr<IBoneThicknessProvider> provider)
            : model_(std::move(model))
            , provider_(std::move(provider))
        {
            model_.validate();
            if (!provider_)
                throw std::invalid_argument("missing bone thickness provider");
        }
        ~Stage() { reset(); }
        const pipeline::PluginInfo& info() const noexcept override { return twoMaterialBhcInfo(); }
        pipeline::Contract<Domain> contract() const override { return { Domain::LogAttenuation, Domain::LogAttenuation, Location::Device, Location::Device }; }
        void validateContext(const Context& c) const override
        {
            if (c.config.reconstruction.preprocessing.beam_hardening.enabled)
                throw std::invalid_argument("Two-material BHC replaces water BHC; disable the built-in water correction");
            const auto& g = c.config.geometry.parameters;
            if (g.detector_u <= 0 || g.detector_v <= 0 || g.views <= 0)
                throw std::invalid_argument("invalid detector dimensions");
        }
        void prepare(const Context& c) override
        {
            reset();
            validateContext(c);
            const auto& g = c.config.geometry.parameters;
            width_ = g.detector_u;
            height_ = g.detector_v;
            views_ = g.views;
            pixels_ = size_t(width_) * height_;
            try {
                alloc(bone_, pixels_);
                alloc(minimum_, model_.bone_rows);
                alloc(step_, model_.bone_rows);
                alloc(table_, model_.target_log.size());
                check(cudaMemcpy(minimum_, model_.minimum_log.data(), model_.bone_rows * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(step_, model_.log_step.data(), model_.bone_rows * sizeof(float), cudaMemcpyHostToDevice));
                check(cudaMemcpy(table_, model_.target_log.data(), model_.target_log.size() * sizeof(float), cudaMemcpyHostToDevice));
                provider_->prepare(c);
            } catch (...) {
                reset();
                throw;
            }
        }
        void process(Frame& f, const Context&) override
        {
            if (!bone_ || f.width != width_ || f.height != height_ || f.pixels != pixels_ || f.view < 0 || f.view >= views_ || !f.device_data || f.location != Location::Device || f.domain != Domain::LogAttenuation)
                throw std::invalid_argument("invalid two-material BHC frame");
            provider_->writeDevice(f.view, width_, height_, bone_);
            apply<<<unsigned((pixels_ + 255) / 256), 256>>>(f.device_data, bone_, pixels_, minimum_, step_, table_, model_.bone_rows, model_.projection_columns, float(model_.bone_step_mm));
            check(cudaGetLastError());
        }
        void reset() noexcept override
        {
            provider_->reset();
            cuda_resources::poolRelease(bone_);
            cuda_resources::poolRelease(minimum_);
            cuda_resources::poolRelease(step_);
            cuda_resources::poolRelease(table_);
            bone_ = minimum_ = step_ = table_ = nullptr;
            pixels_ = 0;
        }
    };
}
std::unique_ptr<IStage> makeTwoMaterialBhcStage(TwoMaterialCalibration m, std::shared_ptr<IBoneThicknessProvider> p) { return std::make_unique<Stage>(std::move(m), std::move(p)); }
}
