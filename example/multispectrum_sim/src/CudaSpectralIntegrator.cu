#include "CudaSpectralIntegrator.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <exception>

namespace yk::spectral {
namespace {

__global__ void integrateSpectrumKernel(const float* material_paths_cm,
    const float* density_attenuation, const float* spectrum_weights,
    int material_count, int energy_count, std::size_t pixels,
    float incident_intensity, float* projection)
{
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;

    float transmitted = 0.f;
    for (int energy = 0; energy < energy_count; ++energy) {
        float exponent = 0.f;
        for (int material = 0; material < material_count; ++material) {
            exponent += material_paths_cm[static_cast<std::size_t>(material) *
                pixels + pixel] * density_attenuation[
                static_cast<std::size_t>(material) * energy_count + energy];
        }
        transmitted += spectrum_weights[energy] * expf(-exponent);
    }
    // Constant efficiency affects I and I0 equally and cancels in -log(I/I0).
    projection[pixel] = -logf(fmaxf(transmitted, 1.0e-30f) /
        incident_intensity);
}

bool cudaOk(cudaError_t status, const char* operation, std::string& diagnostic)
{
    if (status == cudaSuccess) return true;
    diagnostic = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
}

}

struct CudaSpectralIntegrator::Impl {
    std::size_t pixels = 0;
    int material_count = 0;
    int energy_count = 0;
    float incident_intensity = 0.f;
    thrust::device_vector<float> paths;
    thrust::device_vector<float> projection;
    thrust::device_vector<float> density_attenuation;
    thrust::device_vector<float> spectrum_weights;
};

CudaSpectralIntegrator::CudaSpectralIntegrator() : impl_(std::make_unique<Impl>()) {}
CudaSpectralIntegrator::~CudaSpectralIntegrator() = default;

bool CudaSpectralIntegrator::initialize(
    const std::vector<float>& weights,
    const std::vector<float>& table,
    int material_count, std::size_t pixels, std::string& diagnostic)
{
    try {
        if (pixels == 0 || material_count <= 0 || weights.empty() ||
            table.size() != static_cast<std::size_t>(material_count) * weights.size()) {
            diagnostic = "CUDA spectral integrator received invalid dimensions or tables";
            return false;
        }
        impl_->pixels = pixels;
        impl_->material_count = material_count;
        impl_->energy_count = static_cast<int>(weights.size());
        impl_->incident_intensity = 0.f;
        for (float weight : weights) impl_->incident_intensity += weight;
        if (!(impl_->incident_intensity > 0.f)) {
            diagnostic = "CUDA spectral integration requires positive total photon weight";
            return false;
        }
        impl_->paths.resize(static_cast<std::size_t>(impl_->material_count) * pixels);
        impl_->projection.resize(pixels);
        impl_->density_attenuation.resize(table.size());
        impl_->spectrum_weights.resize(weights.size());
        thrust::copy(table.begin(), table.end(),
            impl_->density_attenuation.begin());
        thrust::copy(weights.begin(), weights.end(),
            impl_->spectrum_weights.begin());
        diagnostic.clear();
        return true;
    } catch (const std::exception& e) {
        diagnostic = std::string("CUDA spectral initialization failed: ") + e.what();
        return false;
    }
}

bool CudaSpectralIntegrator::integrate(
    const std::vector<float>& material_paths_cm,
    std::vector<float>& projection, std::string& diagnostic)
{
    try {
        const std::size_t expected = static_cast<std::size_t>(
            impl_->material_count) * impl_->pixels;
        if (impl_->pixels == 0 || material_paths_cm.size() != expected) {
            diagnostic = "Per-view path count does not match CUDA integrator dimensions";
            return false;
        }
        thrust::copy(material_paths_cm.begin(), material_paths_cm.end(),
            impl_->paths.begin());
        constexpr int kThreads = 256;
        const int blocks = static_cast<int>((impl_->pixels + kThreads - 1) /
            kThreads);
        integrateSpectrumKernel<<<blocks, kThreads>>>(
            thrust::raw_pointer_cast(impl_->paths.data()),
            thrust::raw_pointer_cast(impl_->density_attenuation.data()),
            thrust::raw_pointer_cast(impl_->spectrum_weights.data()),
            impl_->material_count, impl_->energy_count, impl_->pixels,
            impl_->incident_intensity,
            thrust::raw_pointer_cast(impl_->projection.data()));
        if (!cudaOk(cudaGetLastError(), "CUDA spectral kernel launch failed", diagnostic) ||
            !cudaOk(cudaDeviceSynchronize(), "CUDA spectral kernel execution failed", diagnostic))
            return false;
        projection.resize(impl_->pixels);
        thrust::copy(impl_->projection.begin(), impl_->projection.end(),
            projection.begin());
        diagnostic.clear();
        return true;
    } catch (const std::exception& e) {
        diagnostic = std::string("CUDA spectral integration failed: ") + e.what();
        return false;
    }
}

}
