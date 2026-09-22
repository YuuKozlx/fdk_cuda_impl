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
    const float* spectrum_energies, const float* geometry_flux,
    int material_count, int energy_count, std::size_t pixels,
    float incident_energy, float* projection, float* energy_signal)
{
    const std::size_t pixel = static_cast<std::size_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;

    float transmitted_energy = 0.f;
    for (int energy = 0; energy < energy_count; ++energy) {
        float exponent = 0.f;
        for (int material = 0; material < material_count; ++material) {
            exponent += material_paths_cm[static_cast<std::size_t>(material) *
                pixels + pixel] * density_attenuation[
                static_cast<std::size_t>(material) * energy_count + energy];
        }
        transmitted_energy += spectrum_weights[energy] * spectrum_energies[energy] * expf(-exponent);
    }
    // Constant efficiency affects I and I0 equally and cancels in -log(I/I0).
    const float flux = geometry_flux ? geometry_flux[pixel] : 1.f;
    energy_signal[pixel] = flux * transmitted_energy;
    projection[pixel] = -logf(fmaxf(transmitted_energy, 1.0e-30f) /
        incident_energy);
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
    float incident_energy = 0.f;
    thrust::device_vector<float> paths;
    thrust::device_vector<float> projection;
    thrust::device_vector<float> density_attenuation;
    thrust::device_vector<float> spectrum_weights;
    thrust::device_vector<float> spectrum_energies;
    thrust::device_vector<float> geometry_flux;
    thrust::device_vector<float> energy_signal;
};

CudaSpectralIntegrator::CudaSpectralIntegrator() : impl_(std::make_unique<Impl>()) {}
CudaSpectralIntegrator::~CudaSpectralIntegrator() = default;

bool CudaSpectralIntegrator::initialize(
    const std::vector<float>& weights, const std::vector<float>& energies,
    const std::vector<float>& table,
    int material_count, std::size_t pixels, std::string& diagnostic)
{
    try {
        if (pixels == 0 || material_count <= 0 || weights.empty() || energies.size() != weights.size() ||
            table.size() != static_cast<std::size_t>(material_count) * weights.size()) {
            diagnostic = "CUDA spectral integrator received invalid dimensions or tables";
            return false;
        }
        impl_->pixels = pixels;
        impl_->material_count = material_count;
        impl_->energy_count = static_cast<int>(weights.size());
        impl_->incident_energy = 0.f;
        for (std::size_t i = 0; i < weights.size(); ++i) impl_->incident_energy += weights[i] * energies[i];
        if (!(impl_->incident_energy > 0.f)) {
            diagnostic = "CUDA spectral integration requires positive incident energy";
            return false;
        }
        impl_->paths.resize(static_cast<std::size_t>(impl_->material_count) * pixels);
        impl_->projection.resize(pixels);
        impl_->energy_signal.resize(pixels);
        impl_->density_attenuation.resize(table.size());
        impl_->spectrum_weights.resize(weights.size());
        thrust::copy(table.begin(), table.end(),
            impl_->density_attenuation.begin());
        thrust::copy(weights.begin(), weights.end(),
            impl_->spectrum_weights.begin());
        impl_->spectrum_energies.resize(energies.size());
        thrust::copy(energies.begin(), energies.end(), impl_->spectrum_energies.begin());
        diagnostic.clear();
        return true;
    } catch (const std::exception& e) {
        diagnostic = std::string("CUDA spectral initialization failed: ") + e.what();
        return false;
    }
}

bool CudaSpectralIntegrator::integrate(
    const std::vector<float>& material_paths_cm,
    const std::vector<float>& flux,
    std::vector<float>& projection, std::vector<float>& energy_signal,
    std::string& diagnostic)
{
    try {
        const std::size_t expected = static_cast<std::size_t>(
            impl_->material_count) * impl_->pixels;
        if (impl_->pixels == 0 || material_paths_cm.size() != expected ||
            (!flux.empty() && flux.size() != impl_->pixels)) {
            diagnostic = "Per-view path count does not match CUDA integrator dimensions";
            return false;
        }
        thrust::copy(material_paths_cm.begin(), material_paths_cm.end(),
            impl_->paths.begin());
        if (!flux.empty()) {
            impl_->geometry_flux.resize(flux.size());
            thrust::copy(flux.begin(), flux.end(), impl_->geometry_flux.begin());
        }
        constexpr int kThreads = 256;
        const int blocks = static_cast<int>((impl_->pixels + kThreads - 1) /
            kThreads);
        integrateSpectrumKernel<<<blocks, kThreads>>>(
            thrust::raw_pointer_cast(impl_->paths.data()),
            thrust::raw_pointer_cast(impl_->density_attenuation.data()),
            thrust::raw_pointer_cast(impl_->spectrum_weights.data()),
            thrust::raw_pointer_cast(impl_->spectrum_energies.data()),
            flux.empty() ? nullptr : thrust::raw_pointer_cast(impl_->geometry_flux.data()),
            impl_->material_count, impl_->energy_count, impl_->pixels,
            impl_->incident_energy,
            thrust::raw_pointer_cast(impl_->projection.data()),
            thrust::raw_pointer_cast(impl_->energy_signal.data()));
        if (!cudaOk(cudaGetLastError(), "CUDA spectral kernel launch failed", diagnostic) ||
            !cudaOk(cudaDeviceSynchronize(), "CUDA spectral kernel execution failed", diagnostic))
            return false;
        projection.resize(impl_->pixels);
        energy_signal.resize(impl_->pixels);
        thrust::copy(impl_->projection.begin(), impl_->projection.end(),
            projection.begin());
        thrust::copy(impl_->energy_signal.begin(), impl_->energy_signal.end(),
            energy_signal.begin());
        diagnostic.clear();
        return true;
    } catch (const std::exception& e) {
        diagnostic = std::string("CUDA spectral integration failed: ") + e.what();
        return false;
    }
}

}
