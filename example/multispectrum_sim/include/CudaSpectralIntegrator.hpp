#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace yk::spectral {

// Per-view CUDA spectral integrator. Spectrum tables remain resident on the
// device; integrate() transfers only one view and never caches a full scan.
class CudaSpectralIntegrator {
public:
    CudaSpectralIntegrator();
    ~CudaSpectralIntegrator();
    CudaSpectralIntegrator(const CudaSpectralIntegrator&) = delete;
    CudaSpectralIntegrator& operator=(const CudaSpectralIntegrator&) = delete;

    // density_attenuation is [material][energy] and includes density.
    bool initialize(const std::vector<float>& spectrum_weights,
        const std::vector<float>& density_attenuation, int material_count,
        std::size_t pixels, std::string& diagnostic);
    bool integrate(const std::vector<float>& material_paths_cm,
        std::vector<float>& projection, std::string& diagnostic);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
