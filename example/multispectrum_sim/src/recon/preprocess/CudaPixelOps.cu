#include "cuda/DeviceMemoryPool.hpp"
#include "CudaPixelOps.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <limits>
namespace yk::spectral::preprocess {
namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(
            code == cudaErrorMemoryAllocation
                ? "preprocess GPU memory unavailable within budget (space reserved for VTK)"
                : cudaGetErrorString(code));
}
__global__ void restore(float *data, const float *air, float *original, size_t n, int *invalid) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float value = air[i] * expf(-data[i]);
    if (!isfinite(data[i]) || !isfinite(air[i]) || air[i] <= 0 || !isfinite(value))
        atomicExch(invalid, 1);
    data[i] = original[i] = value;
}
__global__ void makeMask(const float *data, const float *air, unsigned char *mask, size_t n,
                         double threshold) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mask[i] = -log(fmax(double(data[i]) / air[i], 1e-30)) > threshold;
}
__global__ void residual(float *data, const float *original, const float *air,
                         const float *prediction, size_t n, double scale, int *invalid) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (!isfinite(prediction[i]))
        atomicExch(invalid, 1);
    data[i] = fmaxf(air[i] * 1e-8f, original[i] - float(scale) * fmaxf(0, prediction[i]));
}
__global__ void finish(float *data, const float *air, size_t n, bool energy, bool bhc,
                       const double *coefficients, int count, double low, double high,
                       int *invalid) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float value = data[i];
    if (energy)
        value = float(-log(fmax(double(value) / air[i], double(FLT_MIN))));
    if (!isfinite(value))
        atomicExch(invalid, 1);
    double output = value;
    if (bhc) {
        output = 0;
        for (int j = count - 1; j >= 0; --j)
            output = __dadd_rn(__dmul_rn(output, double(value)), coefficients[j]);
        if (!isfinite(output))
            atomicExch(invalid, 1);
        output = fmin(fmax(output, low), high);
    }
    if (!isfinite(output) || fabs(output) > FLT_MAX)
        atomicExch(invalid, 1);
    data[i] = float(output);
}
} // namespace
struct CudaPixelOps::Impl {
    size_t n;
    float *buffers = nullptr;
    double *coefficients = nullptr;
    unsigned char *mask = nullptr;
    int *invalid = nullptr;
    int coefficient_count = 0;
    double low = 0, high = 20;
    ~Impl() {
        if (buffers)
            yk::spectral::cuda_resources::poolRelease(buffers);
        if (coefficients)
            yk::spectral::cuda_resources::poolRelease(coefficients);
        if (mask)
            yk::spectral::cuda_resources::poolRelease(mask);
        if (invalid)
            yk::spectral::cuda_resources::poolRelease(invalid);
    }
};
CudaPixelOps::CudaPixelOps(size_t n) : impl_(std::make_unique<Impl>()) {
    if (!n || n > size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("empty CUDA preprocess frame");
    impl_->n = n;
    yk::spectral::cuda_resources::initializeDevicePool({6 * n * sizeof(float), 17 * sizeof(double), n, sizeof(int)});
    check(yk::spectral::cuda_resources::poolAllocate(&impl_->buffers, 6 * n * sizeof(float)));
    check(yk::spectral::cuda_resources::poolAllocate(&impl_->coefficients, 17 * sizeof(double)));
    check(yk::spectral::cuda_resources::poolAllocate(&impl_->mask, n));
    check(yk::spectral::cuda_resources::poolAllocate(&impl_->invalid, sizeof(int)));
}
CudaPixelOps::~CudaPixelOps() = default;
float *CudaPixelOps::data() { return impl_->buffers; }
float *CudaPixelOps::air() { return impl_->buffers + impl_->n; }
float *CudaPixelOps::original() { return impl_->buffers + 2 * impl_->n; }
float *CudaPixelOps::guide() { return impl_->buffers + 3 * impl_->n; }
float *CudaPixelOps::guideAir() { return impl_->buffers + 4 * impl_->n; }
float *CudaPixelOps::prediction() { return impl_->buffers + 5 * impl_->n; }
void CudaPixelOps::uploadTo(float *p, const std::vector<float> &v) {
    if (v.size() != impl_->n)
        throw std::runtime_error("preprocess frame size mismatch");
    check(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
}
void CudaPixelOps::downloadFrom(const float *p, std::vector<float> &v) {
    v.resize(impl_->n);
    check(cudaMemcpy(v.data(), p, v.size() * sizeof(float), cudaMemcpyDeviceToHost));
}
void CudaPixelOps::beginFrame() { check(cudaMemset(impl_->invalid, 0, sizeof(int))); }
void CudaPixelOps::upload(const std::vector<float> &log, const std::vector<float> *a) {
    check(cudaMemset(impl_->invalid, 0, sizeof(int)));
    uploadTo(data(), log);
    if (a)
        uploadTo(air(), *a);
}
void CudaPixelOps::download(std::vector<float> &v) {
    int invalid = 0;
    check(cudaMemcpy(&invalid, impl_->invalid, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid)
        throw std::runtime_error("CUDA preprocessing produced an invalid value");
    downloadFrom(data(), v);
}
void CudaPixelOps::restoreEnergy() {
    restore<<<int((impl_->n + 255) / 256), 256>>>(data(), air(), original(), impl_->n,
                                                  impl_->invalid);
    check(cudaGetLastError());
}
const unsigned char* CudaPixelOps::makeMaskDevice(float* p,const float* a,double threshold){
    makeMask<<<int((impl_->n+255)/256),256>>>(p,a,impl_->mask,impl_->n,threshold);check(cudaGetLastError());return impl_->mask;
}
std::vector<unsigned char> CudaPixelOps::mask(float *p, const float *a, double threshold) {
    makeMask<<<int((impl_->n + 255) / 256), 256>>>(p, a, impl_->mask, impl_->n, threshold);
    check(cudaGetLastError());
    std::vector<unsigned char> result(impl_->n);
    check(cudaMemcpy(result.data(), impl_->mask, impl_->n, cudaMemcpyDeviceToHost));
    return result;
}
void CudaPixelOps::guidedResidual(double scale) {
    residual<<<int((impl_->n + 255) / 256), 256>>>(data(), original(), air(), prediction(),
                                                   impl_->n, scale, impl_->invalid);
    check(cudaGetLastError());
}
void CudaPixelOps::configureBhc(const BeamHardeningConfig &c) {
    if (!std::isfinite(c.clamp_min) || !std::isfinite(c.clamp_max) || c.clamp_min > c.clamp_max)
        throw std::runtime_error("invalid BHC clamp");
    if (c.coefficients.empty() || c.coefficients.size() > 17)
        throw std::runtime_error("invalid CUDA BHC coefficients");
    for (double v : c.coefficients)
        if (!std::isfinite(v))
            throw std::runtime_error("nonfinite BHC coefficient");
    impl_->coefficient_count = int(c.coefficients.size());
    impl_->low = c.clamp_min;
    impl_->high = c.clamp_max;
    check(cudaMemcpy(impl_->coefficients, c.coefficients.data(),
                     c.coefficients.size() * sizeof(double), cudaMemcpyHostToDevice));
}
void CudaPixelOps::finalize(bool energy, bool bhc) {
    finish<<<int((impl_->n + 255) / 256), 256>>>(data(), air(), impl_->n, energy, bhc,
                                                 impl_->coefficients, impl_->coefficient_count,
                                                 impl_->low, impl_->high, impl_->invalid);
    check(cudaGetLastError());
}
void CudaPixelOps::synchronize() { check(cudaDeviceSynchronize()); }
} // namespace yk::spectral::preprocess
