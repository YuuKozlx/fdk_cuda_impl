#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace yk::spectral {
namespace {

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

void checkFft(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT error " +
                                 std::to_string(static_cast<int>(result)));
}

__global__ void multiplySpectra(cufftComplex* a, const cufftComplex* b,
                                size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const cufftComplex x = a[i];
    const cufftComplex y = b[i];
    a[i].x = x.x * y.x - x.y * y.y;
    a[i].y = x.x * y.y + x.y * y.x;
}

__global__ void cropToImage(const float* padded, float* output, int width,
                            int height, int padded_width, float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    output[y * width + x] = padded[y * padded_width + x] * scale;
}

__global__ void addImage(float* destination, const float* source, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) destination[i] += source[i];
}

__global__ void accumulateGammaCorrected(float* destination, const float* normal,
                                         const float* wet_convolved, const float* wet,
                                         size_t count, float gamma) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    destination[i] += (1.0f - gamma * wet[i]) * normal[i]
                    + gamma * wet_convolved[i];
}

std::vector<double> edt1d(const std::vector<double>& input, double spacing) {
    const int n = static_cast<int>(input.size());
    std::vector<int> sites;
    for (int i = 0; i < n; ++i)
        if (std::isfinite(input[i])) sites.push_back(i);
    std::vector<double> output(n, std::numeric_limits<double>::infinity());
    if (sites.empty()) return output;
    std::vector<int> v(sites.size());
    std::vector<double> z(sites.size() + 1);
    int k = 0;
    v[0] = sites[0];
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    const double a = spacing * spacing;
    for (size_t site_index = 1; site_index < sites.size(); ++site_index) {
        const int q = sites[site_index];
        double intersection = 0.0;
        do {
            const int p = v[k];
            intersection = ((input[q] + a * q * q) -
                            (input[p] + a * p * p)) /
                           (2.0 * a * (q - p));
            if (intersection <= z[k]) --k;
        } while (intersection <= z[k]);
        ++k;
        v[k] = q;
        z[k] = intersection;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const double delta = q - v[k];
        output[q] = input[v[k]] + a * delta * delta;
    }
    return output;
}

std::vector<float> edgeWindow(const std::vector<unsigned char>& mask, int width,
                              int height, double pixel_u, double pixel_v,
                              const FasksRuntimeModel& model) {
    std::vector<double> distance(static_cast<size_t>(width) * height,
                                 std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < distance.size(); ++i)
        if (!mask[i]) distance[i] = 0.0;
    for (int y = 0; y < height; ++y) {
        std::vector<double> row(width);
        for (int x = 0; x < width; ++x) row[x] = distance[static_cast<size_t>(y) * width + x];
        row = edt1d(row, pixel_u);
        for (int x = 0; x < width; ++x) distance[static_cast<size_t>(y) * width + x] = row[x];
    }
    for (int x = 0; x < width; ++x) {
        std::vector<double> column(height);
        for (int y = 0; y < height; ++y) column[y] = distance[static_cast<size_t>(y) * width + x];
        column = edt1d(column, pixel_v);
        for (int y = 0; y < height; ++y) distance[static_cast<size_t>(y) * width + x] = column[y];
    }
    std::vector<float> result(distance.size(), 0.0f);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!mask[i]) continue;
        const double normalized = std::min(
            distance[i] / model.edgeRampMm(), 1.0);
        result[i] = static_cast<float>(model.edgeBoundaryScale()
                                       + (1.0 - model.edgeBoundaryScale()) * normalized);
    }
    return result;
}

void smoothNearest(std::vector<float>& image, int width, int height,
                   double pixel_u, double pixel_v, double sigma) {
    if (sigma <= 0.0) return;
    const auto weights = [](double pixel, double sigma) {
        const int radius = std::max(1, static_cast<int>(std::ceil(4.0 * sigma / pixel)));
        std::vector<float> result(static_cast<size_t>(2 * radius + 1));
        double sum = 0.0;
        for (int k = -radius; k <= radius; ++k) {
            result[static_cast<size_t>(k + radius)] = static_cast<float>(
                std::exp(-0.5 * std::pow(k * pixel / sigma, 2.0)));
            sum += result[static_cast<size_t>(k + radius)];
        }
        for (float& value : result) value = static_cast<float>(value / sum);
        return result;
    };
    const auto wu = weights(pixel_u, sigma);
    const auto wv = weights(pixel_v, sigma);
    std::vector<float> horizontal(image.size());
    const int ru = static_cast<int>(wu.size() / 2);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int k = -ru; k <= ru; ++k)
                sum += image[static_cast<size_t>(y) * width +
                             std::clamp(x + k, 0, width - 1)] * wu[k + ru];
            horizontal[static_cast<size_t>(y) * width + x] = static_cast<float>(sum);
        }
    image.assign(image.size(), 0.0f);
    const int rv = static_cast<int>(wv.size() / 2);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            for (int k = -rv; k <= rv; ++k)
                sum += horizontal[static_cast<size_t>(std::clamp(y + k, 0, height - 1)) * width + x]
                     * wv[k + rv];
            image[static_cast<size_t>(y) * width + x] = static_cast<float>(sum);
        }
}

class FftWorkspace {
public:
    FftWorkspace(int width, int height) : width_(width), height_(height) {
        padded_width_ = 2 * width_ - 1;
        padded_height_ = 2 * height_ - 1;
        image_count_ = static_cast<size_t>(padded_width_) * padded_height_;
        spectrum_count_ = static_cast<size_t>(padded_height_) *
                          (padded_width_ / 2 + 1);
        check(cudaMalloc(&source_, image_count_ * sizeof(float)), "cudaMalloc fASKS source");
        check(cudaMalloc(&kernel_, image_count_ * sizeof(float)), "cudaMalloc fASKS kernel");
        check(cudaMalloc(&source_fft_, spectrum_count_ * sizeof(cufftComplex)), "cudaMalloc fASKS source FFT");
        check(cudaMalloc(&scatter_, imagePixels() * sizeof(float)), "cudaMalloc fASKS scatter");
        check(cudaMalloc(&normal_, imagePixels() * sizeof(float)), "cudaMalloc fASKS normal");
        check(cudaMalloc(&wet_convolved_, imagePixels() * sizeof(float)), "cudaMalloc fASKS wet convolution");
        check(cudaMalloc(&wet_, imagePixels() * sizeof(float)), "cudaMalloc fASKS WET");
        checkFft(cufftPlan2d(&forward_plan_, padded_height_, padded_width_, CUFFT_R2C), "cufftPlan2d fASKS forward");
        checkFft(cufftPlan2d(&inverse_plan_, padded_height_, padded_width_, CUFFT_C2R), "cufftPlan2d fASKS inverse");
    }

    ~FftWorkspace() {
        for (auto spectrum : kernels_) if (spectrum) cudaFree(spectrum);
        if (forward_plan_) cufftDestroy(forward_plan_);
        if (inverse_plan_) cufftDestroy(inverse_plan_);
        if (wet_) cudaFree(wet_);
        if (wet_convolved_) cudaFree(wet_convolved_);
        if (normal_) cudaFree(normal_);
        if (scatter_) cudaFree(scatter_);
        if (source_fft_) cudaFree(source_fft_);
        if (kernel_) cudaFree(kernel_);
        if (source_) cudaFree(source_);
    }

    size_t imagePixels() const { return static_cast<size_t>(width_) * height_; }

    int registerKernel(const std::vector<float>& centered, int kernel_width,
                       int kernel_height) {
        if (centered.size() != static_cast<size_t>(kernel_width) * kernel_height)
            throw std::runtime_error("invalid fASKS kernel dimensions");
        std::vector<float> padded(image_count_, 0.0f);
        const int radius_u = kernel_width / 2;
        const int radius_v = kernel_height / 2;
        for (int y = -radius_v; y <= radius_v; ++y)
            for (int x = -radius_u; x <= radius_u; ++x) {
                const int py = y < 0 ? padded_height_ + y : y;
                const int px = x < 0 ? padded_width_ + x : x;
                padded[static_cast<size_t>(py) * padded_width_ + px] =
                    centered[static_cast<size_t>(y + radius_v) * kernel_width + x + radius_u];
            }
        check(cudaMemcpy(kernel_, padded.data(), image_count_ * sizeof(float), cudaMemcpyHostToDevice),
              "upload fASKS kernel");
        cufftComplex* spectrum = nullptr;
        check(cudaMalloc(&spectrum, spectrum_count_ * sizeof(cufftComplex)),
              "cudaMalloc fASKS kernel spectrum");
        try {
            checkFft(cufftExecR2C(forward_plan_, kernel_, spectrum), "FFT fASKS kernel");
        } catch (...) {
            cudaFree(spectrum);
            throw;
        }
        kernels_.push_back(spectrum);
        return static_cast<int>(kernels_.size() - 1);
    }

    void begin(const std::vector<float>& wet) {
        check(cudaMemset(scatter_, 0, imagePixels() * sizeof(float)), "clear fASKS scatter");
        check(cudaMemcpy(wet_, wet.data(), imagePixels() * sizeof(float), cudaMemcpyHostToDevice),
              "upload fASKS WET");
    }

    void convolve(const float* host_source, int kernel_id, float* output) {
        check(cudaMemset(source_, 0, image_count_ * sizeof(float)), "clear fASKS source");
        check(cudaMemcpy2D(source_, static_cast<size_t>(padded_width_) * sizeof(float),
                           host_source, static_cast<size_t>(width_) * sizeof(float),
                           static_cast<size_t>(width_) * sizeof(float), height_,
                           cudaMemcpyHostToDevice), "upload fASKS source");
        checkFft(cufftExecR2C(forward_plan_, source_, source_fft_), "FFT fASKS source");
        const int threads = 256;
        const int blocks = static_cast<int>((spectrum_count_ + threads - 1) / threads);
        multiplySpectra<<<blocks, threads>>>(source_fft_, kernels_.at(kernel_id), spectrum_count_);
        check(cudaGetLastError(), "multiply fASKS spectra");
        checkFft(cufftExecC2R(inverse_plan_, source_fft_, source_), "inverse FFT fASKS source");
        const dim3 block(16, 16);
        const dim3 grid((width_ + 15) / 16, (height_ + 15) / 16);
        cropToImage<<<grid, block>>>(source_, output, width_, height_, padded_width_,
                                     1.0f / static_cast<float>(image_count_));
        check(cudaGetLastError(), "crop fASKS convolution");
    }

    void addNormal() {
        const int threads = 256;
        addImage<<<static_cast<int>((imagePixels() + threads - 1) / threads), threads>>>(
            scatter_, normal_, imagePixels());
        check(cudaGetLastError(), "add fASKS normal scatter");
    }

    void addGammaCorrected(double gamma) {
        const int threads = 256;
        accumulateGammaCorrected<<<static_cast<int>((imagePixels() + threads - 1) / threads), threads>>>(
            scatter_, normal_, wet_convolved_, wet_, imagePixels(), static_cast<float>(gamma));
        check(cudaGetLastError(), "add fASKS gamma correction");
    }

    void download(std::vector<float>& output) {
        output.resize(imagePixels());
        check(cudaMemcpy(output.data(), scatter_, imagePixels() * sizeof(float), cudaMemcpyDeviceToHost),
              "download fASKS scatter");
    }

    float* normal() { return normal_; }
    float* wetConvolved() { return wet_convolved_; }

private:
    int width_ = 0, height_ = 0, padded_width_ = 0, padded_height_ = 0;
    size_t image_count_ = 0, spectrum_count_ = 0;
    float* source_ = nullptr;
    float* kernel_ = nullptr;
    float* scatter_ = nullptr;
    float* normal_ = nullptr;
    float* wet_convolved_ = nullptr;
    float* wet_ = nullptr;
    cufftComplex* source_fft_ = nullptr;
    cufftHandle forward_plan_ = 0, inverse_plan_ = 0;
    std::vector<cufftComplex*> kernels_;
};

}

struct CudaFasksScatter::Impl {
    int width = 0;
    int height = 0;
    size_t pixels = 0;
    double pixel_u = 0.0;
    double pixel_v = 0.0;
    const FasksRuntimeModel& model;
    std::vector<int> kernel_ids;
    FftWorkspace workspace;

    Impl(int width_in, int height_in, double pixel_u_in, double pixel_v_in,
         const FasksRuntimeModel& model_in)
        : width(width_in), height(height_in), pixels(static_cast<size_t>(width_in) * height_in),
          pixel_u(pixel_u_in), pixel_v(pixel_v_in), model(model_in),
          workspace(width_in, height_in) {
        if (width <= 0 || height <= 0 || !(pixel_u > 0.0 && pixel_v > 0.0))
            throw std::runtime_error("invalid fASKS detector dimensions");
        const double support_u = std::min(model.supportUMm(),
            std::max((width - 1) * pixel_u, pixel_u));
        const double support_v = std::min(model.supportVMm(),
            std::max((height - 1) * pixel_v, pixel_v));
        const int basis_count = model.discreteKernelCount();
        if (basis_count < 2)
            throw std::runtime_error("fASKS model needs at least two discrete kernels");
        kernel_ids.reserve(static_cast<size_t>(basis_count));
        for (int basis = 0; basis < basis_count; ++basis) {
            const double x = model.discreteAttenuationNode(basis);
            const auto kernel = model.makeDiscreteKernel(
                std::exp(-x), pixel_u, pixel_v, support_u, support_v);
            kernel_ids.push_back(workspace.registerKernel(kernel,
                2 * static_cast<int>(std::ceil(support_u / pixel_u)) + 1,
                2 * static_cast<int>(std::ceil(support_v / pixel_v)) + 1));
        }
    }

    void apply(std::vector<float>& primary, const std::vector<float>& air,
               const std::vector<unsigned char>& mask) {
        if (primary.size() != pixels || air.size() != pixels || mask.size() != pixels)
            throw std::runtime_error("invalid fASKS image dimensions");
        const std::vector<float> edge = edgeWindow(mask, width, height, pixel_u, pixel_v, model);
        std::vector<float> wet(pixels, 0.0f);
        for (size_t p = 0; p < pixels; ++p) {
            if (!std::isfinite(primary[p]) || primary[p] < 0.0f ||
                !std::isfinite(air[p]))
                throw std::runtime_error("fASKS requires finite nonnegative primary and finite air");
            const double raw_t = air[p] > 0.0f
                ? static_cast<double>(primary[p]) / air[p] : 1.0;
            const double physical_t = std::clamp(raw_t,
                std::numeric_limits<double>::min(), 1.0);
            wet[p] = static_cast<float>(model.wetFromLogAttenuation(
                std::log(1.0 / physical_t)));
        }
        smoothNearest(wet, width, height, pixel_u, pixel_v,
                      model.wetSmoothSigmaMm());
        workspace.begin(wet);

        const int basis_count = model.discreteKernelCount();
        const double x_min = model.discreteAttenuationNode(0);
        const double x_max = model.discreteAttenuationNode(basis_count - 1);
        std::vector<std::vector<float>> sources(
            static_cast<size_t>(basis_count), std::vector<float>(pixels, 0.0f));
        for (size_t p = 0; p < pixels; ++p) {
            if (!mask[p] || !(edge[p] > 0.0f)) continue;
            const double raw_t = air[p] > 0.0f
                ? static_cast<double>(primary[p]) / air[p] : 1.0;
            const double source = static_cast<double>(primary[p])
                * model.amplitude(raw_t) * edge[p];
            if (!(source > 0.0) || !std::isfinite(source)) continue;
            const double physical_t = std::clamp(raw_t,
                std::numeric_limits<double>::min(), 1.0);
            const double x = std::log(1.0 / physical_t);
            const double clipped = std::clamp(x, x_min, x_max);
            int lower = 0;
            while (lower + 1 < basis_count &&
                   model.discreteAttenuationNode(lower + 1) <= clipped)
                ++lower;
            const int upper = std::min(lower + 1, basis_count - 1);
            const double lower_x = model.discreteAttenuationNode(lower);
            const double upper_x = model.discreteAttenuationNode(upper);
            const float upper_fraction = upper == lower
                ? 0.0f
                : static_cast<float>((clipped - lower_x) / (upper_x - lower_x));
            sources[lower][p] += static_cast<float>(source) * (1.0f - upper_fraction);
            sources[upper][p] += static_cast<float>(source) * upper_fraction;
        }

        std::vector<float> scatter;
        std::vector<float> wet_source(pixels);
        for (int basis = 0; basis < basis_count; ++basis) {
            const auto& source = sources[basis];
            if (std::none_of(source.begin(), source.end(), [](float value) { return value != 0.0f; }))
                continue;
            workspace.convolve(source.data(), kernel_ids[basis], workspace.normal());
            if (model.gammaPerMm() == 0.0) {
                workspace.addNormal();
            } else {
                for (size_t p = 0; p < pixels; ++p)
                    wet_source[p] = source[p] * wet[p];
                workspace.convolve(wet_source.data(), kernel_ids[basis],
                                   workspace.wetConvolved());
                workspace.addGammaCorrected(model.gammaPerMm());
            }
        }
        workspace.download(scatter);
        const float maximum = *std::max_element(scatter.begin(), scatter.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        const float tolerance = 1e-6f * std::max(1.0f, std::abs(maximum));
        for (size_t p = 0; p < pixels; ++p) {
            if (scatter[p] < 0.0f && scatter[p] > -tolerance) scatter[p] = 0.0f;
            primary[p] += scatter[p];
        }
    }
};

CudaFasksScatter::CudaFasksScatter(int width, int height, double pixel_u_mm,
                                   double pixel_v_mm,
                                   const FasksRuntimeModel& model)
    : impl_(std::make_unique<Impl>(width, height, pixel_u_mm, pixel_v_mm, model)) {}

CudaFasksScatter::~CudaFasksScatter() = default;

void CudaFasksScatter::apply(std::vector<float>& primary,
                             const std::vector<float>& air_primary,
                             const std::vector<unsigned char>& object_mask) {
    impl_->apply(primary, air_primary, object_mask);
}

}
