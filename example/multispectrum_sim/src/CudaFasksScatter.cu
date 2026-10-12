#include "primary/ScatterEdgeWindow.hpp"
#include "common/YkCudaTiming.hpp"
#include "recon/preprocess/DeviceMemoryPool.hpp"
#include "CudaFasksScatter.hpp"
#include "FasksRuntimeModel.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cfloat>
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

__global__ void updateFixedPrimary(float* primary, const float* measured,
    const float* air, const float* scatter, size_t pixels, double damping,
    float scatter_scale, int* invalid) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= pixels) return;
    if (!isfinite(scatter[p])) { atomicExch(invalid, 1); return; }
    const float scaled_scatter = __fmul_rn(scatter_scale, fmaxf(0.0f, scatter[p]));
    const float estimate = fmaxf(__fmul_rn(air[p], 1.0e-8f),
                                __fsub_rn(measured[p], scaled_scatter));
    primary[p] = static_cast<float>((1.0 - damping) * primary[p] + damping * estimate);
    if (!isfinite(primary[p])) atomicExch(invalid, 1);
}

__global__ void multiplyAndSumSpectra(const cufftComplex* spectra,
                                     const cufftComplex* kernels, cufftComplex* total,
                                     int batch, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    cufftComplex sum = {0.0f, 0.0f};
    for (int b = 0; b < batch; ++b) {
        const cufftComplex x = spectra[static_cast<size_t>(b) * count + i];
        const cufftComplex y = kernels[static_cast<size_t>(b) * count + i];
        sum.x += x.x * y.x - x.y * y.y;
        sum.y += x.x * y.y + x.y * y.x;
    }
    total[i] = sum;
}

__global__ void addImage(float* destination, const float* source, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) destination[i] += source[i];
}

__global__ void cropConvolution(float* destination, const float* image,
                               int width, int height, int padded_width, float scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t pixel = static_cast<size_t>(y) * width + x;
    destination[pixel] = image[static_cast<size_t>(y) * padded_width + x] * scale;
}

__global__ void accumulateGammaCorrected(float* destination, const float* normal,
                                         const float* wet_convolved, const float* wet,
                                         size_t count, float gamma) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    destination[i] += (1.0f - gamma * wet[i]) * normal[i]
                    + gamma * wet_convolved[i];
}

__global__ void computeWetFromPrimary(const float* primary, const float* air,
                                      float* wet, size_t count,
                                      const double* attenuation_nodes,
                                      const double* thickness_nodes,
                                      const double* slopes, int node_count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const double raw_t = air[i] > 0.0f
        ? static_cast<double>(primary[i]) / static_cast<double>(air[i]) : 1.0;
    const double physical_t = fmin(fmax(raw_t, static_cast<double>(FLT_MIN)), 1.0);
    double value = log(1.0 / physical_t);
    value = fmin(fmax(value, attenuation_nodes[0]), attenuation_nodes[node_count - 1]);
    int interval = 0;
    while (interval + 2 < node_count && attenuation_nodes[interval + 1] <= value)
        ++interval;
    const double h = attenuation_nodes[interval + 1] - attenuation_nodes[interval];
    const double t = (value - attenuation_nodes[interval]) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    wet[i] = static_cast<float>(
        (2.0 * t3 - 3.0 * t2 + 1.0) * thickness_nodes[interval]
        + (t3 - 2.0 * t2 + t) * h * slopes[interval]
        + (-2.0 * t3 + 3.0 * t2) * thickness_nodes[interval + 1]
        + (t3 - t2) * h * slopes[interval + 1]);
}

__global__ void smoothHorizontalNearest(const float* input, float* output,
                                        int width, int height,
                                        const float* weights, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const int sample_x = max(0, min(width - 1, x + k));
        sum += static_cast<double>(input[static_cast<size_t>(y) * width + sample_x])
             * weights[k + radius];
    }
    output[static_cast<size_t>(y) * width + x] = static_cast<float>(sum);
}

__global__ void smoothVerticalNearest(const float* input, float* output,
                                      int width, int height,
                                      const float* weights, int radius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    double sum = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        const int sample_y = max(0, min(height - 1, y + k));
        sum += static_cast<double>(input[static_cast<size_t>(sample_y) * width + x])
             * weights[k + radius];
    }
    output[static_cast<size_t>(y) * width + x] = static_cast<float>(sum);
}

__global__ void buildBatchSources(float* batch_source, const float* primary,
    const float* air, const float* edge, const double* basis_nodes, int basis_count,
    const double* amplitude_nodes, const double* amplitude_values, int amplitude_count,
    double transmission_max, int width, int height, int padded_width,
    size_t image_count, const float* wet, int* invalid) {
    const int x_pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x_pixel >= width || y >= height) return;
    const size_t p = static_cast<size_t>(y) * width + x_pixel;
    if (!isfinite(primary[p]) || primary[p] < 0 || !isfinite(air[p])) {
        atomicExch(invalid, 1); return;
    }
    if (!(edge[p] > 0)) return;
    const double raw_t = air[p] > 0 ? static_cast<double>(primary[p]) / air[p] : 1.0;
    const double physical_t = fmin(fmax(raw_t, DBL_MIN), 1.0);
    if (physical_t >= 1.0) return;
    const double attenuation = log(1.0 / physical_t);
    const double amplitude_x = fmin(fmax(attenuation, amplitude_nodes[0]),
                                   amplitude_nodes[amplitude_count - 1]);
    int i = 0;
    while (i + 2 < amplitude_count && amplitude_nodes[i + 1] <= amplitude_x) ++i;
    const double fraction = (amplitude_x - amplitude_nodes[i]) /
                            (amplitude_nodes[i + 1] - amplitude_nodes[i]);
    double amplitude = amplitude_values[i] + fraction *
                       (amplitude_values[i + 1] - amplitude_values[i]);
    if (attenuation < amplitude_nodes[0])
        amplitude = transmission_max * amplitude_values[0] * attenuation /
                    amplitude_nodes[0] / physical_t;
    if (!isfinite(amplitude) || amplitude < 0) { atomicExch(invalid, 1); return; }
    const double source = static_cast<double>(primary[p]) * amplitude * edge[p];
    if (!(source > 0) || !isfinite(source)) return;
    const double clipped = fmin(fmax(attenuation, basis_nodes[0]), basis_nodes[basis_count - 1]);
    int lower = 0;
    while (lower + 1 < basis_count && basis_nodes[lower + 1] <= clipped) ++lower;
    const int upper = min(lower + 1, basis_count - 1);
    const float upper_fraction = upper == lower ? 0.0f : static_cast<float>(
        (clipped - basis_nodes[lower]) / (basis_nodes[upper] - basis_nodes[lower]));
    const float signal = static_cast<float>(source);
    const float w = wet ? wet[p] : 1.0f;
    const size_t pixel = static_cast<size_t>(y) * padded_width + x_pixel;
    // Each thread owns this pixel in every basis; no atomics are needed.
    batch_source[static_cast<size_t>(lower) * image_count + pixel] += signal * (1.0f - upper_fraction) * w;
    batch_source[static_cast<size_t>(upper) * image_count + pixel] += signal * upper_fraction * w;
}

__global__ void maximumAbsolute(const float* values, size_t count, unsigned* maximum) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < count) atomicMax(maximum, __float_as_uint(fabsf(values[p])));
}
__global__ void cleanGammaRoundoff(float* values, size_t count, const unsigned* maximum) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count) return;
    const float tolerance = 1e-6f * fmaxf(1.0f, __uint_as_float(*maximum));
    if (values[p] < 0 && values[p] > -tolerance) values[p] = 0;
}

// Exact separable squared-distance transform, matching the CPU edge window.
__global__ void distanceRows(const unsigned char* mask,double* distance,int w,int h,double spacing){
    int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=h)return;int last=-1;
    for(int x=0;x<w;++x){size_t p=size_t(row)*w+x;if(!mask[p])last=x;double d=double(x-last);distance[p]=last<0?INFINITY:spacing*spacing*d*d;}
    last=-1;for(int x=w-1;x>=0;--x){size_t p=size_t(row)*w+x;if(!mask[p])last=x;if(last>=0){double d=double(last-x);distance[p]=fmin(distance[p],spacing*spacing*d*d);}}
}
__global__ void distanceColumns(const unsigned char* mask,const double* input,float* edge,
    int* sites,double* breaks,int w,int h,double spacing,double scale,double ramp){
    int x=blockIdx.x*blockDim.x+threadIdx.x;if(x>=w)return;int* v=sites+size_t(x)*h;double* z=breaks+size_t(x)*(h+1);int k=-1;double a=spacing*spacing;
    for(int q=0;q<h;++q){double fq=input[size_t(q)*w+x];if(!isfinite(fq))continue;
        if(k<0){k=0;v[0]=q;z[0]=-INFINITY;z[1]=INFINITY;continue;}
        double intersection;
        for(;;){int p=v[k];intersection=((fq+a*q*q)-(input[size_t(p)*w+x]+a*p*p))/(2*a*(q-p));if(intersection>z[k])break;--k;}
        ++k;v[k]=q;z[k]=intersection;z[k+1]=INFINITY;
    }
    int interval=0;for(int y=0;y<h;++y){size_t p=size_t(y)*w+x;if(!mask[p]){edge[p]=0;continue;}
        double distance=INFINITY;if(k>=0){while(interval<k&&z[interval+1]<y)++interval;double d=y-v[interval];distance=input[size_t(v[interval])*w+x]+a*d*d;}
        edge[p]=float(scale+(1-scale)*fmin(distance/ramp,1.));
    }
}
std::vector<float> gaussianWeights(double pixel, double sigma) {
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
}

int kernelRadius(int size, double pixel, double support) {
    if (size <= 0 || !std::isfinite(pixel) || pixel <= 0 ||
        !std::isfinite(support) || support <= 0)
        throw std::runtime_error("invalid fASKS convolution geometry");
    const double extent = std::min(support, std::max((size - 1) * pixel, pixel));
    return std::max(1, static_cast<int>(std::ceil(extent / pixel)));
}

int fastFftSize(int minimum) {
    for (int size = minimum; ; ++size) {
        int remaining = size;
        for (int factor : {2, 3, 5})
            while (remaining % factor == 0) remaining /= factor;
        if (remaining == 1) return size;
    }
}

class FftWorkspace {
public:
    FftWorkspace(int width, int height, int radius_u, int radius_v)
        : width_(width), height_(height) {
        // Only the input-sized crop is retained. W + radius prevents wrapped
        // tails entering that crop; 2*radius+1 keeps kernel offsets distinct.
        // A smaller support must also shrink the FFT workspace to save work.
        padded_width_ = fastFftSize(std::max(width_ + radius_u, 2 * radius_u + 1));
        padded_height_ = fastFftSize(std::max(height_ + radius_v, 2 * radius_v + 1));
        image_count_ = static_cast<size_t>(padded_width_) * padded_height_;
        spectrum_count_ = static_cast<size_t>(padded_height_) *
                          (padded_width_ / 2 + 1);
        try {
        check(yk::spectral::preprocess::poolAllocate(&source_, image_count_ * sizeof(float)), "cudaMalloc fASKS source");
        check(yk::spectral::preprocess::poolAllocate(&kernel_, image_count_ * sizeof(float)), "cudaMalloc fASKS kernel");
        check(yk::spectral::preprocess::poolAllocate(&source_fft_, spectrum_count_ * sizeof(cufftComplex)), "cudaMalloc fASKS source FFT");
        check(yk::spectral::preprocess::poolAllocate(&scatter_, imagePixels() * sizeof(float)), "cudaMalloc fASKS scatter");
        check(yk::spectral::preprocess::poolAllocate(&normal_, imagePixels() * sizeof(float)), "cudaMalloc fASKS normal");
        check(yk::spectral::preprocess::poolAllocate(&wet_convolved_, imagePixels() * sizeof(float)), "cudaMalloc fASKS wet convolution");
        check(yk::spectral::preprocess::poolAllocate(&wet_, imagePixels() * sizeof(float)), "cudaMalloc fASKS WET");
        makePlan2d(forward_plan_,forward_work_,CUFFT_R2C);
        makePlan2d(inverse_plan_,inverse_work_,CUFFT_C2R);
        }catch(...){release();throw;}
    }

    ~FftWorkspace() {release();}
    void release()noexcept {
        for (auto spectrum : kernels_) if (spectrum) yk::spectral::preprocess::poolRelease(spectrum);
        if (forward_plan_) cufftDestroy(forward_plan_);
        if (inverse_plan_) cufftDestroy(inverse_plan_);
        if (batch_forward_plan_) cufftDestroy(batch_forward_plan_);
        yk::spectral::preprocess::poolRelease(forward_work_);
        yk::spectral::preprocess::poolRelease(inverse_work_);
        yk::spectral::preprocess::poolRelease(batch_work_);
        if (batch_source_) yk::spectral::preprocess::poolRelease(batch_source_);
        if (batch_fft_) yk::spectral::preprocess::poolRelease(batch_fft_);
        if (batch_kernel_spectra_) yk::spectral::preprocess::poolRelease(batch_kernel_spectra_);
        if (primary_) yk::spectral::preprocess::poolRelease(primary_);
        if (air_) yk::spectral::preprocess::poolRelease(air_);
        if (edge_) yk::spectral::preprocess::poolRelease(edge_);
        if (basis_nodes_) yk::spectral::preprocess::poolRelease(basis_nodes_);
        if (amplitude_nodes_) yk::spectral::preprocess::poolRelease(amplitude_nodes_);
        if (amplitude_values_) yk::spectral::preprocess::poolRelease(amplitude_values_);
        if (invalid_) yk::spectral::preprocess::poolRelease(invalid_);
        if (maximum_) yk::spectral::preprocess::poolRelease(maximum_);
        if (wet_) yk::spectral::preprocess::poolRelease(wet_);
        if (wet_convolved_) yk::spectral::preprocess::poolRelease(wet_convolved_);
        if (normal_) yk::spectral::preprocess::poolRelease(normal_);
        if (scatter_) yk::spectral::preprocess::poolRelease(scatter_);
        if (source_fft_) yk::spectral::preprocess::poolRelease(source_fft_);
        if (kernel_) yk::spectral::preprocess::poolRelease(kernel_);
        if (source_) yk::spectral::preprocess::poolRelease(source_);
    }

    size_t imagePixels() const { return static_cast<size_t>(width_) * height_; }

    int registerKernel(const std::vector<float>& centered, int kernel_width,
                       int kernel_height) {
        YK::diagnostics::TimingSpan watch("scatter.kernel.prepare",std::to_string(kernel_width)+"x"+std::to_string(kernel_height));
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
        check(yk::spectral::preprocess::poolAllocate(&spectrum, spectrum_count_ * sizeof(cufftComplex)),
              "cudaMalloc fASKS kernel spectrum");
        try {
            checkFft(cufftExecR2C(forward_plan_, kernel_, spectrum), "FFT fASKS kernel");
            kernels_.push_back(spectrum);
        } catch (...) {
            yk::spectral::preprocess::poolRelease(spectrum);
            throw;
        }
        return static_cast<int>(kernels_.size() - 1);
    }

    void configureBatch(int batch) {
        batch_ = batch;
        const int dims[2] = {padded_height_, padded_width_};
        int inembed[2] = {padded_height_, padded_width_};
        int onembed[2] = {padded_height_, padded_width_ / 2 + 1};
        const int istride = 1;
        const int ostride = 1;
        const int idist = static_cast<int>(image_count_);
        const int odist = static_cast<int>(spectrum_count_);
        size_t work_bytes=0;
        checkFft(cufftCreate(&batch_forward_plan_),"create fASKS batch plan");
        checkFft(cufftSetAutoAllocation(batch_forward_plan_,0),"disable fASKS batch auto allocation");
        checkFft(cufftMakePlanMany(batch_forward_plan_, 2, const_cast<int*>(dims),
            inembed, istride, idist, onembed, ostride, odist,
            CUFFT_R2C, batch,&work_bytes), "cufftMakePlanMany fASKS forward");
        if(work_bytes)check(yk::spectral::preprocess::poolAllocate(&batch_work_,work_bytes),"allocate pooled FFT batch workspace");
        checkFft(cufftSetWorkArea(batch_forward_plan_,batch_work_),"set pooled FFT batch workspace");
        check(yk::spectral::preprocess::poolAllocate(&batch_source_, static_cast<size_t>(batch_) * image_count_ * sizeof(float)),
              "cudaMalloc fASKS batch source");
        check(yk::spectral::preprocess::poolAllocate(&batch_fft_, static_cast<size_t>(batch_) * spectrum_count_ * sizeof(cufftComplex)),
              "cudaMalloc fASKS batch FFT");
        check(yk::spectral::preprocess::poolAllocate(&batch_kernel_spectra_,
                         static_cast<size_t>(batch_) * spectrum_count_ * sizeof(cufftComplex)),
              "cudaMalloc fASKS batch kernel spectra");
        for (int b = 0; b < batch_; ++b)
            check(cudaMemcpy(batch_kernel_spectra_ + static_cast<size_t>(b) * spectrum_count_,
                             kernels_.at(b), spectrum_count_ * sizeof(cufftComplex),
                             cudaMemcpyDeviceToDevice),
                  "copy fASKS batch kernel spectrum");
    }

    void configureSources(const std::vector<double>& basis, const FasksRuntimeModel& model) {
        amplitude_count_ = static_cast<int>(model.amplitudeValues().size());
        transmission_max_ = model.transmissionMax();
        check(yk::spectral::preprocess::poolAllocate(&primary_, imagePixels() * sizeof(float)), "allocate fASKS primary");
        check(yk::spectral::preprocess::poolAllocate(&air_, imagePixels() * sizeof(float)), "allocate fASKS air");
        check(yk::spectral::preprocess::poolAllocate(&edge_, imagePixels() * sizeof(float)), "allocate fASKS edge window");
        check(yk::spectral::preprocess::poolAllocate(&basis_nodes_, basis.size() * sizeof(double)), "allocate fASKS basis nodes");
        check(yk::spectral::preprocess::poolAllocate(&amplitude_nodes_, model.amplitudeAttenuationNodes().size() * sizeof(double)), "allocate fASKS amplitude nodes");
        check(yk::spectral::preprocess::poolAllocate(&amplitude_values_, model.amplitudeValues().size() * sizeof(double)), "allocate fASKS amplitude values");
        check(yk::spectral::preprocess::poolAllocate(&invalid_, sizeof(int)), "allocate fASKS source error");
        check(yk::spectral::preprocess::poolAllocate(&maximum_, sizeof(unsigned)), "allocate fASKS roundoff maximum");
        check(cudaMemcpy(basis_nodes_, basis.data(), basis.size() * sizeof(double), cudaMemcpyHostToDevice), "upload fASKS basis nodes");
        check(cudaMemcpy(amplitude_nodes_, model.amplitudeAttenuationNodes().data(),
            model.amplitudeAttenuationNodes().size() * sizeof(double), cudaMemcpyHostToDevice), "upload fASKS amplitude nodes");
        check(cudaMemcpy(amplitude_values_, model.amplitudeValues().data(),
            model.amplitudeValues().size() * sizeof(double), cudaMemcpyHostToDevice), "upload fASKS amplitude values");
    }

    void begin() {
        check(cudaMemset(scatter_, 0, imagePixels() * sizeof(float)), "clear fASKS scatter");
        check(cudaMemset(invalid_, 0, sizeof(int)), "clear fASKS source error");
    }

    void convolveBatch(const float* primary, const float* air,
                       bool apply_wet, float* output) {
        check(cudaMemset(batch_source_, 0,
                         static_cast<size_t>(batch_) * image_count_ * sizeof(float)),
              "clear fASKS batch sources");
        const dim3 source_block(16, 16);
        const dim3 source_grid((width_ + 15) / 16, (height_ + 15) / 16);
        YK::diagnostics::cudaTimed("scatter.sources","kernels="+std::to_string(batch_),[&]{
        buildBatchSources<<<source_grid, source_block>>>(batch_source_, primary, air, edge_,
            basis_nodes_, batch_, amplitude_nodes_, amplitude_values_, amplitude_count_,
            transmission_max_, width_, height_, padded_width_, image_count_,
            apply_wet ? wet_ : nullptr, invalid_);
        check(cudaGetLastError(), "generate fASKS sources on CUDA");
        });
        YK::diagnostics::cudaTimed("scatter.fft.forward","kernels="+std::to_string(batch_)+" padded="+std::to_string(padded_width_)+"x"+std::to_string(padded_height_),[&]{
        checkFft(cufftExecR2C(batch_forward_plan_, batch_source_, batch_fft_),
                 "batched FFT fASKS sources");
        });
        const int threads = 256;
        if (batch_kernel_spectra_ == nullptr)
            throw std::runtime_error("fASKS batch kernels are not initialized");
        // Linearity allows summing all kernel contributions before the IFFT.
        // Keep all forward transforms, but perform only one inverse transform.
        YK::diagnostics::cudaTimed("scatter.fft.multiply_sum","kernels="+std::to_string(batch_),[&]{
        multiplyAndSumSpectra<<<static_cast<int>((spectrum_count_ + threads - 1) / threads), threads>>>(
            batch_fft_, batch_kernel_spectra_, source_fft_, batch_, spectrum_count_);
        check(cudaGetLastError(), "sum fASKS spectra");
        });
        YK::diagnostics::cudaTimed("scatter.fft.inverse","",[&]{
        checkFft(cufftExecC2R(inverse_plan_, source_fft_, source_),
                 "single inverse FFT fASKS sources");
        });
        const dim3 block(16, 16);
        const dim3 grid((width_ + 15) / 16, (height_ + 15) / 16);
        YK::diagnostics::cudaTimed("scatter.crop","",[&]{
        cropConvolution<<<grid, block>>>(output, source_, width_, height_, padded_width_,
            1.0f / static_cast<float>(image_count_));
        check(cudaGetLastError(), "crop batched fASKS convolution");
        });
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
    float* wet() { return wet_; }
    float* primary() { return primary_; }
    float* air() { return air_; }
    float* edge() { return edge_; }
    float* scatter() { return scatter_; }

    void finish(bool gamma) {
        if (gamma) {
            check(cudaMemset(maximum_, 0, sizeof(unsigned)), "clear fASKS roundoff maximum");
            const int blocks = static_cast<int>((imagePixels() + 255) / 256);
            maximumAbsolute<<<blocks, 256>>>(scatter_, imagePixels(), maximum_);
            cleanGammaRoundoff<<<blocks, 256>>>(scatter_, imagePixels(), maximum_);
            check(cudaGetLastError(), "clean fASKS gamma roundoff");
        }
        int invalid = 0;
        check(cudaMemcpy(&invalid, invalid_, sizeof(int), cudaMemcpyDeviceToHost), "check fASKS source values");
        if (invalid) throw std::runtime_error("fASKS requires finite nonnegative primary and amplitude, and finite air");
    }

private:
    void makePlan2d(cufftHandle& plan,void*& work,cufftType type){
        size_t bytes=0;checkFft(cufftCreate(&plan),"create fASKS FFT plan");
        checkFft(cufftSetAutoAllocation(plan,0),"disable fASKS FFT auto allocation");
        checkFft(cufftMakePlan2d(plan,padded_height_,padded_width_,type,&bytes),"make fASKS FFT plan");
        if(bytes)check(yk::spectral::preprocess::poolAllocate(&work,bytes),"allocate pooled FFT workspace");
        checkFft(cufftSetWorkArea(plan,work),"set pooled FFT workspace");
    }
    void *forward_work_=nullptr,*inverse_work_=nullptr,*batch_work_=nullptr;
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
    cufftHandle batch_forward_plan_ = 0;
    int batch_ = 0;
    float* batch_source_ = nullptr;
    cufftComplex* batch_fft_ = nullptr;
    cufftComplex* batch_kernel_spectra_ = nullptr;
    std::vector<cufftComplex*> kernels_;
    float *primary_ = nullptr, *air_ = nullptr, *edge_ = nullptr;
    double *basis_nodes_ = nullptr, *amplitude_nodes_ = nullptr, *amplitude_values_ = nullptr;
    int *invalid_ = nullptr, amplitude_count_ = 0;
    unsigned* maximum_ = nullptr;
    double transmission_max_ = 0;
};

class WetWorkspace {
public:
    WetWorkspace(int width, int height, double pixel_u, double pixel_v,
                 const FasksRuntimeModel& model)
        : width_(width), height_(height), pixels_(static_cast<size_t>(width) * height),
          node_count_(static_cast<int>(model.wetAttenuationNodes().size())) {
        if (node_count_ < 2 || model.wetThicknessNodes().size() != node_count_ ||
            model.wetSlopes().size() != node_count_)
            throw std::runtime_error("invalid fASKS WET interpolation table");
        try {
            check(yk::spectral::preprocess::poolAllocate(&raw_, pixels_ * sizeof(float)), "cudaMalloc fASKS raw WET");
            check(yk::spectral::preprocess::poolAllocate(&temporary_, pixels_ * sizeof(float)), "cudaMalloc fASKS temporary WET");
            check(yk::spectral::preprocess::poolAllocate(&attenuation_nodes_, static_cast<size_t>(node_count_) * sizeof(double)),
                  "cudaMalloc fASKS WET attenuation nodes");
            check(yk::spectral::preprocess::poolAllocate(&thickness_nodes_, static_cast<size_t>(node_count_) * sizeof(double)),
                  "cudaMalloc fASKS WET thickness nodes");
            check(yk::spectral::preprocess::poolAllocate(&slopes_, static_cast<size_t>(node_count_) * sizeof(double)),
                  "cudaMalloc fASKS WET slopes");
            check(cudaMemcpy(attenuation_nodes_, model.wetAttenuationNodes().data(),
                             static_cast<size_t>(node_count_) * sizeof(double), cudaMemcpyHostToDevice),
                  "upload fASKS WET attenuation nodes");
            check(cudaMemcpy(thickness_nodes_, model.wetThicknessNodes().data(),
                             static_cast<size_t>(node_count_) * sizeof(double), cudaMemcpyHostToDevice),
                  "upload fASKS WET thickness nodes");
            check(cudaMemcpy(slopes_, model.wetSlopes().data(),
                             static_cast<size_t>(node_count_) * sizeof(double), cudaMemcpyHostToDevice),
                  "upload fASKS WET slopes");

            if (model.wetSmoothSigmaMm() > 0.0) {
                const auto weights_u = gaussianWeights(pixel_u, model.wetSmoothSigmaMm());
                const auto weights_v = gaussianWeights(pixel_v, model.wetSmoothSigmaMm());
                radius_u_ = static_cast<int>(weights_u.size() / 2);
                radius_v_ = static_cast<int>(weights_v.size() / 2);
                check(yk::spectral::preprocess::poolAllocate(&weights_u_, weights_u.size() * sizeof(float)),
                      "cudaMalloc fASKS WET U weights");
                check(yk::spectral::preprocess::poolAllocate(&weights_v_, weights_v.size() * sizeof(float)),
                      "cudaMalloc fASKS WET V weights");
                check(cudaMemcpy(weights_u_, weights_u.data(), weights_u.size() * sizeof(float),
                                 cudaMemcpyHostToDevice), "upload fASKS WET U weights");
                check(cudaMemcpy(weights_v_, weights_v.data(), weights_v.size() * sizeof(float),
                                 cudaMemcpyHostToDevice), "upload fASKS WET V weights");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    ~WetWorkspace() { release(); }

    void compute(const float* primary, const float* air,
                 float* destination) {
        const int threads = 256;
        computeWetFromPrimary<<<static_cast<int>((pixels_ + threads - 1) / threads), threads>>>(
            primary, air, raw_, pixels_, attenuation_nodes_, thickness_nodes_, slopes_, node_count_);
        check(cudaGetLastError(), "compute fASKS WET");
        if (!weights_u_ || !weights_v_) {
            check(cudaMemcpy(destination, raw_, pixels_ * sizeof(float), cudaMemcpyDeviceToDevice),
                  "copy fASKS WET");
            return;
        }
        const dim3 block(16, 16);
        const dim3 grid((width_ + 15) / 16, (height_ + 15) / 16);
        smoothHorizontalNearest<<<grid, block>>>(raw_, temporary_, width_, height_,
                                                 weights_u_, radius_u_);
        check(cudaGetLastError(), "smooth fASKS WET horizontally");
        smoothVerticalNearest<<<grid, block>>>(temporary_, destination, width_, height_,
                                               weights_v_, radius_v_);
        check(cudaGetLastError(), "smooth fASKS WET vertically");
    }

private:
    void release() noexcept {
        if (weights_v_) yk::spectral::preprocess::poolRelease(weights_v_);
        if (weights_u_) yk::spectral::preprocess::poolRelease(weights_u_);
        if (slopes_) yk::spectral::preprocess::poolRelease(slopes_);
        if (thickness_nodes_) yk::spectral::preprocess::poolRelease(thickness_nodes_);
        if (attenuation_nodes_) yk::spectral::preprocess::poolRelease(attenuation_nodes_);
        if (temporary_) yk::spectral::preprocess::poolRelease(temporary_);
        if (raw_) yk::spectral::preprocess::poolRelease(raw_);
        weights_v_ = nullptr;
        weights_u_ = nullptr;
        slopes_ = nullptr;
        thickness_nodes_ = nullptr;
        attenuation_nodes_ = nullptr;
        temporary_ = nullptr;
        raw_ = nullptr;
    }

    int width_ = 0;
    int height_ = 0;
    size_t pixels_ = 0;
    int node_count_ = 0;
    int radius_u_ = 0;
    int radius_v_ = 0;
    float* raw_ = nullptr;
    float* temporary_ = nullptr;
    float* weights_u_ = nullptr;
    float* weights_v_ = nullptr;
    double* attenuation_nodes_ = nullptr;
    double* thickness_nodes_ = nullptr;
    double* slopes_ = nullptr;
};

}

struct CudaFasksScatter::Impl {
    int width = 0;
    int height = 0;
    size_t pixels = 0;
    double pixel_u = 0.0;
    double pixel_v = 0.0;
    const FasksRuntimeModel& model;
    std::vector<double> attenuation_nodes;
    FftWorkspace workspace;
    std::unique_ptr<WetWorkspace> wet_workspace;
    std::vector<unsigned char> cached_mask;
    double *device_distance=nullptr,*device_breaks=nullptr;int* device_sites=nullptr;
    float* fixed_measured = nullptr;
    int* fixed_invalid = nullptr;
    float *guide_weights_u = nullptr, *guide_weights_v = nullptr;
    double guide_sigma_u = -1, guide_sigma_v = -1;

    ~Impl() {
        yk::spectral::preprocess::poolRelease(device_distance);yk::spectral::preprocess::poolRelease(device_breaks);yk::spectral::preprocess::poolRelease(device_sites);
        yk::spectral::preprocess::poolRelease(fixed_measured); yk::spectral::preprocess::poolRelease(fixed_invalid);
        yk::spectral::preprocess::poolRelease(guide_weights_u); yk::spectral::preprocess::poolRelease(guide_weights_v);
    }

    Impl(int width_in, int height_in, double pixel_u_in, double pixel_v_in,
         const FasksRuntimeModel& model_in,
         int transmission_segments)
        : width(width_in), height(height_in), pixels(static_cast<size_t>(width_in) * height_in),
          pixel_u(pixel_u_in), pixel_v(pixel_v_in), model(model_in),
          workspace(width_in, height_in,
              kernelRadius(width_in, pixel_u_in, model_in.supportUMm()),
              kernelRadius(height_in, pixel_v_in, model_in.supportVMm())) {
        if (width <= 0 || height <= 0 || !(pixel_u > 0.0 && pixel_v > 0.0))
            throw std::runtime_error("invalid fASKS detector dimensions");
        const double support_u = std::min(model.supportUMm(),
            std::max((width - 1) * pixel_u, pixel_u));
        const double support_v = std::min(model.supportVMm(),
            std::max((height - 1) * pixel_v, pixel_v));
        if (transmission_segments <= 0) {
            attenuation_nodes.reserve(static_cast<size_t>(model.discreteKernelCount()));
            for (int i = 0; i < model.discreteKernelCount(); ++i)
                attenuation_nodes.push_back(model.discreteAttenuationNode(i));
        } else {
            const int segment_count = std::max(2, transmission_segments);
            const double x_min = model.discreteAttenuationNode(0);
            const double x_max = model.discreteAttenuationNode(model.discreteKernelCount() - 1);
            const double width = (x_max - x_min) / segment_count;
            attenuation_nodes.reserve(static_cast<size_t>(segment_count));
            for (int segment = 0; segment < segment_count; ++segment)
                attenuation_nodes.push_back(x_min + (segment + 0.5) * width);
        }
        const int basis_count = static_cast<int>(attenuation_nodes.size());
        if (basis_count < 2)
            throw std::runtime_error("fASKS model needs at least two discrete kernels");
        for (int basis = 0; basis < basis_count; ++basis) {
            const double x = attenuation_nodes[static_cast<size_t>(basis)];
            const auto kernel = model.makeDiscreteKernel(
                std::exp(-x), pixel_u, pixel_v, support_u, support_v);
            workspace.registerKernel(kernel,
                2 * static_cast<int>(std::ceil(support_u / pixel_u)) + 1,
                2 * static_cast<int>(std::ceil(support_v / pixel_v)) + 1);
        }
        workspace.configureBatch(basis_count);
        workspace.configureSources(attenuation_nodes, model);
        if (model.gammaPerMm() != 0.0)
            wet_workspace = std::make_unique<WetWorkspace>(
                width, height, pixel_u, pixel_v, model);
    }

    void predictOnDevice(const float* primary, const float* air,
                         const std::vector<unsigned char>& mask) {
        if (!primary || !air || mask.size() != pixels)
            throw std::runtime_error("invalid fASKS device image dimensions");
        if (cached_mask != mask) {
            YK::diagnostics::TimingSpan watch("scatter.edge.cpu",std::to_string(width)+"x"+std::to_string(height));
        const auto edge = scatter_detail::edgeWindow(mask, width, height, pixel_u, pixel_v, model);
            check(cudaMemcpy(workspace.edge(), edge.data(), pixels * sizeof(float),
                             cudaMemcpyHostToDevice), "upload fASKS cached edge window");
            cached_mask = mask;
        }
        evaluate(primary,air);
    }
    void evaluate(const float* primary,const float* air){
        YK::diagnostics::TimingSpan watch("scatter.evaluate");
        const bool gamma_enabled = model.gammaPerMm() != 0.0;
        workspace.begin();
        if (gamma_enabled) wet_workspace->compute(primary, air, workspace.wet());
        workspace.convolveBatch(primary, air, false, workspace.normal());
        if (!gamma_enabled) {
            workspace.addNormal();
        } else {
            workspace.convolveBatch(primary, air, true, workspace.wetConvolved());
            workspace.addGammaCorrected(model.gammaPerMm());
        }
        workspace.finish(gamma_enabled);
    }

    void predict(const std::vector<float>& primary, const std::vector<float>& air,
                 const std::vector<unsigned char>& mask, std::vector<float>& scatter) {
        if (primary.size() != pixels || air.size() != pixels || mask.size() != pixels)
            throw std::runtime_error("invalid fASKS image dimensions");
        check(cudaMemcpy(workspace.primary(), primary.data(), pixels * sizeof(float),
                         cudaMemcpyHostToDevice), "upload fASKS primary");
        check(cudaMemcpy(workspace.air(), air.data(), pixels * sizeof(float),
                         cudaMemcpyHostToDevice), "upload fASKS air");
        predictOnDevice(workspace.primary(), workspace.air(), mask);
        workspace.download(scatter);
    }

    void apply(std::vector<float>& primary, const std::vector<float>& air,
               const std::vector<unsigned char>& mask) {
        std::vector<float> scatter;
        predict(primary, air, mask, scatter);
        for (size_t p = 0; p < pixels; ++p) primary[p] += scatter[p];
    }
};

CudaFasksScatter::CudaFasksScatter(int width, int height, double pixel_u_mm,
                                   double pixel_v_mm,
                                   const FasksRuntimeModel& model,
                                   int transmission_segments)
    : impl_(std::make_unique<Impl>(width, height, pixel_u_mm, pixel_v_mm, model,
                                   transmission_segments)) {}

CudaFasksScatter::~CudaFasksScatter() = default;

void CudaFasksScatter::prepareDeviceMask(){
    if(!impl_->device_distance)check(preprocess::poolAllocate(&impl_->device_distance,impl_->pixels*sizeof(double)),"allocate GPU distance rows");
    if(!impl_->device_sites)check(preprocess::poolAllocate(&impl_->device_sites,impl_->pixels*sizeof(int)),"allocate GPU EDT sites");
    if(!impl_->device_breaks)check(preprocess::poolAllocate(&impl_->device_breaks,(impl_->pixels+impl_->width)*sizeof(double)),"allocate GPU EDT breakpoints");
}
void CudaFasksScatter::predictDeviceMask(const float* primary,const float* air,const unsigned char* mask,float* output){
    YK::diagnostics::TimingSpan watch("scatter.predict.device");
    if(!primary||!air||!mask||!output)throw std::runtime_error("invalid resident scatter input");prepareDeviceMask();
    YK::diagnostics::cudaTimed("scatter.edge.cuda","",[&]{
    distanceRows<<<(impl_->height+127)/128,128>>>(mask,impl_->device_distance,impl_->width,impl_->height,impl_->pixel_u);
    distanceColumns<<<(impl_->width+127)/128,128>>>(mask,impl_->device_distance,impl_->workspace.edge(),impl_->device_sites,impl_->device_breaks,
        impl_->width,impl_->height,impl_->pixel_v,impl_->model.edgeBoundaryScale(),impl_->model.edgeRampMm());
    check(cudaGetLastError(),"GPU scatter edge window");    });impl_->cached_mask.clear();impl_->evaluate(primary,air);
    check(cudaMemcpy(output,impl_->workspace.scatter(),impl_->pixels*sizeof(float),cudaMemcpyDeviceToDevice),"copy resident scatter prediction");
}
void CudaFasksScatter::predictDevice(const float* primary, const float* air,
    const std::vector<unsigned char>& mask, float* scatter) {
    if (!scatter) throw std::runtime_error("null fASKS device output");
    impl_->predictOnDevice(primary, air, mask);
    check(cudaMemcpy(scatter, impl_->workspace.scatter(), impl_->pixels * sizeof(float),
                     cudaMemcpyDeviceToDevice), "copy fASKS device scatter");
}

void CudaFasksScatter::predict(const std::vector<float>& primary,
                               const std::vector<float>& air_primary,
                               const std::vector<unsigned char>& object_mask,
                               std::vector<float>& scatter) {
    impl_->predict(primary, air_primary, object_mask, scatter);
}

void CudaFasksScatter::apply(std::vector<float>& primary,
                             const std::vector<float>& air_primary,
                             const std::vector<unsigned char>& object_mask) {
    impl_->apply(primary, air_primary, object_mask);
}

void CudaFasksScatter::correctFixed(const std::vector<float>& measured,
    const std::vector<float>& air, const std::vector<unsigned char>& mask,
    int iterations, double damping, double scale, std::vector<float>& primary) {
    if (measured.size() != impl_->pixels || air.size() != impl_->pixels ||
        mask.size() != impl_->pixels || iterations <= 0 ||
        !std::isfinite(damping) || damping <= 0.0 || damping > 1.0 ||
        !std::isfinite(scale) || scale < 0.0 || scale > FLT_MAX)
        throw std::runtime_error("invalid fixed scatter correction arguments");
    for (size_t p = 0; p < impl_->pixels; ++p)
        if (!std::isfinite(measured[p]) || measured[p] < 0.0f ||
            !std::isfinite(air[p]) || air[p] <= 0.0f)
            throw std::runtime_error("invalid fixed scatter correction signal");
    const size_t bytes = impl_->pixels * sizeof(float);
    if (!impl_->fixed_measured)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_measured, bytes), "allocate fixed scatter measurement");
    if (!impl_->fixed_invalid)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_invalid, sizeof(int)), "allocate fixed scatter validity flag");
    check(cudaMemset(impl_->fixed_invalid, 0, sizeof(int)), "clear fixed scatter validity flag");
    check(cudaMemcpy(impl_->fixed_measured, measured.data(), bytes, cudaMemcpyHostToDevice),
          "upload fixed scatter measurement");
    check(cudaMemcpy(impl_->workspace.primary(), impl_->fixed_measured, bytes, cudaMemcpyDeviceToDevice),
          "initialize fixed scatter primary");
    check(cudaMemcpy(impl_->workspace.air(), air.data(), bytes, cudaMemcpyHostToDevice),
          "upload fixed scatter air");
    constexpr int threads = 256;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        impl_->predictOnDevice(impl_->workspace.primary(), impl_->workspace.air(), mask);
        updateFixedPrimary<<<static_cast<int>((impl_->pixels + threads - 1) / threads), threads>>>(
            impl_->workspace.primary(), impl_->fixed_measured, impl_->workspace.air(),
            impl_->workspace.scatter(), impl_->pixels, damping, static_cast<float>(scale), impl_->fixed_invalid);
        check(cudaGetLastError(), "update fixed scatter primary");
    }
    int invalid = 0;
    check(cudaMemcpy(&invalid, impl_->fixed_invalid, sizeof(int), cudaMemcpyDeviceToHost),
          "check fixed scatter correction result");
    if (invalid) throw std::runtime_error("fixed scatter correction produced a non-finite value");
    primary.resize(impl_->pixels);
    check(cudaMemcpy(primary.data(), impl_->workspace.primary(), bytes, cudaMemcpyDeviceToHost),
          "download fixed scatter primary");
}

void CudaFasksScatter::prepareCorrectionWorkspace(bool guidance) {
    if(!impl_->fixed_measured)check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_measured,impl_->pixels*sizeof(float)),"prepare fixed scatter measurement");
    if(!impl_->fixed_invalid)check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_invalid,sizeof(int)),"prepare fixed scatter validity");
    if(guidance){
        if(!impl_->guide_weights_u)check(yk::spectral::preprocess::poolAllocate(&impl_->guide_weights_u,193*sizeof(float)),"prepare guide U weights");
        if(!impl_->guide_weights_v)check(yk::spectral::preprocess::poolAllocate(&impl_->guide_weights_v,193*sizeof(float)),"prepare guide V weights");
    }
}
void CudaFasksScatter::correctFixedDevice(const float* measured,
    const float* air, const std::vector<unsigned char>& mask,
    int iterations, double damping, double scale, float* output) {
    if (!measured || !air || !output ||
        mask.size() != impl_->pixels || iterations <= 0 ||
        !std::isfinite(damping) || damping <= 0.0 || damping > 1.0 ||
        !std::isfinite(scale) || scale < 0.0 || scale > FLT_MAX)
        throw std::runtime_error("invalid fixed scatter correction arguments");
    const size_t bytes = impl_->pixels * sizeof(float);
    if (!impl_->fixed_measured)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_measured, bytes), "allocate fixed scatter measurement");
    if (!impl_->fixed_invalid)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_invalid, sizeof(int)), "allocate fixed scatter validity flag");
    check(cudaMemset(impl_->fixed_invalid, 0, sizeof(int)), "clear fixed scatter validity flag");
    check(cudaMemcpy(impl_->fixed_measured, measured, bytes, cudaMemcpyDeviceToDevice),
          "upload fixed scatter measurement");
    check(cudaMemcpy(impl_->workspace.primary(), impl_->fixed_measured, bytes, cudaMemcpyDeviceToDevice),
          "initialize fixed scatter primary");
    check(cudaMemcpy(impl_->workspace.air(), air, bytes, cudaMemcpyDeviceToDevice),
          "upload fixed scatter air");
    constexpr int threads = 256;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        impl_->predictOnDevice(impl_->workspace.primary(), impl_->workspace.air(), mask);
        updateFixedPrimary<<<static_cast<int>((impl_->pixels + threads - 1) / threads), threads>>>(
            impl_->workspace.primary(), impl_->fixed_measured, impl_->workspace.air(),
            impl_->workspace.scatter(), impl_->pixels, damping, static_cast<float>(scale), impl_->fixed_invalid);
        check(cudaGetLastError(), "update fixed scatter primary");
    }
    int invalid = 0;
    check(cudaMemcpy(&invalid, impl_->fixed_invalid, sizeof(int), cudaMemcpyDeviceToHost),
          "check fixed scatter correction result");
    if (invalid) throw std::runtime_error("fixed scatter correction produced a non-finite value");
    check(cudaMemcpy(output, impl_->workspace.primary(), bytes, cudaMemcpyDeviceToDevice),
          "download fixed scatter primary");
}

void CudaFasksScatter::correctFixedDeviceMask(const float* measured,
    const float* air, const unsigned char* mask,
    int iterations, double damping, double scale, float* output) {
    if (!measured || !air || !output ||
        !mask || iterations <= 0 ||
        !std::isfinite(damping) || damping <= 0.0 || damping > 1.0 ||
        !std::isfinite(scale) || scale < 0.0 || scale > FLT_MAX)
        throw std::runtime_error("invalid fixed scatter correction arguments");
    const size_t bytes = impl_->pixels * sizeof(float);
    if (!impl_->fixed_measured)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_measured, bytes), "allocate fixed scatter measurement");
    if (!impl_->fixed_invalid)
        check(yk::spectral::preprocess::poolAllocate(&impl_->fixed_invalid, sizeof(int)), "allocate fixed scatter validity flag");
    check(cudaMemset(impl_->fixed_invalid, 0, sizeof(int)), "clear fixed scatter validity flag");
    check(cudaMemcpy(impl_->fixed_measured, measured, bytes, cudaMemcpyDeviceToDevice),
          "upload fixed scatter measurement");
    check(cudaMemcpy(impl_->workspace.primary(), impl_->fixed_measured, bytes, cudaMemcpyDeviceToDevice),
          "initialize fixed scatter primary");
    check(cudaMemcpy(impl_->workspace.air(), air, bytes, cudaMemcpyDeviceToDevice),
          "upload fixed scatter air");
    prepareDeviceMask();
    YK::diagnostics::cudaTimed("scatter.edge.cuda.fixed","",[&]{
        distanceRows<<<(impl_->height+127)/128,128>>>(mask,impl_->device_distance,impl_->width,impl_->height,impl_->pixel_u);
        distanceColumns<<<(impl_->width+127)/128,128>>>(mask,impl_->device_distance,impl_->workspace.edge(),impl_->device_sites,impl_->device_breaks,
            impl_->width,impl_->height,impl_->pixel_v,impl_->model.edgeBoundaryScale(),impl_->model.edgeRampMm());
        check(cudaGetLastError(),"fixed scatter GPU edge window");
    });
    impl_->cached_mask.clear();
    constexpr int threads = 256;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        impl_->evaluate(impl_->workspace.primary(),impl_->workspace.air());
        updateFixedPrimary<<<static_cast<int>((impl_->pixels + threads - 1) / threads), threads>>>(
            impl_->workspace.primary(), impl_->fixed_measured, impl_->workspace.air(),
            impl_->workspace.scatter(), impl_->pixels, damping, static_cast<float>(scale), impl_->fixed_invalid);
        check(cudaGetLastError(), "update fixed scatter primary");
    }
    int invalid = 0;
    check(cudaMemcpy(&invalid, impl_->fixed_invalid, sizeof(int), cudaMemcpyDeviceToHost),
          "check fixed scatter correction result");
    if (invalid) throw std::runtime_error("fixed scatter correction produced a non-finite value");
    check(cudaMemcpy(output, impl_->workspace.primary(), bytes, cudaMemcpyDeviceToDevice),
          "download fixed scatter primary");
}

void CudaFasksScatter::smoothGuide(const std::vector<float>& input, double su,
    double sv, std::vector<float>& output) {
    if (input.size() != impl_->pixels || !std::isfinite(su) || !std::isfinite(sv) ||
        su <= 0 || su > 32 || sv <= 0 || sv > 32)
        throw std::runtime_error("invalid Gaussian guide dimensions or sigma");
    for (float x : input) if (!std::isfinite(x) || x < 0)
        throw std::runtime_error("invalid Gaussian guide signal");
    auto configure = [&](double sigma, double& cached, float*& weights) {
        if (!weights) check(yk::spectral::preprocess::poolAllocate(&weights, 193*sizeof(float)), "allocate guide weights");
        const int radius = static_cast<int>(std::ceil(3*sigma));
        if (cached != sigma) {
            std::vector<float> w(2*radius+1);
            double sum = 0;
            for (int k=-radius;k<=radius;++k) { w[k+radius]=static_cast<float>(std::exp(-.5*k*k/(sigma*sigma))); sum+=w[k+radius]; }
            for (float& x : w) x=static_cast<float>(x/sum);
            check(cudaMemcpy(weights,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice), "upload guide weights");
            cached=sigma;
        }
        return radius;
    };
    const int ru=configure(su,impl_->guide_sigma_u,impl_->guide_weights_u);
    const int rv=configure(sv,impl_->guide_sigma_v,impl_->guide_weights_v);
    const size_t bytes=impl_->pixels*sizeof(float);
    check(cudaMemcpy(impl_->workspace.primary(),input.data(),bytes,cudaMemcpyHostToDevice), "upload Gaussian guide");
    const dim3 block(16,16), grid((impl_->width+15)/16,(impl_->height+15)/16);
    smoothHorizontalNearest<<<grid,block>>>(impl_->workspace.primary(),impl_->workspace.normal(),
        impl_->width,impl_->height,impl_->guide_weights_u,ru);
    check(cudaGetLastError(), "smooth guide U");
    smoothVerticalNearest<<<grid,block>>>(impl_->workspace.normal(),impl_->workspace.air(),
        impl_->width,impl_->height,impl_->guide_weights_v,rv);
    check(cudaGetLastError(), "smooth guide V");
    output.resize(impl_->pixels);
    check(cudaMemcpy(output.data(),impl_->workspace.air(),bytes,cudaMemcpyDeviceToHost), "download Gaussian guide");
}
void CudaFasksScatter::smoothGuideDevice(const float* input, double su,
    double sv, float* output) {
    if (!input || !output || !std::isfinite(su) || !std::isfinite(sv) ||
        su <= 0 || su > 32 || sv <= 0 || sv > 32)
        throw std::runtime_error("invalid Gaussian guide dimensions or sigma");
    auto configure = [&](double sigma, double& cached, float*& weights) {
        if (!weights) check(yk::spectral::preprocess::poolAllocate(&weights, 193*sizeof(float)), "allocate guide weights");
        const int radius = static_cast<int>(std::ceil(3*sigma));
        if (cached != sigma) {
            std::vector<float> w(2*radius+1);
            double sum = 0;
            for (int k=-radius;k<=radius;++k) { w[k+radius]=static_cast<float>(std::exp(-.5*k*k/(sigma*sigma))); sum+=w[k+radius]; }
            for (float& x : w) x=static_cast<float>(x/sum);
            check(cudaMemcpy(weights,w.data(),w.size()*sizeof(float),cudaMemcpyHostToDevice), "upload guide weights");
            cached=sigma;
        }
        return radius;
    };
    const int ru=configure(su,impl_->guide_sigma_u,impl_->guide_weights_u);
    const int rv=configure(sv,impl_->guide_sigma_v,impl_->guide_weights_v);
    const dim3 block(16,16), grid((impl_->width+15)/16,(impl_->height+15)/16);
    smoothHorizontalNearest<<<grid,block>>>(input,impl_->workspace.normal(),
        impl_->width,impl_->height,impl_->guide_weights_u,ru);
    check(cudaGetLastError(), "smooth guide U");
    smoothVerticalNearest<<<grid,block>>>(impl_->workspace.normal(),output,
        impl_->width,impl_->height,impl_->guide_weights_v,rv);
    check(cudaGetLastError(), "smooth guide V");
}

}
