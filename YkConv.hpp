#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include "YkFFT.hpp"  // 上面的 CudaFFTPlan

namespace YK {

    __global__ void _kernel_pointwise_mul(cufftComplex* data, const float* weights, int n_complex, int batch) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (u < n_complex && b < batch) {
            int idx = b * n_complex + u;
            float w = weights[u];
            data[idx].x *= w;
            data[idx].y *= w;
        }
    }

    class FrequencyFilter {
    public:
        bool init(int paddedN, int batch, cudaStream_t stream = 0) {
            release();
            paddedN_ = paddedN;
            batch_ = batch;
            stream_ = stream;

            n_complex_ = paddedN_ / 2 + 1;

            // 复用 FFT plan
            fft_.init(paddedN_, batch_, stream_);

            // 复用频域缓冲
            YK_CUDA_CHECK(cudaMalloc(&d_complex_, (size_t)batch_ * n_complex_ * sizeof(cufftComplex)));
            return true;
        }

        // d_padded_data: [batch * paddedN]
        void apply(float* d_padded_data, const float* d_filter_fft /*[n_complex]*/) {
            fft_.fft(d_padded_data, d_complex_);

            dim3 block(256, 1);
            dim3 grid((n_complex_ + 255) / 256, batch_);
            _kernel_pointwise_mul << <grid, block, 0, stream_ >> > (d_complex_, d_filter_fft, n_complex_, batch_);
            YK_CUDA_CHECK(cudaGetLastError());

            fft_.ifft(d_complex_, d_padded_data);
        }

        void setStream(cudaStream_t s) {
            stream_ = s;
            fft_.setStream(s);
        }

        void release() {
            fft_.release();
            if (d_complex_) { cudaFree(d_complex_); d_complex_ = nullptr; }
            paddedN_ = 0; n_complex_ = 0; batch_ = 0; stream_ = 0;
        }

        ~FrequencyFilter() { release(); }

    private:
        int paddedN_ = 0;
        int n_complex_ = 0;
        int batch_ = 0;
        cudaStream_t stream_ = 0;

        CudaFFT fft_;
        cufftComplex* d_complex_ = nullptr;
    };

} // namespace YK
