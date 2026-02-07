#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include "YkConv.hpp"
#include "YkFFT.hpp"     // 你给的 CudaFFT（可复用 plan + stream）
#include "YkGlobals.h"   // 你的 YK_CUDA_CHECK / YK_CUFFT_CHECK / YK_KERNEL_CHECK 等

namespace YK {

    // ============================================================
    // 1) Generate spatial Ram-Lak kernel (circularly shifted, n=0 at idx 0)
    // ============================================================
    __global__ void _kernel_gen_spatial_rl_kernel(float* kernel, int paddedN, float du) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u >= paddedN) return;

        int n = (u <= paddedN / 2) ? u : (u - paddedN);
        int an = (n < 0) ? -n : n;

        float val = 0.0f;
        if (n == 0) {
            val = 1.0f / (4.0f * du * du);
        }
        else if (an & 1) { // odd
            const float pi = 3.14159265358979323846f;
            val = -1.0f / (pi * pi * (float)(n * n) * du * du);
        }

        // bake 1/N normalization to cancel cuFFT C2R scaling (C2R has N gain)
        kernel[u] = val / (float)paddedN;
    }

    // ============================================================
    // 2) Extract real-part weights from FFT(kernel)
    // ============================================================
    __global__ void _kernel_extract_fft_weights(const cufftComplex* src, float* dst, int n_complex) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        if (u < n_complex) dst[u] = src[u].x;
    }

    // ============================================================
    // 3) Padding with axis offset (per batch row)
    // ============================================================
    __global__ void _kernel_pad_with_offset(const float* src, float* dst,
        int Nu, int batch, int paddedN, float offsetX) {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (u >= paddedN || b >= batch) return;

        float axis_idx = (Nu - 1) * 0.5f + offsetX;
        // round-to-nearest: avoids systematic 0.5-pixel bias
        int start_u = __float2int_rn(paddedN * 0.5f - axis_idx);

        int dst_idx = b * paddedN + u;
        int su = u - start_u;
        dst[dst_idx] = (su >= 0 && su < Nu) ? src[b * Nu + su] : 0.0f;
    }



    // ============================================================
    // FilterManager using CudaFFT (no raw cufftHandle here)
    // ============================================================
    class FilterManager {
    private:
        // FFT tool (batched) for projections
        CudaFFT fft_batch_;
        // FFT tool (single batch=1) for kernel->weights generation
        CudaFFT fft_single_;

        // precomputed frequency weights (real, length n_complex)
        float* d_filter_weights_ = nullptr;

        // workspace for batched FFT spectrum: [batch * n_complex]
        cufftComplex* d_complex_buf_ = nullptr;

        // workspace for kernel FFT spectrum: [n_complex]
        cufftComplex* d_tmp_complex_ = nullptr;

        // geometry params
        int paddedN_ = 0;
        int n_complex_ = 0;
        int batch_ = 0;

        cudaStream_t stream_ = 0;
        bool is_initialized_ = false;

    public:
        FilterManager() = default;
        ~FilterManager() { release(); }

        bool init(int Nu, float du, int batch, cudaStream_t stream = 0) {
            release();
            stream_ = stream;
            batch_ = batch;

            // paddedN = next pow2 >= 2*Nu
            paddedN_ = 1;
            while (paddedN_ < 2 * Nu) paddedN_ <<= 1;
            n_complex_ = paddedN_ / 2 + 1;

            // init FFT tools (plans bound to stream)
            fft_batch_.init(paddedN_, batch_, stream_);
            fft_single_.init(paddedN_, 1, stream_);

            // allocate persistent buffers
            YK_CUDA_CHECK(cudaMalloc(&d_filter_weights_, n_complex_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_, (size_t)batch_ * n_complex_ * sizeof(cufftComplex)));
            YK_CUDA_CHECK(cudaMalloc(&d_tmp_complex_, n_complex_ * sizeof(cufftComplex)));

            // --- build weights: spatial RL kernel -> FFT -> extract real part ---
            float* d_temp_kernel = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_temp_kernel, paddedN_ * sizeof(float)));

            _kernel_gen_spatial_rl_kernel << <(paddedN_ + 255) / 256, 256, 0, stream_ >> > (d_temp_kernel, paddedN_, du);
            YK_CUDA_KERNEL_CHECK();

            // FFT kernel (batch=1)
            fft_single_.fft(d_temp_kernel, d_tmp_complex_);

            _kernel_extract_fft_weights << <(n_complex_ + 255) / 256, 256, 0, stream_ >> > (d_tmp_complex_, d_filter_weights_, n_complex_);
            YK_CUDA_KERNEL_CHECK();

            YK_CUDA_CHECK(cudaFree(d_temp_kernel));

            is_initialized_ = true;
            return true;
        }

        void setStream(cudaStream_t stream) {
            stream_ = stream;
            if (!is_initialized_) return;
            fft_batch_.setStream(stream_);
            fft_single_.setStream(stream_);
        }

        cudaStream_t getStream() const { return stream_; }
        int getPaddedN() const { return paddedN_; }
        int getNComplex() const { return n_complex_; }
        const float* getWeights() const { return d_filter_weights_; }

        // d_padded_data: [batch * paddedN], in-place filtered
        void apply(float* d_padded_data) {
            if (!is_initialized_) return;

            // FFT (batched)
            fft_batch_.fft(d_padded_data, d_complex_buf_);

            // multiply weights
            dim3 block(256, 1);
            dim3 grid((n_complex_ + 255) / 256, batch_);
            _kernel_pointwise_mul << <grid, block, 0, stream_ >> > (d_complex_buf_, d_filter_weights_, n_complex_, batch_);
            YK_CUDA_KERNEL_CHECK();

            // IFFT (batched)
            fft_batch_.ifft(d_complex_buf_, d_padded_data);
            // normalization already baked into spatial RL kernel (1/paddedN)
        }

        void release() {
            fft_batch_.release();
            fft_single_.release();

            if (d_filter_weights_) { cudaFree(d_filter_weights_); d_filter_weights_ = nullptr; }
            if (d_complex_buf_) { cudaFree(d_complex_buf_);    d_complex_buf_ = nullptr; }
            if (d_tmp_complex_) { cudaFree(d_tmp_complex_);    d_tmp_complex_ = nullptr; }

            paddedN_ = 0;
            n_complex_ = 0;
            batch_ = 0;
            stream_ = 0;
            is_initialized_ = false;
        }
    };

    // ============================================================
    // Helper: padding + filtering on specified stream
    // ============================================================
    inline void executeFdkFiltering_Optimized(FilterManager& manager,
        float* d_input,          // [batch*Nu]
        float* d_output_padded,  // [batch*paddedN]
        int Nu, int batch, float offsetX,
        cudaStream_t stream = 0)
    {
        manager.setStream(stream);

        dim3 block(256, 1);
        dim3 grid_pad((manager.getPaddedN() + 255) / 256, batch);

        _kernel_pad_with_offset << <grid_pad, block, 0, stream >> > (d_input, d_output_padded,
            Nu, batch, manager.getPaddedN(), offsetX);
        YK_CUDA_KERNEL_CHECK();

        manager.apply(d_output_padded);
    }

} // namespace YK
