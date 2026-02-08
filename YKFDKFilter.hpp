#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include "YkConv.hpp"
#include "YkFFT.hpp"
#include "YkFDKCreateFilterKernel.hpp"
#include "YkGlobals.h"

namespace YK {

    // ============================================================
    // Padding with axis offset (per batch row)
    // ============================================================
    __global__ void _kernel_pad_with_offset(const float* src, float* dst,
        int Nu, int batch, int paddedN, float offsetX)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int b = blockIdx.y;
        if (u >= paddedN || b >= batch) return;

        float axis_idx = (Nu - 1) * 0.5f + offsetX;
        int start_u = __float2int_rn(paddedN * 0.5f - axis_idx);

        int dst_idx = b * paddedN + u;
        int su = u - start_u;
        dst[dst_idx] = (su >= 0 && su < Nu) ? src[b * Nu + su] : 0.0f;
    }

    // ============================================================
    // Scale in-place: data *= s
    // ============================================================
    __global__ void _kernel_scale_inplace(float* data, int n, float s)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) data[i] *= s;
    }

    // ============================================================
    // FilterManager using CudaFFT + FilterKernelFFT
    //
    // ✅ Conventions (fully unified):
    //  - Frequency weights are ALWAYS DU=1 convention (NO du inside)
    //  - Weights ALWAYS bake 1/paddedN (cancel cuFFT C2R N gain)
    //  - du scaling ALWAYS happens here (postScale = 1/(du_mm^2))
    //
    // This removes all "du_in_kernel" ambiguity.
    // ============================================================
    class FilterManager {
    public:
        enum class EWeightsSource {
            Analytic,         // build_analytic()
            DiscreteRamLakDu1 // build_discrete_ramlak_du1()
        };

    private:
        CudaFFT fft_batch_;
        FilterKernelFFT kernel_fft_;

        float* d_filter_weights_ = nullptr;     // [n_complex]
        cufftComplex* d_complex_buf_ = nullptr; // [batch*n_complex]

        int paddedN_ = 0;
        int n_complex_ = 0;
        int batch_ = 0;

        float du_mm_ = 1.0f;     // store detector pitch (mm)
        float postScale_ = 1.0f; // = 1/(du_mm^2)

        cudaStream_t stream_ = 0;
        bool is_initialized_ = false;

        EWeightsSource weights_src_ = EWeightsSource::Analytic;

    public:
        FilterManager() = default;
        ~FilterManager() { release(); }

        // ------------------------------------------------------------
        // init (default: analytic RamLak)
        // ------------------------------------------------------------
        bool init(int Nu, float du_mm, int batch, cudaStream_t stream = 0)
        {
            release();
            stream_ = stream;
            batch_ = batch;

            paddedN_ = 1;
            while (paddedN_ < 2 * Nu) paddedN_ <<= 1;
            n_complex_ = paddedN_ / 2 + 1;

            du_mm_ = du_mm;
            postScale_ = (du_mm_ > 0.0f) ? (1.0f / (du_mm_ * du_mm_)) : 1.0f;

            fft_batch_.init(paddedN_, batch_, stream_);

            YK_CUDA_CHECK(cudaMalloc(&d_filter_weights_, (size_t)n_complex_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_, (size_t)batch_ * (size_t)n_complex_ * sizeof(cufftComplex)));

            kernel_fft_.prepare(paddedN_, stream_);

            // Default: analytic RamLak (DU=1 weights), bake invN
            //FilterKernelDesc desc;
            //desc.kind = EFilterKernel::RamLak;
            //desc.cutoff = 0.5f;
            //desc.gain = 1.0f;
            //desc.normalized_ramp = true;

            //setAnalyticWeights(desc);
            setDiscreteRamLakWeightsDu1();
            

            is_initialized_ = true;
            return true;
        }

        void setStream(cudaStream_t stream)
        {
            stream_ = stream;
            if (!is_initialized_) return;
            fft_batch_.setStream(stream_);
            kernel_fft_.setStream(stream_);
        }

        cudaStream_t getStream() const { return stream_; }
        int getPaddedN() const { return paddedN_; }
        int getNComplex() const { return n_complex_; }
        const float* getWeights() const { return d_filter_weights_; }

        // optional: allow changing du after init
        void setDetectorPitchMm(float du_mm)
        {
            du_mm_ = du_mm;
            postScale_ = (du_mm_ > 0.0f) ? (1.0f / (du_mm_ * du_mm_)) : 1.0f;
        }

        float detectorPitchMm() const { return du_mm_; }
        float postScale() const { return postScale_; }

        // ------------------------------------------------------------
        // Build analytic weights (DU=1), bake 1/N
        // ------------------------------------------------------------
        void setAnalyticWeights(const FilterKernelDesc& desc)
        {
            if (!is_initialized_ && paddedN_ == 0) return;

            weights_src_ = EWeightsSource::Analytic;

            kernel_fft_.prepare(paddedN_, stream_);
            kernel_fft_.build_analytic(d_filter_weights_, desc, /*bake_invN=*/true);
        }

        // ------------------------------------------------------------
        // Build discrete RamLak (DU=1): spatial kernel -> FFT -> RealPart
        // bake 1/N inside spatial kernel
        // ------------------------------------------------------------
        void setDiscreteRamLakWeightsDu1()
        {
            if (!is_initialized_ && paddedN_ == 0) return;

            weights_src_ = EWeightsSource::DiscreteRamLakDu1;

            kernel_fft_.prepare(paddedN_, stream_);
            kernel_fft_.build_discrete_ramlak_du1(
                d_filter_weights_,
                /*bake_invN=*/true,
                FilterKernelFFT::EKernelToWeightsMode::RealPart);
        }

        // ------------------------------------------------------------
        // Apply filtering in-place on padded data: [batch*paddedN]
        // ------------------------------------------------------------
        void apply(float* d_padded_data)
        {
            if (!is_initialized_) return;

            // FFT (batched)
            fft_batch_.fft(d_padded_data, d_complex_buf_);

            // multiply weights
            dim3 block(256, 1);
            dim3 grid((n_complex_ + block.x - 1) / block.x, batch_);
            _kernel_pointwise_mul << <grid, block, 0, stream_ >> > (
                d_complex_buf_, d_filter_weights_, n_complex_, batch_);
            YK_CUDA_KERNEL_CHECK();

            // IFFT (batched)
            fft_batch_.ifft(d_complex_buf_, d_padded_data);

            // ✅ No 1/paddedN here (weights already baked invN)
            // ✅ Always apply du scaling here (DU belongs to filter module)
            if (postScale_ != 1.0f) {
                const int total = batch_ * paddedN_;
                dim3 b2(256, 1);
                dim3 g2((total + b2.x - 1) / b2.x, 1);
                _kernel_scale_inplace << <g2, b2, 0, stream_ >> > (
                    d_padded_data, total, postScale_);
                YK_CUDA_KERNEL_CHECK();
            }
        }

        void release()
        {
            fft_batch_.release();
            kernel_fft_.release();

            if (d_filter_weights_) { YK_CUDA_CHECK(cudaFree(d_filter_weights_)); d_filter_weights_ = nullptr; }
            if (d_complex_buf_) { YK_CUDA_CHECK(cudaFree(d_complex_buf_));    d_complex_buf_ = nullptr; }

            paddedN_ = 0;
            n_complex_ = 0;
            batch_ = 0;

            du_mm_ = 1.0f;
            postScale_ = 1.0f;

            stream_ = 0;
            is_initialized_ = false;
            weights_src_ = EWeightsSource::Analytic;
        }
    };

    // ============================================================
    // Helper: padding + filtering on specified stream
    // ============================================================
    inline void executeFdkFiltering_Optimized(
        FilterManager& manager,
        float* d_input,          // [batch*Nu]
        float* d_output_padded,  // [batch*paddedN]
        int Nu, int batch, float offsetX,
        cudaStream_t stream = 0)
    {
        manager.setStream(stream);

        dim3 block(256, 1);
        dim3 grid_pad((manager.getPaddedN() + block.x - 1) / block.x, batch);

        _kernel_pad_with_offset << <grid_pad, block, 0, stream >> > (
            d_input, d_output_padded,
            Nu, batch, manager.getPaddedN(), offsetX);
        YK_CUDA_KERNEL_CHECK();

        manager.apply(d_output_padded);
    }

} // namespace YK
