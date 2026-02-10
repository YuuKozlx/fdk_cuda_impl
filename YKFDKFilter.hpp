#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#include "YkConv.hpp"
#include "YkFFT.hpp"
#include "YkFDKCreateFilterKernel.hpp"
#include "YkGlobals.h"

// ============================================================
// logging helpers (可替换成你的 spdlog/plog)
// ============================================================
#ifndef YK_FM_LOGE
#define YK_FM_LOGE(fmt, ...) std::fprintf(stderr, "[YK][FilterManager][E] " fmt "\n", ##__VA_ARGS__)
#endif
#ifndef YK_FM_LOGW
#define YK_FM_LOGW(fmt, ...) std::fprintf(stderr, "[YK][FilterManager][W] " fmt "\n", ##__VA_ARGS__)
#endif
#ifndef YK_FM_LOGI
#define YK_FM_LOGI(fmt, ...) std::fprintf(stdout, "[YK][FilterManager][I] " fmt "\n", ##__VA_ARGS__)
#endif

namespace YK {

    // ============================================================
    // Scale in-place: data *= s
    // (static: avoid multiple definition when included in many TUs)
    // ============================================================
    static __global__ void _kernel_scale_inplace(float* data, int n, float s)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) data[i] *= s;
    }

    // ============================================================
    // FilterManager using CudaFFT + FilterKernelFFT  (LAZY weights)
    // ============================================================
    class FilterManager {
    private:
        CudaFFT fft_batch_;
        FilterKernelFFT kernel_fft_;

        float* d_filter_weights_ = nullptr; // [n_complex]
        cufftComplex* d_complex_buf_ = nullptr; // [batch*n_complex]

        int paddedN_ = 0;
        int n_complex_ = 0;
        int batch_ = 0;

        float du_mm_ = 1.0f; // detector pitch (mm)
        float postScale_ = 1.0f; // 1/(du_mm^2)

        cudaStream_t stream_ = 0;
        bool is_initialized_ = false;

        // ---------- state flags ----------
        bool weights_ready_ = false; // GPU 上是否已有“与当前配置一致”的权重
        bool weights_dirty_ = true;  // 配置变更后置 true，apply 时重建
        // --------------------------------

        // 统一缓存：权重构建的全部配置（包含 source：AnalyticFreq/DiscreteRLFFT/None）
        FilterKernelDesc cached_desc_{};

        // 你也可以把这个做成 settable；先给默认更稳
        FilterKernelFFT::ERampExtractMode extract_mode_ = FilterKernelFFT::ERampExtractMode::RealPart;

    private:
        bool validateInitArgs_(int Nu, int paddedN, float du_mm, int batch) const
        {
            if (Nu <= 0) { YK_FM_LOGE("init failed: Nu(%d) must be > 0.", Nu); return false; }
            if (batch <= 0) { YK_FM_LOGE("init failed: batch(%d) must be > 0.", batch); return false; }
            if (paddedN < Nu) {
                YK_FM_LOGE("init failed: paddedN(%d) must be >= Nu(%d).", paddedN, Nu);
                return false;
            }
            if ((paddedN & (paddedN - 1)) != 0) {
                YK_FM_LOGE("init failed: paddedN(%d) must be power-of-two.", paddedN);
                return false;
            }
            if (!(du_mm > 0.0f)) {
                YK_FM_LOGW("init warning: du_mm(%.6f) <= 0, fallback to 1.0.", du_mm);
            }
            return true;
        }

        bool validateApplyBasic_(const float* d_padded_data) const
        {
            if (!is_initialized_) {
                YK_FM_LOGE("apply aborted: manager not initialized.");
                return false;
            }
            if (!d_padded_data) {
                YK_FM_LOGE("apply aborted: d_padded_data is null.");
                return false;
            }
            if (!d_filter_weights_ || !d_complex_buf_) {
                YK_FM_LOGE("apply aborted: internal buffers null (weights=%p, complex=%p).",
                    (void*)d_filter_weights_, (void*)d_complex_buf_);
                return false;
            }
            if (paddedN_ <= 0 || n_complex_ <= 0 || batch_ <= 0) {
                YK_FM_LOGE("apply aborted: invalid dims paddedN=%d n_complex=%d batch=%d.",
                    paddedN_, n_complex_, batch_);
                return false;
            }
            if (n_complex_ != paddedN_ / 2 + 1) {
                YK_FM_LOGE("apply aborted: n_complex mismatch n_complex=%d but paddedN=%d implies %d.",
                    n_complex_, paddedN_, paddedN_ / 2 + 1);
                return false;
            }
            return true;
        }

        // 核心：lazy build 权重（只在 needed 时触发）
        void ensureWeightsBuilt_()
        {
            if (!is_initialized_) return;

            if (!weights_ready_ || weights_dirty_) {
                kernel_fft_.prepare(paddedN_, stream_);

                cached_desc_.kind = EFilterKernel::RamLak;

                kernel_fft_.build_weights(
                    d_filter_weights_,
                    cached_desc_,
                    /*bake_invN=*/true,
                    extract_mode_);

                YK_FM_LOGI(
                    "weights built (LAZY): source=%d kind=%d cutoff=%.3f gain=%.3f  dc0=%d paddedN=%d mode=%d",
                    (int)cached_desc_.source,
                    (int)cached_desc_.kind,
                    cached_desc_.cutoff,
                    cached_desc_.gain,
                    (int)cached_desc_.force_dc_zero,
                    paddedN_,
                    (int)extract_mode_);

                weights_ready_ = true;
                weights_dirty_ = false;
            }
        }

    public:
        FilterManager() = default;
        ~FilterManager() { release(); }

        // ------------------------------------------------------------
        // init：只初始化资源/plan；不生成权重（lazy）
        // ------------------------------------------------------------
        bool init(int Nu, int paddedN, float du_mm, int batch, cudaStream_t stream = 0)
        {
            release();
            stream_ = stream;
            batch_ = batch;

            if (!validateInitArgs_(Nu, paddedN, du_mm, batch)) return false;

            paddedN_ = paddedN;
            n_complex_ = paddedN_ / 2 + 1;

            du_mm_ = (du_mm > 0.0f) ? du_mm : 1.0f;
            postScale_ = 1.0f / (du_mm_ * du_mm_);

            // FFT plan
            fft_batch_.init(paddedN_, batch_, stream_);

            // buffers
            YK_CUDA_CHECK(cudaMalloc(&d_filter_weights_, (size_t)n_complex_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_, (size_t)batch_ * (size_t)n_complex_ * sizeof(cufftComplex)));

            // kernel generator plan (R2C only inside)
            kernel_fft_.prepare(paddedN_, stream_);

            is_initialized_ = true;

            // -------- default weight config (LAZY, do not build now) --------
            cached_desc_ = FilterKernelDesc{};
            cached_desc_.source = YK::EWeightsBuildSource::DiscreteRLFFT; // 默认离散RL->FFT
            cached_desc_.kind = YK::EFilterKernel::RamLak;              // 默认 RamLak（纯 ramp）
            cached_desc_.cutoff = 0.5f;
            cached_desc_.gain = 1.0f;
            cached_desc_.force_dc_zero = false;                          // 推荐 false
            // --------------------------------------------------------------

            extract_mode_ = FilterKernelFFT::ERampExtractMode::RealPart;

            weights_ready_ = false;
            weights_dirty_ = true;

            YK_FM_LOGI(
                "init ok (LAZY): Nu=%d paddedN=%d n_complex=%d batch=%d du_mm=%.6f postScale=%.6g stream=%p",
                Nu, paddedN_, n_complex_, batch_, du_mm_, postScale_, (void*)stream_);

            YK_FM_LOGI(
                "default weights (LAZY): source=%d kind=%d cutoff=%.3f gain=%.3f dc0=%d",
                (int)cached_desc_.source, (int)cached_desc_.kind,
                cached_desc_.cutoff, cached_desc_.gain, (int)cached_desc_.force_dc_zero);

            return true;
        }

        // stream 切换：保守同步旧 stream，避免 build/FFT 未完成就换流
        void setStream(cudaStream_t stream)
        {
            if (!is_initialized_) {
                YK_FM_LOGW("setStream called before init, stored only. stream=%p", (void*)stream);
                stream_ = stream;
                return;
            }
            if (stream_ != stream) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                stream_ = stream;
                fft_batch_.setStream(stream_);
                kernel_fft_.setStream(stream_);
                YK_FM_LOGI("setStream: switched to stream=%p", (void*)stream_);
            }
        }

        cudaStream_t getStream() const { return stream_; }
        int getPaddedN()  const { return paddedN_; }
        int getNComplex() const { return n_complex_; }
        int batch() const { return batch_; }
        const float* getWeights() const { return d_filter_weights_; }

        // ------------------------------------------------------------
        // du 只影响 postScale（按你的 DU=1 约定，不标脏）
        // ------------------------------------------------------------
        void setDetectorPitchMm(float du_mm)
        {
            du_mm_ = (du_mm > 0.0f) ? du_mm : 1.0f;
            postScale_ = 1.0f / (du_mm_ * du_mm_);
            YK_FM_LOGI("setDetectorPitchMm: du_mm=%.6f postScale=%.6g (weights unchanged)", du_mm_, postScale_);
        }

        float detectorPitchMm() const { return du_mm_; }
        float postScale() const { return postScale_; }

        // ------------------------------------------------------------
        // NEW: 统一设置权重配置（LAZY：只标脏）
        // ------------------------------------------------------------
        void setWeightsDesc(const FilterKernelDesc& desc)
        {
            if (!is_initialized_) {
                YK_FM_LOGE("setWeightsDesc failed: called before init.");
                return;
            }
            cached_desc_ = desc;
            weights_dirty_ = true;

            YK_FM_LOGI(
                "setWeightsDesc (LAZY): marked dirty. source=%d kind=%d cutoff=%.3f gain=%.3f dc0=%d",
                (int)cached_desc_.source,
                (int)cached_desc_.kind,
                cached_desc_.cutoff,
                cached_desc_.gain,
                (int)cached_desc_.force_dc_zero);
        }

        // 可选：设置离散提取模式（只影响 DiscreteRLFFT）
        void setRampExtractMode(FilterKernelFFT::ERampExtractMode mode)
        {
            extract_mode_ = mode;
            // mode 变更会影响离散提取结果，所以标脏
            weights_dirty_ = true;
            YK_FM_LOGI("setRampExtractMode: mode=%d (marked dirty)", (int)mode);
        }

        // 便捷：切换为 AnalyticFreq（不改 desc 其他字段）
        void setWeightsSourceAnalytic()
        {
            if (!is_initialized_) return;
            cached_desc_.source = YK::EWeightsBuildSource::AnalyticFreq;
            weights_dirty_ = true;
            YK_FM_LOGI("setWeightsSourceAnalytic (LAZY): source=AnalyticFreq (marked dirty)");
        }

        // 便捷：切换为 DiscreteRLFFT
        void setWeightsSourceDiscreteRL()
        {
            if (!is_initialized_) return;
            cached_desc_.source = YK::EWeightsBuildSource::DiscreteRLFFT;
            weights_dirty_ = true;
            YK_FM_LOGI("setWeightsSourceDiscreteRL (LAZY): source=DiscreteRLFFT (marked dirty)");
        }

        // ------------------------------------------------------------
        // apply：入口确保权重已构建（lazy）
        // ------------------------------------------------------------
        void apply(float* d_padded_data)
        {
            if (!validateApplyBasic_(d_padded_data)) return;

            // lazy build weights
            ensureWeightsBuilt_();

            if (!weights_ready_) {
                YK_FM_LOGE("apply aborted: weights still not ready after ensureWeightsBuilt_().");
                return;
            }

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

            // post du scaling
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

            weights_ready_ = false;
            weights_dirty_ = true;

            cached_desc_ = FilterKernelDesc{};
            extract_mode_ = FilterKernelFFT::ERampExtractMode::Magnitude;
        }

        // 调试用状态
        bool isInitialized() const { return is_initialized_; }
        bool weightsReady()  const { return weights_ready_; }
        bool weightsDirty()  const { return weights_dirty_; }

        // 读出当前 desc（便于你外部打印/检查）
        FilterKernelDesc weightsDesc() const { return cached_desc_; }
    };

    // ============================================================
    // helper: filtering only (assumes already padded)
    // ============================================================
    inline void executeFdkFiltering_PaddedInPlace(
        FilterManager& manager,
        float* d_padded,   // [batch * paddedN]
        cudaStream_t stream = 0)
    {
        manager.setStream(stream);
        manager.apply(d_padded);
    }

} // namespace YK
