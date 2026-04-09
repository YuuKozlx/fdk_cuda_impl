#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#include "global/IProcessor.hpp"
#include "Filter/YkConv.hpp"
#include "Filter/YkFFT.hpp"
#include "Filter/YkFilterKernel.cuh"
#include "FDK/YkFdkPipelineContext.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"



namespace YK {
    namespace cg = cooperative_groups;

    // ============================================================
    // Policy
    // ============================================================
    inline SKernelLaunchPolicy normalizeFilterPolicy(SKernelLaunchPolicy p) {
        if (p.block_threads < 32) p.block_threads = 32;
        p.block_threads = (p.block_threads + 31) & ~31;
        p.block_threads = std::min(p.block_threads, 1024);
        return p;
    }

    // ============================================================
    // Device helpers
    // ============================================================
    __host__ __device__ __forceinline__
        int fp_computePaddedN(int Nu) {
        int n = 1;
        while (n < 2 * Nu) n <<= 1;
        return n;
    }

    __host__ __device__ __forceinline__
        int fp_computeStartU(int Nu, int paddedN, float offsetU_pix) {
        const float axis = (Nu - 1) * 0.5f + offsetU_pix;
        return (int)lrintf(paddedN * 0.5f - axis);
    }

    // ============================================================
    // Kernels
    // ============================================================
    static __global__ void _fp_scale_inplace(float* data, int n, float s) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) data[i] *= s;
    }

    __global__ void _fp_pad_kernel(
        const float* __restrict__ src,
        float* __restrict__ dst,
        const int* __restrict__ startu,
        int Nu, int Nv, int paddedN, int K, int bounds_check)
    {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

        const int wpb = int(tb.size() / 32);
        const int warp_global = int(blockIdx.x) * wpb + int(tb.thread_rank() / 32);
        if (warp_global >= K * Nv) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;
        const int sU = startu[i];

        const size_t base_src = ((size_t)i * Nv + v) * Nu;
        const size_t base_dst = ((size_t)i * Nv + v) * paddedN;

        for (int u = (int)warp.thread_rank(); u < paddedN; u += 32) {
            const int su = u - sU;
            float val = 0.0f;
            if (!bounds_check)                    val = src[base_src + su];
            else if ((unsigned)su < (unsigned)Nu) val = src[base_src + su];
            dst[base_dst + u] = val;
        }
    }

    __global__ void _fp_crop_kernel(
        const float* __restrict__ src,
        float* __restrict__ dst,
        const int* __restrict__ startu,
        int Nu, int Nv, int paddedN, int K, int bounds_check)
    {
        cg::thread_block tb = cg::this_thread_block();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(tb);

        const int wpb = int(tb.size() / 32);
        const int warp_global = int(blockIdx.x) * wpb + int(tb.thread_rank() / 32);
        if (warp_global >= K * Nv) return;

        const int i = warp_global / Nv;
        const int v = warp_global - i * Nv;
        const int sU = startu[i];

        const size_t base_src = ((size_t)i * Nv + v) * paddedN;
        const size_t base_dst = ((size_t)i * Nv + v) * Nu;

        for (int u = (int)warp.thread_rank(); u < Nu; u += 32) {
            const int su = sU + u;
            float val = 0.0f;
            if (!bounds_check)                          val = src[base_src + su];
            else if ((unsigned)su < (unsigned)paddedN)  val = src[base_src + su];
            dst[base_dst + u] = val;
        }
    }

    // ============================================================
    // FilterProcessor
    //
    //   生命周期：
    //     setInitContext(&FdkFilterInitContext{...})
    //     init()
    //     loop:
    //       setContext(&FdkFilterContext{ h_gv+base, K })
    //       process(d_in, d_out, stream)
    //     release()
    //
    //   流水：[K*Nv*Nu] --pad--> [K*Nv*paddedN]
    //                  --FFT·W·IFFT--> [K*Nv*paddedN]
    //                  --crop--> [K*Nv*Nu]
    // ============================================================
    class FilterProcessor : public IProcessor {
    public:
        FilterProcessor() = default;
        ~FilterProcessor() override { release(); }

        FilterProcessor(const FilterProcessor&) = delete;
        FilterProcessor& operator=(const FilterProcessor&) = delete;

        // ----------------------------------------------------------------
        // IProcessor::setInitContext
        // ----------------------------------------------------------------
        void setInitContext(const void* ctx) override
        {
            if (!ctx) { YK_LOGE("setInitContext: null."); return; }
            const auto* ic = static_cast<const FdkFilterInitContext*>(ctx);

            cfg_.Nu = (int)ic->dims.iPU;
            cfg_.Nv = (int)ic->dims.iPV;
            cfg_.K = (int)ic->dims.iPAng;   // Kchunk
            cfg_.desc = ic->desc;
            cfg_.policy = normalizeFilterPolicy(ic->policy);
            cfg_.stream = ic->stream;
            cfg_ready_ = true;

            YK_LOGI("setInitContext: Nu=%d Nv=%d Kchunk=%d stream=%p",
                cfg_.Nu, cfg_.Nv, cfg_.K, (void*)cfg_.stream);
        }

        // ----------------------------------------------------------------
        // IProcessor::init
        // ----------------------------------------------------------------
        bool init() override
        {
            release();

            if (!cfg_ready_) {
                YK_LOGE("init failed: setInitContext() not called."); return false;
            }
            if (!validateCfg_()) return false;

            Nu_ = cfg_.Nu;
            Nv_ = cfg_.Nv;
            K_ = cfg_.K;
            paddedN_ = fp_computePaddedN(Nu_);
            n_cmplx_ = paddedN_ / 2 + 1;
            stream_ = cfg_.stream;
            policy_ = cfg_.policy;
            desc_ = cfg_.desc;

            // startU buffer（大小 K_，内容由 setContext 填充）
            YK_CUDA_CHECK(cudaMalloc(&d_startu_,
                (size_t)K_ * sizeof(int)));

            // padded 中间缓冲
            YK_CUDA_CHECK(cudaMalloc(&d_padded_,
                (size_t)K_ * Nv_ * paddedN_ * sizeof(float)));

            // 滤波权重 + 复数缓冲（batch = K_*Nv_ 行）
            YK_CUDA_CHECK(cudaMalloc(&d_weights_,
                (size_t)n_cmplx_ * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_,
                (size_t)K_ * Nv_ * n_cmplx_ * sizeof(cufftComplex)));

            fft_batch_.init(paddedN_, K_ * Nv_, stream_);
            kernel_fft_.prepare(paddedN_, stream_);

            is_initialized_ = true;
            weights_ready_ = false;
            weights_dirty_ = true;
            current_K_ = 0;   // 等待 setContext()

            YK_LOGI(
                "init ok: Nu=%d Nv=%d Kchunk=%d paddedN=%d n_cmplx=%d stream=%p",
                Nu_, Nv_, K_, paddedN_, n_cmplx_, (void*)stream_);

            return true;
        }

        // ----------------------------------------------------------------
        // IProcessor::setContext — 每 chunk 前调用
        // ----------------------------------------------------------------
        void setContext(const void* ctx) override
        {
            if (!is_initialized_) {
                YK_LOGE("setContext: not initialized."); return;
            }
            if (!ctx) {
                YK_LOGE("setContext: null."); return;
            }

            const auto* fc = static_cast<const FdkFilterContext*>(ctx);

            if (!fc->h_gv) {
                YK_LOGE("setContext: h_gv is null."); return;
            }
            if (fc->K <= 0 || fc->K > K_) {
                YK_LOGE("setContext: K(%d) out of range [1, %d].", fc->K, K_); return;
            }

            current_K_ = fc->K;

            // 重算 startU 并上传
            host_startu_.resize(current_K_);
            for (int i = 0; i < current_K_; ++i)
                host_startu_[i] = fp_computeStartU(
                    Nu_, paddedN_, fc->h_gv[i].offsetU_pix);

            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_startu_, host_startu_.data(),
                (size_t)current_K_ * sizeof(int),
                cudaMemcpyHostToDevice, stream_));

            const float du = (fc->h_gv[0].du_mm > 0.0f) ? fc->h_gv[0].du_mm : 1.0f;
            postScale_ = 1.0f / du;
        }

        // ----------------------------------------------------------------
        // IProcessor::process
        // ----------------------------------------------------------------
        void process(const void* d_input,
            void* d_output,
            cudaStream_t stream = 0) override
        {
            const float* d_in = static_cast<const float*>(d_input);
            float* d_out = static_cast<float*>(d_output);

            if (!validateApply_(d_in, d_out)) return;
            if (stream != 0 && stream != stream_) setStream_(stream);

            ensureWeights_();
            if (!weights_ready_) {
                YK_LOGE("process aborted: weights not ready."); return;
            }

            launchPad_(d_in, current_K_);
            runFilter_(current_K_);
            launchCrop_(d_out, current_K_);
        }

        // ----------------------------------------------------------------
        // IProcessor::release
        // ----------------------------------------------------------------
        void release() override
        {
            fft_batch_.release();
            kernel_fft_.release();

            if (d_startu_) { cudaFree(d_startu_);      d_startu_ = nullptr; }
            if (d_padded_) { cudaFree(d_padded_);      d_padded_ = nullptr; }
            if (d_weights_) { cudaFree(d_weights_);     d_weights_ = nullptr; }
            if (d_complex_buf_) { cudaFree(d_complex_buf_); d_complex_buf_ = nullptr; }

            host_startu_.clear();

            Nu_ = Nv_ = K_ = paddedN_ = n_cmplx_ = 0;
            postScale_ = 1.0f;
            current_K_ = 0;
            stream_ = 0;
            is_initialized_ = false;
            weights_ready_ = false;
            weights_dirty_ = true;
            desc_ = {};
        }

        // ----------------------------------------------------------------
        // IProcessor 状态查询
        // ----------------------------------------------------------------
        bool        isInitialized() const override { return is_initialized_; }
        const char* name()          const override { return "FilterProcessor"; }

        // 调试
        int   Nu()           const { return Nu_; }
        int   Nv()           const { return Nv_; }
        int   Kchunk()       const { return K_; }
        int   currentK()     const { return current_K_; }
        int   paddedN()      const { return paddedN_; }
        int   nComplex()     const { return n_cmplx_; }
        float postScale()    const { return postScale_; }
        bool  weightsDirty() const { return weights_dirty_; }

    private:
        // ---- 配置暂存 ----
        struct Cfg {
            int                 Nu = 0;
            int                 Nv = 0;
            int                 K = 0;
            SFilterKernelDesc    desc = {};
            SKernelLaunchPolicy policy = {};
            cudaStream_t        stream = 0;
        } cfg_;
        bool cfg_ready_ = false;

        // ---- 运行期状态 ----
        int          Nu_ = 0;
        int          Nv_ = 0;
        int          K_ = 0;       // Kchunk（plan 容量）
        int          current_K_ = 0;       // 本次实际视角数（setContext 注入）
        int          paddedN_ = 0;
        int          n_cmplx_ = 0;
        float        postScale_ = 1.0f;
        cudaStream_t stream_ = 0;

        SKernelLaunchPolicy policy_ = {};
        SFilterKernelDesc    desc_ = {};

        // GPU buffers
        int* d_startu_ = nullptr; // [K_]
        float* d_padded_ = nullptr; // [K_*Nv_*paddedN_]
        float* d_weights_ = nullptr; // [n_cmplx_]
        cufftComplex* d_complex_buf_ = nullptr; // [K_*Nv_*n_cmplx_]

        CudaFFT         fft_batch_;
        YK::Filter::FilterKernelFFT kernel_fft_;

        std::vector<int> host_startu_;

        bool is_initialized_ = false;
        bool weights_ready_ = false;
        bool weights_dirty_ = true;

        // ---- 内部辅助 ----

        bool validateCfg_() const
        {
            if (cfg_.Nu <= 0 || cfg_.Nv <= 0 || cfg_.K <= 0) {
                YK_LOGE("validateCfg: invalid dims Nu=%d Nv=%d K=%d.",
                    cfg_.Nu, cfg_.Nv, cfg_.K);
                return false;
            }
            return true;
        }

        bool validateApply_(const float* d_in, const float* d_out) const
        {
            if (!is_initialized_) {
                YK_LOGE("process aborted: not initialized."); return false;
            }
            if (!d_in) {
                YK_LOGE("process aborted: d_input is null.");  return false;
            }
            if (!d_out) {
                YK_LOGE("process aborted: d_output is null."); return false;
            }
            if (current_K_ <= 0) {
                YK_LOGE("process aborted: setContext() not called."); return false;
            }
            return true;
        }

        void ensureWeights_()
        {
            if (weights_ready_ && !weights_dirty_) return;

            kernel_fft_.prepare(paddedN_, stream_);
            kernel_fft_.build_weights(d_weights_, desc_, /*bake_invN=*/true);

            YK_LOGI(
                "weights built (lazy): source=%d kind=%d cutoff=%.3f "
                "gain=%.3f dc0=%d paddedN=%d",
                (int)desc_.source, (int)desc_.kind,
                desc_.cutoff, desc_.gain,
                (int)desc_.force_dc_zero, paddedN_);

            weights_ready_ = true;
            weights_dirty_ = false;
        }

        void launchPad_(const float* d_src, int K)
        {
            const int bk = policy_.block_threads;
            const int wpb = bk / 32;
            const int blk = (K * Nv_ + wpb - 1) / wpb;
            _fp_pad_kernel << <blk, bk, 0, stream_ >> > (
                d_src, d_padded_, d_startu_,
                Nu_, Nv_, paddedN_, K,
                policy_.bounds_check ? 1 : 0);
            YK_CUDA_KERNEL_CHECK();
        }

        void runFilter_(int K)
        {
            const int batch = K * Nv_;

            fft_batch_.fft(d_padded_, d_complex_buf_);

            dim3 blk(256);
            dim3 grd((n_cmplx_ + 255) / 256, batch);
            _kernel_pointwise_mul << <grd, blk, 0, stream_ >> > (
                d_complex_buf_, d_weights_, n_cmplx_, batch);
            YK_CUDA_KERNEL_CHECK();

            fft_batch_.ifft(d_complex_buf_, d_padded_);

            if (postScale_ != 1.0f) {
                const int total = batch * paddedN_;
                _fp_scale_inplace << <(total + 255) / 256, 256, 0, stream_ >> > (
                    d_padded_, total, postScale_);
                YK_CUDA_KERNEL_CHECK();
            }
        }

        void launchCrop_(float* d_dst, int K)
        {
            const int bk = policy_.block_threads;
            const int wpb = bk / 32;
            const int blk = (K * Nv_ + wpb - 1) / wpb;
            _fp_crop_kernel << <blk, bk, 0, stream_ >> > (
                d_padded_, d_dst, d_startu_,
                Nu_, Nv_, paddedN_, K,
                policy_.bounds_check ? 1 : 0);
            YK_CUDA_KERNEL_CHECK();
        }

        void setStream_(cudaStream_t s)
        {
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            stream_ = s;
            fft_batch_.setStream(s);
            kernel_fft_.setStream(s);
            YK_LOGI("setStream_: switched to %p", (void*)s);
        }
    };

} // namespace YK