#pragma once
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "FDK/YkFdkStageTypes.hpp"
#include "Filter/YkConv.hpp"
#include "Filter/YkFFT.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"

#include "FDK/kernels/YkFDKFilterHelpers.cuh"  // normalizeFilterPolicy / fp_computePaddedN / fp_computeStartU
#include "FDK/kernels/YkFDKFilterLaunch.cuh"   // fp_launchPad / fp_launchFilter / fp_launchCrop
#include "Filter/YkCreateFilterKernel.cuh"
#include "global/YkLog.h"

namespace YK {
    namespace Fdk {

        // ============================================================
        // FilterProcessor
        //
        //   生命周期：prepare(config) -> apply(input, output, chunk) -> release()
        //
        //   流水：[K*Nv*Nu] --pad--> [K*Nv*paddedN]
        //                  --FFT·W·IFFT--> [K*Nv*paddedN]
        //                  --crop--> [K*Nv*Nu]
        // ============================================================
        class FilterProcessor {
        public:
            FilterProcessor() = default;
            ~FilterProcessor() { release(); }

            FilterProcessor(const FilterProcessor&) = delete;
            FilterProcessor& operator=(const FilterProcessor&) = delete;

            // 强类型初始化入口。滤波器需要为最大 chunk 建立 FFT 工作区；尾包
            // 通过 apply() 的 context.K 直接处理，不再为了改变 K 反复重建 plan。
            bool prepare(const FdkFilterConfig& config)
            {
                release();
                cfg_.Nu = static_cast<int>(config.dims.iPU);
                cfg_.Nv = static_cast<int>(config.dims.iPV);
                cfg_.K = static_cast<int>(config.dims.iPAng);
                cfg_.desc = config.desc;
                cfg_.policy = detail::normalizeFilterPolicy(config.policy);
                cfg_.stream = config.stream;
                if (!validateCfg_()) return false;

                Nu_ = cfg_.Nu;
                Nv_ = cfg_.Nv;
                K_ = cfg_.K;
                paddedN_ = detail::fp_computePaddedN(Nu_);
                n_cmplx_ = paddedN_ / 2 + 1;
                stream_ = cfg_.stream;
                policy_ = cfg_.policy;
                desc_ = cfg_.desc;

                YK_CUDA_CHECK(cudaMalloc(&d_startu_, static_cast<size_t>(K_) * sizeof(int)));
                YK_CUDA_CHECK(cudaMalloc(&d_padded_,
                    static_cast<size_t>(K_) * Nv_ * paddedN_ * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_weights_, static_cast<size_t>(n_cmplx_) * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_,
                    static_cast<size_t>(K_) * Nv_ * n_cmplx_ * sizeof(cufftComplex)));
                fft_batch_.init(paddedN_, K_ * Nv_, stream_);
                kernel_fft_.prepare(paddedN_, stream_);
                is_initialized_ = true;
                weights_ready_ = false;
                weights_dirty_ = true;
                current_K_ = 0;
                return true;
            }

            // 强类型批次入口。h_gv 只作为本批几何派生数据的只读视图，
            // 不保存在 processor 中，避免上一批 context 遗留到下一批。
            bool apply(const float* d_input, float* d_output,
                const FdkFilterChunk& context, cudaStream_t stream)
            {
                if (!is_initialized_ || !d_input || !d_output || !context.h_gv ||
                    context.K <= 0 || context.K > K_) {
                    YK_LOGE("[YK][Filter][E] apply: invalid prepared state or chunk.");
                    return false;
                }
                if (stream != 0 && stream != stream_) setStream_(stream);

                current_K_ = context.K;
                host_startu_.resize(current_K_);
                for (int i = 0; i < current_K_; ++i) {
                    host_startu_[i] = detail::fp_computeStartU(
                        Nu_, paddedN_, context.h_gv[i].offsetU_pix);
                }
                YK_CUDA_CHECK(cudaMemcpyAsync(d_startu_, host_startu_.data(),
                    static_cast<size_t>(current_K_) * sizeof(int),
                    cudaMemcpyHostToDevice, stream_));
                du_real_ = context.h_gv[0].du_mm > 0.f ? context.h_gv[0].du_mm : 1.f;

                ensureWeights_();
                if (!weights_ready_) return false;
                detail::fp_launchPad(d_input, d_padded_, d_startu_, Nu_, Nv_, paddedN_,
                    current_K_, policy_, stream_);
                detail::fp_launchFilter(d_padded_, d_complex_buf_, d_weights_, n_cmplx_,
                    paddedN_, current_K_, Nv_, fft_batch_, stream_);
                detail::fp_launchCrop(d_padded_, d_output, d_startu_, Nu_, Nv_, paddedN_,
                    current_K_, policy_, stream_);
                return true;
            }

            void release()
            {
                fft_batch_.release();
                kernel_fft_.release();

                if (d_startu_) { cudaFree(d_startu_);      d_startu_ = nullptr; }
                if (d_padded_) { cudaFree(d_padded_);      d_padded_ = nullptr; }
                if (d_weights_) { cudaFree(d_weights_);     d_weights_ = nullptr; }
                if (d_complex_buf_) { cudaFree(d_complex_buf_); d_complex_buf_ = nullptr; }

                host_startu_.clear();

                Nu_ = Nv_ = K_ = paddedN_ = n_cmplx_ = 0;
                du_real_ = 1.0f;
                current_K_ = 0;
                stream_ = 0;
                is_initialized_ = false;
                weights_ready_ = false;
                weights_dirty_ = true;
                desc_ = {};
                policy_ = {};
            }

            // ----------------------------------------------------------------
            bool isPrepared() const { return is_initialized_; }

            // 调试访问器
            int   Nu()           const { return Nu_; }
            int   Nv()           const { return Nv_; }
            int   Kchunk()       const { return K_; }
            int   currentK()     const { return current_K_; }
            int   paddedN()      const { return paddedN_; }
            int   nComplex()     const { return n_cmplx_; }
            float postScale()    const { return du_real_; }
            bool  weightsDirty() const { return weights_dirty_; }

        private:
            // ---- 配置暂存 ----
            struct Cfg {
                int                 Nu = 0;
                int                 Nv = 0;
                int                 K = 0;
                SFilterKernelDesc   desc = {};
                SKernelLaunchPolicy policy = {};
                cudaStream_t        stream = 0;
            } cfg_;

            // ---- 运行期状态 ----
            int          Nu_ = 0;
            int          Nv_ = 0;
            int          K_ = 0;
            int          current_K_ = 0;
            int          paddedN_ = 0;
            int          n_cmplx_ = 0;
            float        du_real_ = 1.0f;
            cudaStream_t stream_ = 0;

            SKernelLaunchPolicy policy_ = {};
            SFilterKernelDesc   desc_ = {};

            // GPU buffers
            int* d_startu_ = nullptr;  // [K_]
            float* d_padded_ = nullptr;  // [K_*Nv_*paddedN_]
            float* d_weights_ = nullptr;  // [n_cmplx_]
            cufftComplex* d_complex_buf_ = nullptr;  // [K_*Nv_*n_cmplx_]

            CudaFFT         fft_batch_;
            YK::Filter::CreateFilterKernelFromFFT kernel_fft_;
            std::vector<int> host_startu_;

            bool is_initialized_ = false;
            bool weights_ready_ = false;
            bool weights_dirty_ = true;

            // ---- 内部辅助 ----

            bool validateCfg_() const
            {
                if (cfg_.Nu <= 0 || cfg_.Nv <= 0 || cfg_.K <= 0) {
                    YK_LOGE("validateCfg: invalid dims Nu={} Nv={} K={}.",
                        cfg_.Nu, cfg_.Nv, cfg_.K);
                    return false;
                }
                return true;
            }

            void ensureWeights_()
            {
                if (weights_ready_ && !weights_dirty_) return;

                kernel_fft_.prepare(paddedN_, stream_);
                kernel_fft_.build_weights(d_weights_, desc_, du_real_, /*bake_invN=*/true);

                YK_LOGI("weights built (lazy): source={} kind={} cutoff={:.3f} "
                    "gain={:.3f} dc0={} paddedN={}",
                    static_cast<int>(desc_.source), static_cast<int>(desc_.kind),
                    desc_.cutoff, desc_.gain,
                    static_cast<int>(desc_.source == EWeightsBuildSource::DiscreteRLFFT),
                    paddedN_);

                weights_ready_ = true;
                weights_dirty_ = false;
            }

            void setStream_(cudaStream_t s)
            {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                stream_ = s;
                fft_batch_.setStream(s);
                kernel_fft_.setStream(s);
                YK_LOGI("setStream_: switched to {}", static_cast<void*>(s));
            }
        };

    }; // namespace YK::Fdk
};
