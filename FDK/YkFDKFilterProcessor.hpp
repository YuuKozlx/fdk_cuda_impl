#pragma once
#include <vector>

#include <cufft.h>
#include <cuda_runtime.h>

#include "../global/IProcessor.hpp"
#include "../Filter/YkConv.hpp"
#include "../Filter/YkFFT.hpp"
#include "../FDK/YkFdkPipelineContext.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

#include "cuh/YkFDKFilterLaunch.cuh"   // fp_launchPad / fp_launchFilter / fp_launchCrop
#include "cuh/YkFDKFilterHelpers.cuh"  // normalizeFilterPolicy / fp_computePaddedN / fp_computeStartU
#include "../Filter/YkFilterKernelFFT.cuh"

namespace YK {
    namespace Fdk {

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

                cfg_.Nu = static_cast<int>(ic->dims.iPU);
                cfg_.Nv = static_cast<int>(ic->dims.iPV);
                cfg_.K = static_cast<int>(ic->dims.iPAng);
                cfg_.desc = ic->desc;
                cfg_.policy = detail::normalizeFilterPolicy(ic->policy);
                cfg_.stream = ic->stream;
                cfg_ready_ = true;

                YK_LOGI("setInitContext: Nu=%d Nv=%d Kchunk=%d stream=%p",
                    cfg_.Nu, cfg_.Nv, cfg_.K, static_cast<void*>(cfg_.stream));
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
                paddedN_ = detail::fp_computePaddedN(Nu_);
                n_cmplx_ = paddedN_ / 2 + 1;
                stream_ = cfg_.stream;
                policy_ = cfg_.policy;
                desc_ = cfg_.desc;

                YK_CUDA_CHECK(cudaMalloc(&d_startu_,
                    static_cast<size_t>(K_) * sizeof(int)));
                YK_CUDA_CHECK(cudaMalloc(&d_padded_,
                    static_cast<size_t>(K_) * Nv_ * paddedN_ * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_weights_,
                    static_cast<size_t>(n_cmplx_) * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_complex_buf_,
                    static_cast<size_t>(K_) * Nv_ * n_cmplx_ * sizeof(cufftComplex)));

                fft_batch_.init(paddedN_, K_ * Nv_, stream_);
                kernel_fft_.prepare(paddedN_, stream_);

                is_initialized_ = true;
                weights_ready_ = false;
                weights_dirty_ = true;
                current_K_ = 0;

                YK_LOGI("init ok: Nu=%d Nv=%d Kchunk=%d paddedN=%d n_cmplx=%d stream=%p",
                    Nu_, Nv_, K_, paddedN_, n_cmplx_, static_cast<void*>(stream_));
                return true;
            }

            // ----------------------------------------------------------------
            // IProcessor::setContext — 每 chunk 前调用
            // ----------------------------------------------------------------
            void setContext(const void* ctx) override
            {
                if (!is_initialized_) { YK_LOGE("setContext: not initialized."); return; }
                if (!ctx) { YK_LOGE("setContext: null.");            return; }

                const auto* fc = static_cast<const FdkFilterContext*>(ctx);

                if (!fc->h_gv) {
                    YK_LOGE("setContext: h_gv is null."); return;
                }
                if (fc->K <= 0 || fc->K > K_) {
                    YK_LOGE("setContext: K(%d) out of range [1, %d].", fc->K, K_); return;
                }

                current_K_ = fc->K;

                host_startu_.resize(current_K_);
                for (int i = 0; i < current_K_; ++i)
                    host_startu_[i] = detail::fp_computeStartU(
                        Nu_, paddedN_, fc->h_gv[i].offsetU_pix);

                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_startu_, host_startu_.data(),
                    static_cast<size_t>(current_K_) * sizeof(int),
                    cudaMemcpyHostToDevice, stream_));

                postScale_ = (fc->h_gv[0].du_mm > 0.0f)
                    ? 1.0f / fc->h_gv[0].du_mm
                    : 1.0f;
            }

            // ----------------------------------------------------------------
            // IProcessor::process
            // ----------------------------------------------------------------
            void process(const void* d_input, void* d_output,
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

                detail::fp_launchPad(
                    d_in, d_padded_, d_startu_,
                    Nu_, Nv_, paddedN_, current_K_,
                    policy_, stream_);

                detail::fp_launchFilter(
                    d_padded_, d_complex_buf_, d_weights_,
                    n_cmplx_, paddedN_, current_K_, Nv_,
                    postScale_, fft_batch_, stream_);

                detail::fp_launchCrop(
                    d_padded_, d_out, d_startu_,
                    Nu_, Nv_, paddedN_, current_K_,
                    policy_, stream_);
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
                policy_ = {};
            }

            // ----------------------------------------------------------------
            // IProcessor 状态查询
            // ----------------------------------------------------------------
            bool        isInitialized() const override { return is_initialized_; }
            const char* name()          const override { return "FilterProcessor"; }

            // 调试访问器
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
                SFilterKernelDesc   desc = {};
                SKernelLaunchPolicy policy = {};
                cudaStream_t        stream = 0;
            } cfg_;
            bool cfg_ready_ = false;

            // ---- 运行期状态 ----
            int          Nu_ = 0;
            int          Nv_ = 0;
            int          K_ = 0;
            int          current_K_ = 0;
            int          paddedN_ = 0;
            int          n_cmplx_ = 0;
            float        postScale_ = 1.0f;
            cudaStream_t stream_ = 0;

            SKernelLaunchPolicy policy_ = {};
            SFilterKernelDesc   desc_ = {};

            // GPU buffers
            int* d_startu_ = nullptr;  // [K_]
            float* d_padded_ = nullptr;  // [K_*Nv_*paddedN_]
            float* d_weights_ = nullptr;  // [n_cmplx_]
            cufftComplex* d_complex_buf_ = nullptr;  // [K_*Nv_*n_cmplx_]

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

                YK_LOGI("weights built (lazy): source=%d kind=%d cutoff=%.3f "
                    "gain=%.3f dc0=%d paddedN=%d",
                    static_cast<int>(desc_.source), static_cast<int>(desc_.kind),
                    desc_.cutoff, desc_.gain,
                    static_cast<int>(desc_.force_dc_zero), paddedN_);

                weights_ready_ = true;
                weights_dirty_ = false;
            }

            void setStream_(cudaStream_t s)
            {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                stream_ = s;
                fft_batch_.setStream(s);
                kernel_fft_.setStream(s);
                YK_LOGI("setStream_: switched to %p", static_cast<void*>(s));
            }
        };

    }; // namespace YK::Fdk
};
