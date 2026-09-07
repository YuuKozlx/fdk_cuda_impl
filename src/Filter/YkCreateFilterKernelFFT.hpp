#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <utility>


#include <limits>
#include "../Filter/YkFFT.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkFilterTypes.hpp"
#include "../global/YkMem3d.hpp"
#include "YkCreateFilterKernelLaunch.cuh"

namespace YK {
    namespace Filter {

        class CreateFilterKernelFromFFT {
            // 生成的核默认单位都为du=1，需要反应信息，需要在外部操作（1/(du*du)）
        public:

            CreateFilterKernelFromFFT() = default;
            ~CreateFilterKernelFromFFT() { release(); }

            CreateFilterKernelFromFFT(const CreateFilterKernelFromFFT&) = delete;
            CreateFilterKernelFromFFT& operator=(const CreateFilterKernelFromFFT&) = delete;

            CreateFilterKernelFromFFT(CreateFilterKernelFromFFT&& o) noexcept { move_from(o); }
            CreateFilterKernelFromFFT& operator=(CreateFilterKernelFromFFT&& o) noexcept {
                if (this != &o) { release(); move_from(o); }
                return *this;
            }

            // --------------------------------------------------------
            // Lifecycle
            // --------------------------------------------------------

            void prepare(int paddedN, cudaStream_t stream = 0)
            {
                if (paddedN <= 0) {
                    YK_ASSERT(false && "paddedN must be > 0");
                    return;
                }
                if (ready_ && paddedN_ == paddedN) {
                    setStream(stream);
                    return;
                }

                release();

                paddedN_ = paddedN;
                n_complex_ = paddedN_ / 2 + 1;
                stream_ = stream;

                d_tmp_fft_.alloc(n_complex_, 0);

                bool ok = fft_r2c_.init(paddedN_, /*batch=*/1, YK::CudaFFT::EPlanMode::R2COnly, stream_);
                YK_ASSERT(ok && "CudaFFT R2COnly init failed");
                ready_ = ok;
            }

            void setStream(cudaStream_t s)
            {
                stream_ = s;
                if (ready_) fft_r2c_.setStream(stream_);
            }

            void release()
            {
                d_tmp_fft_.reset();
                fft_r2c_.release();

                paddedN_ = 0;
                n_complex_ = 0;
                stream_ = 0;
                ready_ = false;
            }

            // --------------------------------------------------------
            // Accessors
            // --------------------------------------------------------

            int          paddedN()   const { return paddedN_; }
            int          n_complex() const { return n_complex_; }
            cudaStream_t stream()    const { return stream_; }

            /// Allocate a device buffer sized for the weight array.
            /// Caller takes ownership; free with cudaFree().
            float* alloc_weights() const
            {
                YK_ASSERT(ready_);
                float* d_w = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_w, (size_t)n_complex_ * sizeof(float)));
                return d_w;
            }

            // --------------------------------------------------------
            // Build
            //
            // Fills d_weights_fft[n_complex] according to desc.
            //
            // bake_invN = true  : bakes 1/N into weights to cancel the
            //                     cuFFT C2R gain of N downstream.
            //
            // du_real : physical detector pixel spacing [mm].
            //           The spatial RL kernel is generated with du=1 (dimensionless).
            //           A factor of 1/du_real is applied here to convert the
            //           discrete convolution result to physical units [mm⁻¹].
            //           See: kernel_gen_spatial_rl_kernel_du1 for derivation.
            //
            // Two build paths (selected by desc.source):
            //   AnalyticFreq   – direct frequency-domain construction
            //   DiscreteRLFFT  – spatial RL kernel -> FFT -> window -> physicalize
            // --------------------------------------------------------
            void build_weights(
                float* d_weights_fft,
                const SFilterKernelDesc& desc,
                float du_real,
                bool bake_invN = true) const
            {
                YK_ASSERT(ready_);
                YK_ASSERT(d_weights_fft);
                YK_ASSERT(du_real > 0.f);

                if (desc.source == EWeightsBuildSource::SpatialRampFFT) {
                    const size_t m = desc.spatial_ramp.size();
                    YK_ASSERT(m > 0 && (m & 1u) == 1u);
                    YK_ASSERT(m <= static_cast<size_t>(paddedN_));
                    if (m == 0 || (m & 1u) == 0 || m > static_cast<size_t>(paddedN_))
                        return;
                    for (size_t i = 0; i < m / 2; ++i) {
                        YK_ASSERT(std::isfinite(desc.spatial_ramp[i]));
                        YK_ASSERT(std::isfinite(desc.spatial_ramp[m - 1 - i]));
                        YK_ASSERT(std::fabs(desc.spatial_ramp[i] - desc.spatial_ramp[m - 1 - i]) < 1e-4f);
                    }
                    YK_ASSERT(std::isfinite(desc.spatial_ramp[m / 2]));

                    Mem::DeviceLinearBuffer<float> d_spatial;
                    Mem::DeviceLinearBuffer<float> d_ramp;
                    d_spatial.alloc(paddedN_, 0);
                    d_ramp.alloc(static_cast<int>(m), 0);
                    YK_CUDA_CHECK(cudaMemcpyAsync(d_ramp.data(), desc.spatial_ramp.data(),
                        m * sizeof(float), cudaMemcpyHostToDevice, stream_));
                    const bool ok = flt_launch_kernel_build_spatial_ramp(
                        d_spatial.data(), paddedN_, d_ramp.data(), static_cast<int>(m), bake_invN, stream_);
                    YK_ASSERT(ok);
                    fft_r2c_.fft(d_spatial.data(), d_tmp_fft_.data());
                    flt_launch_kernel_extract_weights_from_fft(
                        d_tmp_fft_.data(), d_weights_fft, n_complex_,
                        desc.extract_mode, stream_);
                    // 空域柱面 ramp 也必须经过与普通 ramp 一致的窗函数；
                    // 旧实现直接返回，导致 Hamming/Hann 等配置静默失效。
                    if (desc.kind != EFilterKernel::RamLak)
                        flt_launch_kernel_apply_window_to_weights_inplace(
                            d_weights_fft, n_complex_, paddedN_, desc, stream_);
                    YK_CUDA_CHECK(cudaMemsetAsync(d_weights_fft, 0,
                        sizeof(float), stream_));
                    flt_launch_kernel_scale_inplace(
                        d_weights_fft, n_complex_, desc.gain / du_real, stream_);
                    YK_CUDA_KERNEL_CHECK();
                    return;
                }

                bool bBuildSuccess = build_custom_weights(desc, d_weights_fft, bake_invN, du_real);

                if (bBuildSuccess) {
                    return;
                }

                // ---------------------------------------------

                using namespace detail;

                dim3 block(256, 1);

                if (desc.kind == EFilterKernel::None) {
                    flt_launch_kernel_fill_identity_weights(
                        d_weights_fft, n_complex_, paddedN_, desc.gain, bake_invN, stream_);
                    return;
                }

                if (desc.source == EWeightsBuildSource::AnalyticFreq) {
                    flt_launch_kernel_build_weights_analytic_freq(
                        d_weights_fft, n_complex_, paddedN_, desc, bake_invN, stream_);
                    YK_CUDA_KERNEL_CHECK();
                    // 补 du 缩放，和 DiscreteRLFFT 路径保持一致
                    flt_launch_kernel_scale_inplace(
                        d_weights_fft, n_complex_, 1.0f / du_real, stream_);
                    YK_CUDA_KERNEL_CHECK();
                    return;
                }

                // ---- DiscreteRLFFT path ----

                // Step 1: 生成 du=1 的纯数字离散空域核
                Mem::DeviceLinearBuffer<float> d_spatial;
                d_spatial.alloc(paddedN_, 0);
                flt_launch_kernel_gen_spatial_rl_kernel_du1(
                    d_spatial.data(), paddedN_, bake_invN, stream_);
                YK_CUDA_KERNEL_CHECK();


                // Step 2: FFT 变换到频域 du = 1
                fft_r2c_.fft(d_spatial.data(), d_tmp_fft_.data());

                // Step 3: 提取有限离散 RL 频谱（取实部或模）du = 1。
                // 截断后的空域核求和通常不严格为零，因此 FFT 会残留一个
                // O(1/N) 的 DC 偏置。Ram-Lak 必须抑制常量投影分量，这里对
                // DiscreteRLFFT 路径强制令 k=0 为零，不依赖调用方选项。
                flt_launch_kernel_extract_weights_from_fft(
                    d_tmp_fft_.data(), d_weights_fft, n_complex_,
                    desc.extract_mode, stream_);
                YK_CUDA_CHECK(cudaMemsetAsync(
                    d_weights_fft, 0, sizeof(float), stream_));
                YK_CUDA_KERNEL_CHECK();

                // Step 4: 加窗（Hamming / Hann 等） du =1
                flt_launch_kernel_apply_window_to_weights_inplace(
                    d_weights_fft, n_complex_, paddedN_, desc, stream_);
                YK_CUDA_KERNEL_CHECK();


                // Step 5: 物理化 — 将 du=1 的离散滤波核转换为物理单位 [mm⁻¹]
                //
                // 连续 Ram-Lak 核在采样点 t = n·du 处的值：
                //
                //   h(n·du) =  1 / (4·du²)          n = 0
                //   h(n·du) = -1 / (π²·du²·n²)      n 奇
                //   h(n·du) = 0  n 偶
                //
                //
                // 要从 h̃[n] 还原为物理离散核 h[n]，需要两步：
                //   (1) 乘以 1/du² — 补回连续核本身的物理量纲
                //   (2) 乘以 du    — 离散化黎曼步长（采样间隔），将连续积分转为离散求和
                //
                // 两步合并：1/du² × du = 1/du，即净补偿因子为 1/du_real：
                //
                //   h[n] = h̃[n] / du =  1/(4·du)       n = 0
                //   h[n] = h̃[n] / du = -1/(π²·du·n²)   n 奇
                //
                // ---------------------------------------------------------------
                // 附注：FFT 卷积与连续积分的关系
                //
                //   FFT 卷积实现的是离散求和：
                //     q[n] = Σ_m w[m] × h̃[n-m]
                //
                //   而物理上需要的是连续积分的离散近似：
                //     q(u) = ∫ w(u')·h(u-u') du' ≈ Σ_m w[m]·h[n-m] × du
                //
                //   其中 du 是采样间隔（= 1/采样率 fs）。
                //   FFT 卷积本身不含此步长，必须显式补偿。
                //
                // 综合以上两点，频域权重乘以 1/du_real（净补偿），
                // 使最终滤波结果的单位为 mm⁻¹。
                flt_launch_kernel_scale_inplace(
                    d_weights_fft, n_complex_, 1.0f / du_real, stream_);
                YK_CUDA_KERNEL_CHECK();

            }

            bool build_custom_weights(const YK::SFilterKernelDesc& desc, float* d_weights_fft, bool bake_invN, float du_real) const
            {
                // ── Custom 路径 ───────────────────────────────────────
                if (desc.kind == EFilterKernel::Custom) {
                    YK_ASSERT(!desc.custom_weights.empty());
                    YK_ASSERT((int)desc.custom_weights.size() == n_complex_);

                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_weights_fft,
                        desc.custom_weights.data(),
                        (size_t)n_complex_ * sizeof(float),
                        cudaMemcpyHostToDevice, stream_));

                    // gain 缩放
                    if (desc.gain != 1.0f) {
                        flt_launch_kernel_scale_inplace(
                            d_weights_fft, n_complex_, desc.gain, stream_);

                        YK_CUDA_KERNEL_CHECK();
                    }

                    YK_CUDA_KERNEL_CHECK();
                    // bake_invN
                    if (bake_invN && paddedN_ > 0) {
                        flt_launch_kernel_scale_inplace(
                            d_weights_fft, n_complex_, 1.f / (float)paddedN_, stream_);

                        YK_CUDA_KERNEL_CHECK();
                    }


                    // Custom 路径补 du 缩放，和 DiscreteRLFFT 路径保持一致
                    if (fabs(du_real) > std::numeric_limits<float>::epsilon()) {
                        flt_launch_kernel_scale_inplace(
                            d_weights_fft, n_complex_, 1.0f / du_real, stream_);

                        YK_CUDA_KERNEL_CHECK();
                    }

                    // custom_weights 属于调用方的普通 host vector。这里是滤波核
                    // 的一次性准备阶段，没有完成事件对外返回，因此必须等上传及
                    // 后续缩放结束后再返回，调用方随后修改或销毁 desc 才是安全的。
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
                    return true;
                }
                return false;
            }


        private:
            void move_from(CreateFilterKernelFromFFT& o) noexcept
            {
                paddedN_ = o.paddedN_;    o.paddedN_ = 0;
                n_complex_ = o.n_complex_;  o.n_complex_ = 0;
                stream_ = o.stream_;     o.stream_ = 0;
                ready_ = o.ready_;      o.ready_ = false;
                d_tmp_fft_ = std::move(o.d_tmp_fft_);
                fft_r2c_ = std::move(o.fft_r2c_);
            }

        private:
            int           paddedN_ = 0;
            int           n_complex_ = 0;
            cudaStream_t  stream_ = 0;
            bool          ready_ = false;

            Mem::DeviceLinearBuffer<cufftComplex> d_tmp_fft_;
            YK::CudaFFT   fft_r2c_;   // R2C only
        };

    } // namespace Filter
} // namespace YK
