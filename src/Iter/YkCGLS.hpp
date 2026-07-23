#pragma once
#include "FP/YkFPRunner.hpp"
#include "FP/YkFpRunnerExVec.hpp"
#include "kernel/YkIterLaunch.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkLog.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"

#include <cuda_runtime.h>
#include "BP/YkSiddonBPRunner.hpp"
#include <global/YkCBCTParams.h>
#include <vector>
#include <numeric>

namespace YK {
    class CGLS {
    public:
        struct Config {
            int   n_iter = 50;
            float eps = 1e-8f;
            bool  restart = true;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_Joseph_v2;
        };

        bool init(const SCBCTParams& params, const Config& cfg,
            cudaStream_t stream, int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            // r = b - Ax（残差，正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_r_, sino_n * sizeof(float)));
            // p = A^T r（搜索方向，体积空间）
            YK_CUDA_CHECK(cudaMalloc(&d_p_, vol_n * sizeof(float)));
            // q = A p（正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_q_, sino_n * sizeof(float)));
            // s = A^T r（体积空间，临时）
            YK_CUDA_CHECK(cudaMalloc(&d_s_, vol_n * sizeof(float)));
            // 上一次 x 的备份（用于发散时回退）
            YK_CUDA_CHECK(cudaMalloc(&d_x_prev_, vol_n * sizeof(float)));
            // 正投影临时缓冲
            YK_CUDA_CHECK(cudaMalloc(&d_ax_, sino_n * sizeof(float)));

            fp_.init(params, cfg.fp_task, deviceId);
            bp_.init(params, cfg.bp_task, deviceId);

            is_initialized_ = true;
            YK_LOGI("[CGLS] init OK: {} angles", params.iPAng);
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            int re_init_at = -1;
            float l2_prev = std::numeric_limits<float>::max();

            // ── 初始化 ───────────────────────────────────────────────
            // r = b - A x
            initialize_(d_sino_meas, d_vol, params, stream, vol_n, sino_n);
            float gamma = compute_gamma_(vol_n, stream);

            for (int iter = 0; iter < cfg_.n_iter; ++iter)
            {
                // ── 备份当前 x ───────────────────────────────────────
                YK_CUDA_CHECK(cudaMemcpyAsync(d_x_prev_, d_vol,
                    vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                // q = A p
                YK_CUDA_CHECK(cudaMemsetAsync(d_q_, 0,
                    sino_n * sizeof(float), stream));
                fp_.run(d_p_, params, d_q_, stream);

                // alpha = gamma / ||q||^2
                const float q_norm2 = dot_device_(d_q_, d_q_, sino_n, stream);
                if (q_norm2 < cfg_.eps) {
                    YK_LOGI("[CGLS] q_norm^2 too small, stop at iter {}", iter);
                    break;
                }
                const float alpha = gamma / q_norm2;

                // x += alpha * p
                YK::Iter::axpy_launch(d_vol, d_p_, alpha, vol_n, stream);

                // ── 收敛检查（l2 norm of b - Ax）─────────────────────
                YK_CUDA_CHECK(cudaMemsetAsync(d_ax_, 0,
                    sino_n * sizeof(float), stream));
                fp_.run(d_vol, params, d_ax_, stream);
                YK::Iter::residual_launch(d_sino_meas, d_ax_,
                    d_ax_, sino_n, stream);
                const float l2_cur = std::sqrt(
                    dot_device_(d_ax_, d_ax_, sino_n, stream));

                YK_LOGI("[CGLS] iter {}/{} l2={:.6e}", iter + 1, cfg_.n_iter, l2_cur);

                // ── 发散检测 ─────────────────────────────────────────
                if (iter > 0 && l2_cur > l2_prev) {
                    // 回退 x
                    YK_CUDA_CHECK(cudaMemcpyAsync(d_vol, d_x_prev_,
                        vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                    YK_LOGW("[CGLS] divergence at iter {}, rollback", iter);

                    if (re_init_at + 1 == iter || !cfg_.restart) {
                        YK_LOGW("[CGLS] exited due to divergence");
                        break;
                    }
                    re_init_at = iter;

                    // 重新初始化
                    initialize_(d_sino_meas, d_vol, params, stream, vol_n, sino_n);
                    gamma = compute_gamma_(vol_n, stream);
                    l2_prev = std::numeric_limits<float>::max();
                    continue;
                }
                l2_prev = l2_cur;

                // r -= alpha * q
                YK::Iter::axpy_launch(d_r_, d_q_, -alpha, sino_n, stream);

                // s = A^T r
                bp_.run(d_r_, params, d_s_, stream, /*clear_vol=*/true);

                // gamma1 = ||s||^2
                const float gamma1 = dot_device_(d_s_, d_s_, vol_n, stream);

                // beta = gamma1 / gamma
                const float beta = gamma1 / (gamma + cfg_.eps);
                gamma = gamma1;

                // 改后（正确）
                YK::Iter::scale_launch(d_p_, beta, vol_n, stream);   // p *= beta
                YK::Iter::axpy_launch(d_p_, d_s_, 1.0f, vol_n, stream);  // p += s

                // ── 约束 ─────────────────────────────────────────────
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(
                        d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(
                        d_vol, vol_n, cfg_.max_constraint, stream);
            }
            return true;
        }

        void release()
        {
            if (d_r_) { cudaFree(d_r_);      d_r_ = nullptr; }
            if (d_p_) { cudaFree(d_p_);      d_p_ = nullptr; }
            if (d_q_) { cudaFree(d_q_);      d_q_ = nullptr; }
            if (d_s_) { cudaFree(d_s_);      d_s_ = nullptr; }
            if (d_x_prev_) { cudaFree(d_x_prev_); d_x_prev_ = nullptr; }
            if (d_ax_) { cudaFree(d_ax_);     d_ax_ = nullptr; }
            fp_.release();
            bp_.release();
            is_initialized_ = false;
        }

        ~CGLS() { release(); }

    private:
        void initialize_(
            const float* d_sino_meas,
            const float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream,
            size_t vol_n, size_t sino_n)
        {
            // r = b - A x
            YK_CUDA_CHECK(cudaMemsetAsync(d_r_, 0,
                sino_n * sizeof(float), stream));
            fp_.run(d_vol, params, d_r_, stream);
            YK::Iter::residual_launch(d_sino_meas, d_r_,
                d_r_, sino_n, stream);

            // p = A^T r
            bp_.run(d_r_, params, d_p_, stream, /*clear_vol=*/true);
        }

        float compute_gamma_(size_t vol_n, cudaStream_t stream)
        {
            // gamma = ||p||^2
            return dot_device_(d_p_, d_p_, vol_n, stream);
        }

        float dot_device_(const float* a, const float* b,
            size_t n, cudaStream_t stream)
        {
            // 用 cublas 或自己的 reduce kernel
            float result = 0.f;
            YK::Iter::dot_launch(a, b, n, &result, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            return result;
        }

        bool         is_initialized_ = false;
        SCBCTParams  params_;
        Config       cfg_;

        ConeProjector     fp_;
        ConeBackprojector bp_;

        float* d_r_ = nullptr;   // 残差（正弦图空间）
        float* d_p_ = nullptr;   // 搜索方向（体积空间）
        float* d_q_ = nullptr;   // A*p（正弦图空间）
        float* d_s_ = nullptr;   // A^T*r（体积空间）
        float* d_x_prev_ = nullptr;   // x 备份
        float* d_ax_ = nullptr;   // 正投影临时缓冲
    };

    YK_INLINE bool cgls_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t stream,
        CGLS::Config cfg = {})
    {
        CGLS recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }


    class CGLSAstra {
    public:
        struct Config {
            int   n_iter = 50;
            float eps = 1e-8f;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_FDK_matched;
        };

        bool init(const SCBCTParams& params, const Config& cfg,
            cudaStream_t stream, int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            YK_CUDA_CHECK(cudaMalloc(&d_r_, sino_n * sizeof(float)));  // r（正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_w_, sino_n * sizeof(float)));  // w = A p（正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_p_, vol_n * sizeof(float)));  // p（体积空间）
            YK_CUDA_CHECK(cudaMalloc(&d_z_, vol_n * sizeof(float)));  // z = A^T r（体积空间）

            fp_.init(params, cfg.fp_task, deviceId);
            bp_.init(params, cfg.bp_task, deviceId);

            is_initialized_ = true;
            YK_LOGI("[CGLSAstra] init OK: {} angles", params.iPAng);
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            // ── 初始化：r = b - A x，p = A^T r ───────────────────────
            // r = b
            YK_CUDA_CHECK(cudaMemcpyAsync(d_r_, d_sino_meas,
                sino_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

            // r = b - A x（callFP with scale -1，等价于 r -= A x）
            fp_.run(d_vol, params, d_r_, stream);  // d_r_ += A*x（accumulate=true）
            // 但我们需要 r = b - Ax，所以先 r=b，再 r -= Ax
            // 用 residual_launch 实现
            {
                float* d_ax = d_w_;  // 借用 d_w_ 作临时缓冲
                YK_CUDA_CHECK(cudaMemsetAsync(d_ax, 0,
                    sino_n * sizeof(float), stream));
                fp_.run(d_vol, params, d_ax, stream);
                YK::Iter::residual_launch(d_sino_meas, d_ax,
                    d_r_, sino_n, stream);
            }

            // p = A^T r
            bp_.run(d_r_, params, d_p_, stream, /*clear_vol=*/true);

            // gamma = <p, p>
            float gamma = 0.f;
            YK::Iter::dot_launch(d_p_, d_p_, vol_n, &gamma, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            // ── 迭代 ─────────────────────────────────────────────────
            for (int iter = 0; iter < cfg_.n_iter; ++iter)
            {
                // w = A p
                YK_CUDA_CHECK(cudaMemsetAsync(d_w_, 0,
                    sino_n * sizeof(float), stream));
                fp_.run(d_p_, params, d_w_, stream);

                // alpha = gamma / <w, w>
                float ww = 0.f;
                YK::Iter::dot_launch(d_w_, d_w_, sino_n, &ww, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                if (ww < cfg_.eps) {
                    YK_LOGI("[CGLSAstra] ww too small, stop at iter {}", iter);
                    break;
                }
                const float alpha = gamma / ww;

                // x += alpha * p
                YK::Iter::axpy_launch(d_vol, d_p_, alpha, vol_n, stream);

                // r -= alpha * w
                YK::Iter::axpy_launch(d_r_, d_w_, -alpha, sino_n, stream);

                // z = A^T r
                bp_.run(d_r_, params, d_z_, stream, /*clear_vol=*/true);

                // gamma1 = <z, z>
                float gamma1 = 0.f;
                YK::Iter::dot_launch(d_z_, d_z_, vol_n, &gamma1, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                // beta = gamma1 / gamma
                const float beta = gamma1 / (gamma + cfg_.eps);
                gamma = gamma1;

                // p = z + beta * p
                YK::Iter::scale_launch(d_p_, beta, vol_n, stream);   // p *= beta
                YK::Iter::axpy_launch(d_p_, d_z_, 1.0f, vol_n, stream); // p += z

                // 约束
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(
                        d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(
                        d_vol, vol_n, cfg_.max_constraint, stream);

                YK_LOGI("[CGLSAstra] iter {}/{} gamma={:.6e}",
                    iter + 1, cfg_.n_iter, gamma1);
            }
            return true;
        }

        void release()
        {
            if (d_r_) { cudaFree(d_r_); d_r_ = nullptr; }
            if (d_w_) { cudaFree(d_w_); d_w_ = nullptr; }
            if (d_p_) { cudaFree(d_p_); d_p_ = nullptr; }
            if (d_z_) { cudaFree(d_z_); d_z_ = nullptr; }
            fp_.release();
            bp_.release();
            is_initialized_ = false;
        }

        ~CGLSAstra() { release(); }

    private:
        bool        is_initialized_ = false;
        SCBCTParams params_;
        Config      cfg_;

        ConeProjector     fp_;
        ConeBackprojector bp_;

        float* d_r_ = nullptr;   // 残差（正弦图空间）
        float* d_w_ = nullptr;   // w = A p（正弦图空间）
        float* d_p_ = nullptr;   // 搜索方向（体积空间）
        float* d_z_ = nullptr;   // z = A^T r（体积空间）
    };

    YK_INLINE bool cgls_astra_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t stream,
        CGLSAstra::Config cfg = {})
    {
        CGLSAstra recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }
}


namespace YK {
    // ============================================================
    //  CGLSEx
    //
    //  与 CGLS 的差异：正反投影算子换成 ConeProjectorEx /
    //  ConeBackprojectorEx，几何以外部传入的 h_views
    //  （std::vector<SConeProjGeomVec>）驱动，不依赖
    //  SCBCTParams.angle_list 内部反推几何 —— 跟 OSSARTEx 相对
    //  OSSART 的关系一致，因此天然支持任意轨迹（螺旋等）。
    //
    //  迭代算法本身（CG 数学结构、收敛判据、回退重启逻辑）与
    //  CGLS 完全一致，不做任何改动。
    // ============================================================
    class CGLSEx {
    public:
        struct Config {
            int   n_iter = 50;
            float eps = 1e-8f;
            bool  restart = true;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_Joseph_v2;
        };

        bool init(const SCBCTParams& params,
            const Config& cfg,
            const std::vector<SConeProjGeomVec>& h_views,
            cudaStream_t stream,
            int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;
            h_views_ = h_views;
            deviceId_ = deviceId;

            if ((int)h_views.size() != params.iPAng) {
                YK_LOGE("[CGLSEx] h_views size mismatch: {} vs iPAng={}",
                    (int)h_views.size(), params.iPAng);
                return false;
            }

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            // r = b - Ax（残差，正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_r_, sino_n * sizeof(float)));
            // p = A^T r（搜索方向，体积空间）
            YK_CUDA_CHECK(cudaMalloc(&d_p_, vol_n * sizeof(float)));
            // q = A p（正弦图空间）
            YK_CUDA_CHECK(cudaMalloc(&d_q_, sino_n * sizeof(float)));
            // s = A^T r（体积空间，临时）
            YK_CUDA_CHECK(cudaMalloc(&d_s_, vol_n * sizeof(float)));
            // 上一次 x 的备份（用于发散时回退）
            YK_CUDA_CHECK(cudaMalloc(&d_x_prev_, vol_n * sizeof(float)));
            // 正投影临时缓冲
            YK_CUDA_CHECK(cudaMalloc(&d_ax_, sino_n * sizeof(float)));

            fp_.init(params, cfg.fp_task, deviceId);
            bp_.init(params, cfg.bp_task, deviceId);

            is_initialized_ = true;
            YK_LOGI("[CGLSEx] init OK: {} angles (external geometry)",
                params.iPAng);
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            int re_init_at = -1;
            float l2_prev = std::numeric_limits<float>::max();

            // ── 初始化 ───────────────────────────────────────────────
            initialize_(d_sino_meas, d_vol, params, stream, vol_n, sino_n);
            float gamma = compute_gamma_(vol_n, stream);

            for (int iter = 0; iter < cfg_.n_iter; ++iter)
            {
                Util::CpuTimer iter_timer("CGLSEx iter total");

                // ── 备份当前 x ───────────────────────────────────────
                YK_CUDA_CHECK(cudaMemcpyAsync(d_x_prev_, d_vol,
                    vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                // q = A p
                YK_CUDA_CHECK(cudaMemsetAsync(d_q_, 0,
                    sino_n * sizeof(float), stream));
                {
                    Util::CpuTimer t("CGLSEx fp:q=Ap");
                    fp_.run(d_p_, params, h_views_, d_q_, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                // alpha = gamma / ||q||^2
                const float q_norm2 = dot_device_(d_q_, d_q_, sino_n, stream);
                if (q_norm2 < cfg_.eps) {
                    YK_LOGI("[CGLSEx] q_norm^2 too small, stop at iter {}", iter);
                    break;
                }
                const float alpha = gamma / q_norm2;

                // x += alpha * p
                YK::Iter::axpy_launch(d_vol, d_p_, alpha, vol_n, stream);

                // ── 收敛检查（l2 norm of b - Ax）─────────────────────
                YK_CUDA_CHECK(cudaMemsetAsync(d_ax_, 0,
                    sino_n * sizeof(float), stream));
                {
                    Util::CpuTimer t("CGLSEx fp:Ax(convergence check)");
                    fp_.run(d_vol, params, h_views_, d_ax_, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                YK::Iter::residual_launch(d_sino_meas, d_ax_,
                    d_ax_, sino_n, stream);
                const float l2_cur = std::sqrt(
                    dot_device_(d_ax_, d_ax_, sino_n, stream));

                YK_LOGI("[CGLSEx] iter {}/{} l2={:.6e}", iter + 1, cfg_.n_iter, l2_cur);

                // ── 发散检测 ─────────────────────────────────────────
                if (iter > 0 && l2_cur > l2_prev) {
                    // 回退 x
                    YK_CUDA_CHECK(cudaMemcpyAsync(d_vol, d_x_prev_,
                        vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                    YK_LOGW("[CGLSEx] divergence at iter {}, rollback", iter);

                    if (re_init_at + 1 == iter || !cfg_.restart) {
                        YK_LOGW("[CGLSEx] exited due to divergence");
                        break;
                    }
                    re_init_at = iter;

                    // 重新初始化
                    initialize_(d_sino_meas, d_vol, params, stream, vol_n, sino_n);
                    gamma = compute_gamma_(vol_n, stream);
                    l2_prev = std::numeric_limits<float>::max();
                    continue;
                }
                l2_prev = l2_cur;

                // r -= alpha * q
                YK::Iter::axpy_launch(d_r_, d_q_, -alpha, sino_n, stream);

                // s = A^T r
                {
                    Util::CpuTimer t("CGLSEx bp:s=A^Tr");
                    bp_.run(d_r_, params, h_views_, stream, d_s_, /*clear_vol=*/true, deviceId_);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                // gamma1 = ||s||^2
                const float gamma1 = dot_device_(d_s_, d_s_, vol_n, stream);

                // beta = gamma1 / gamma
                const float beta = gamma1 / (gamma + cfg_.eps);
                gamma = gamma1;

                YK::Iter::scale_launch(d_p_, beta, vol_n, stream);       // p *= beta
                YK::Iter::axpy_launch(d_p_, d_s_, 1.0f, vol_n, stream);  // p += s

                // ── 约束 ─────────────────────────────────────────────
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(
                        d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(
                        d_vol, vol_n, cfg_.max_constraint, stream);
            }
            return true;
        }

        void release()
        {
            if (d_r_) { cudaFree(d_r_);      d_r_ = nullptr; }
            if (d_p_) { cudaFree(d_p_);      d_p_ = nullptr; }
            if (d_q_) { cudaFree(d_q_);      d_q_ = nullptr; }
            if (d_s_) { cudaFree(d_s_);      d_s_ = nullptr; }
            if (d_x_prev_) { cudaFree(d_x_prev_); d_x_prev_ = nullptr; }
            if (d_ax_) { cudaFree(d_ax_);     d_ax_ = nullptr; }
            h_views_.clear();
            fp_.release();
            bp_.release();
            is_initialized_ = false;
        }

        ~CGLSEx() { release(); }

    private:
        void initialize_(
            const float* d_sino_meas,
            const float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream,
            size_t vol_n, size_t sino_n)
        {
            // r = b - A x
            YK_CUDA_CHECK(cudaMemsetAsync(d_r_, 0,
                sino_n * sizeof(float), stream));
            {
                Util::CpuTimer t("CGLSEx init:fp Ax");
                fp_.run(d_vol, params, h_views_, d_r_, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            YK::Iter::residual_launch(d_sino_meas, d_r_,
                d_r_, sino_n, stream);

            // p = A^T r
            {
                Util::CpuTimer t("CGLSEx init:bp A^Tr");
                bp_.run(d_r_, params, h_views_, stream, d_p_, /*clear_vol=*/true, deviceId_);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }

        float compute_gamma_(size_t vol_n, cudaStream_t stream)
        {
            return dot_device_(d_p_, d_p_, vol_n, stream);
        }

        float dot_device_(const float* a, const float* b,
            size_t n, cudaStream_t stream)
        {
            float result = 0.f;
            YK::Iter::dot_launch(a, b, n, &result, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            return result;
        }

        bool         is_initialized_ = false;
        int          deviceId_ = 0;
        SCBCTParams  params_;
        Config       cfg_;

        std::vector<SConeProjGeomVec> h_views_;

        ConeProjectorEx     fp_;
        ConeBackprojectorEx bp_;

        float* d_r_ = nullptr;   // 残差（正弦图空间）
        float* d_p_ = nullptr;   // 搜索方向（体积空间）
        float* d_q_ = nullptr;   // A*p（正弦图空间）
        float* d_s_ = nullptr;   // A^T*r（体积空间）
        float* d_x_prev_ = nullptr;   // x 备份
        float* d_ax_ = nullptr;   // 正投影临时缓冲
    };

    YK_INLINE bool cgls_ex_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& h_views,
        cudaStream_t stream,
        CGLSEx::Config cfg = {})
    {
        CGLSEx recon;
        if (!recon.init(params, cfg, h_views, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }

} // namespace YK
