#pragma once
#include "common/YkProjectionOperators.hpp"
#include "common/YkDeviceWorkspace.hpp"
#include "kernels/YkIterLaunch.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkLog.h"
#include "util/YkCpuProfiler.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

#include <cuda_runtime.h>
#include <global/YkCBCTParams.h>
#include <vector>
#include <numeric>

namespace YK {
    class CglsAstraBackend {
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
            std::vector<SConeProjGeomVec> geometry;
            detail::buildCircularViews(params, geometry);
            return init(params, cfg, geometry, stream, deviceId);
        }

        bool init(const SCBCTParams& params, const Config& cfg,
            const std::vector<SConeProjGeomVec>& geometry,
            cudaStream_t stream, int deviceId = 0)
        {
            if (static_cast<int>(geometry.size()) != params.iPAng) {
                YK_LOGE("[CglsAstraBackend] geometry size mismatch");
                return false;
            }
            params_ = params;
            cfg_ = cfg;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            d_r_.allocate(sino_n, deviceId);  // r（正弦图空间）
            d_w_.allocate(sino_n, deviceId);  // w = A p（正弦图空间）
            d_p_.allocate(vol_n, deviceId);   // p（体积空间）
            d_z_.allocate(vol_n, deviceId);   // z = A^T r（体积空间）

            fp_.init(params, geometry, cfg.fp_task, deviceId, stream);
            bp_.init(params, geometry, cfg.bp_task, deviceId, stream);

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
            d_r_.reset(); d_w_.reset(); d_p_.reset(); d_z_.reset();
            fp_.release();
            bp_.release();
            is_initialized_ = false;
        }

        ~CglsAstraBackend() { release(); }

    private:
        bool        is_initialized_ = false;
        SCBCTParams params_;
        Config      cfg_;

        ForwardOperatorAdapter fp_;
        BackOperatorAdapter bp_;

        DeviceWorkspaceF32 d_r_; // 残差（正弦图空间）
        DeviceWorkspaceF32 d_w_; // w = A p（正弦图空间）
        DeviceWorkspaceF32 d_p_; // 搜索方向（体积空间）
        DeviceWorkspaceF32 d_z_; // z = A^T r（体积空间）
    };

    YK_INLINE bool cgls_astra_backend_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t stream,
        CglsAstraBackend::Config cfg = {})
    {
        CglsAstraBackend recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }
}


namespace YK {
    // ============================================================
    //  Robust CGLS backend
    //
    //  与 CGLS 的差异仅在于 GeometryContext 由外部 h_views 构造，
    //  FP/BP 仍使用同一套通用 operator，不再维护第二套 runner。
    //
    //  迭代算法本身（CG 数学结构、收敛判据、回退重启逻辑）与
    //  CGLS 完全一致，不做任何改动。
    // ============================================================
    class CglsRobustBackend {
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
                YK_LOGE("[CglsRobustBackend] geometry size mismatch: {} vs iPAng={}",
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

            if (!fp_.init(params, h_views_, cfg.fp_task, deviceId, stream) ||
                !bp_.init(params, h_views_, cfg.bp_task, deviceId, stream)) {
                release();
                return false;
            }

            is_initialized_ = true;
            YK_LOGI("[CglsRobustBackend] init OK: {} angles",
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
                Util::CpuTimer iter_timer("CglsRobustBackend iteration");

                // ── 备份当前 x ───────────────────────────────────────
                YK_CUDA_CHECK(cudaMemcpyAsync(d_x_prev_, d_vol,
                    vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                // q = A p
                YK_CUDA_CHECK(cudaMemsetAsync(d_q_, 0,
                    sino_n * sizeof(float), stream));
                {
                    Util::CpuTimer t("CglsRobustBackend fp:q=Ap");
                    fp_.run(d_p_, params, d_q_, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                }

                // alpha = gamma / ||q||^2
                const float q_norm2 = dot_device_(d_q_, d_q_, sino_n, stream);
                if (q_norm2 < cfg_.eps) {
                    YK_LOGI("[CglsRobustBackend] q_norm^2 too small, stop at iter {}", iter);
                    break;
                }
                const float alpha = gamma / q_norm2;

                // x += alpha * p
                YK::Iter::axpy_launch(d_vol, d_p_, alpha, vol_n, stream);

                // ── 收敛检查（l2 norm of b - Ax）─────────────────────
                YK_CUDA_CHECK(cudaMemsetAsync(d_ax_, 0,
                    sino_n * sizeof(float), stream));
                {
                    Util::CpuTimer t("CglsRobustBackend fp:Ax check");
                    fp_.run(d_vol, params, d_ax_, stream);
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                YK::Iter::residual_launch(d_sino_meas, d_ax_,
                    d_ax_, sino_n, stream);
                const float l2_cur = std::sqrt(
                    dot_device_(d_ax_, d_ax_, sino_n, stream));

                YK_LOGI("[CglsRobustBackend] iter {}/{} l2={:.6e}",
                    iter + 1, cfg_.n_iter, l2_cur);

                // ── 发散检测 ─────────────────────────────────────────
                if (iter > 0 && l2_cur > l2_prev) {
                    // 回退 x
                    YK_CUDA_CHECK(cudaMemcpyAsync(d_vol, d_x_prev_,
                        vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));

                    YK_LOGW("[CglsRobustBackend] divergence at iter {}, rollback", iter);

                    if (re_init_at + 1 == iter || !cfg_.restart) {
                        YK_LOGW("[CglsRobustBackend] exited due to divergence");
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
                    Util::CpuTimer t("CglsRobustBackend bp:s=A^Tr");
                    bp_.run(d_r_, params, d_s_, stream, /*clear_vol=*/true);
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

        ~CglsRobustBackend() { release(); }

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
                Util::CpuTimer t("CglsRobustBackend init:fp Ax");
                fp_.run(d_vol, params, d_r_, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            YK::Iter::residual_launch(d_sino_meas, d_r_,
                d_r_, sino_n, stream);

            // p = A^T r
            {
                Util::CpuTimer t("CglsRobustBackend init:bp A^Tr");
                bp_.run(d_r_, params, d_p_, stream, /*clear_vol=*/true);
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

        ForwardOperatorAdapter fp_;
        BackOperatorAdapter bp_;

        float* d_r_ = nullptr;   // 残差（正弦图空间）
        float* d_p_ = nullptr;   // 搜索方向（体积空间）
        float* d_q_ = nullptr;   // A*p（正弦图空间）
        float* d_s_ = nullptr;   // A^T*r（体积空间）
        float* d_x_prev_ = nullptr;   // x 备份
        float* d_ax_ = nullptr;   // 正投影临时缓冲
    };

    YK_INLINE bool cgls_robust_backend_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& h_views,
        cudaStream_t stream,
        CglsRobustBackend::Config cfg = {})
    {
        CglsRobustBackend recon;
        if (!recon.init(params, cfg, h_views, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }

} // namespace YK
