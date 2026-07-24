// YkSART.hpp
#pragma once
#include "common/YkProjectionOperators.hpp"
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

    class SART {
    public:
        struct Config {
            int   n_iter = 10;
            float lambda = 1.0f;
            float lambda_red = 1.0f;
            float eps = 1e-6f;
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
            lambda_cur_ = cfg.lambda;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;

            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, view_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, view_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_row_w_, view_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

            fp_.init(params, cfg.fp_task, deviceId, stream);
            bp_.init(params, cfg.bp_task, deviceId, stream);

            // R uses a temporary 2x2x2 volume.  It must have its own FP
            // geometry/texture contract; reusing fp_ would interpret an
            // eight-float buffer as the full reconstruction volume.
            const float sVolX = params.iVX * params.vox_x_mm;
            const float sVolY = params.iVY * params.vox_y_mm;
            const float sVolZ = params.iVZ * params.vox_z_mm;
            const float sDetZ = params.iPV * params.dv_mm;
            params_lo_ = params;
            params_lo_.iVX = 2; params_lo_.iVY = 2; params_lo_.iVZ = 2;
            params_lo_.vox_x_mm = sVolX * 1.1f / 2.f;
            params_lo_.vox_y_mm = sVolY * 1.1f / 2.f;
            params_lo_.vox_z_mm = std::max(sDetZ, sVolZ) / 2.f;
            fp_lo_.init(params_lo_, cfg.fp_task, deviceId, stream);

            // ── 预计算 C = A^T · 1_proj（全局，所有角度）────────────
            {
                const size_t sino_n = (size_t)params.iPAng * view_n;

                // 缩小体积 *scale
                const float sVolX = params.iVX * params.vox_x_mm;
                const float sVolY = params.iVY * params.vox_y_mm;
                const float norm_xy = std::sqrt(sVolX * sVolX + sVolY * sVolY);
                const float scale = (norm_xy > 1e-8f)
                    ? std::max(sVolX, sVolY) / norm_xy * 0.9f
                    : 0.9f;

                SCBCTParams ps_c = params;
                ps_c.vox_x_mm *= scale;
                ps_c.vox_y_mm *= scale;
                ps_c.vox_z_mm *= scale;

                float* d_ones_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_ones_sino, sino_n * sizeof(float)));
                YK::Iter::fill_ones_launch(d_ones_sino, sino_n, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                bp_.run(d_ones_sino, ps_c, d_col_w_, stream, /*clear_vol=*/true);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                cudaFree(d_ones_sino);

                YK::Iter::threshold_inf_launch(d_col_w_, vol_n, 0.f, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            is_initialized_ = true;
            YK_LOGI("[SART] init OK: {} angles", params.iPAng);
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t       stream)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;
            const int    Na = params.iPAng;

            for (int iter = 0; iter < cfg_.n_iter; ++iter)
            {
                for (int ia = 0; ia < Na; ++ia)
                {
                    // 构建单角度参数
                    SCBCTParams p1 = params;
                    p1.iPAng = 1;
                    p1.angle_list = { params.angle_list[ia] };

                    // ── 实时计算 R = A_i · 1_vol（单角度，扩大体积粗网格）
                    {
                        SCBCTParams ps_r = params_lo_;
                        ps_r.iPAng = 1;
                        ps_r.angle_list = p1.angle_list;

                        float* d_ones_coarse = nullptr;
                        YK_CUDA_CHECK(cudaMalloc(&d_ones_coarse, 8 * sizeof(float)));
                        YK::Iter::fill_ones_launch(d_ones_coarse, 8, stream);

                        YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_, 0,
                            view_n * sizeof(float), stream));
                        fp_lo_.run(d_ones_coarse, ps_r, d_row_w_, stream);
                        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                        cudaFree(d_ones_coarse);

                        const float min_vox = std::min({
                            ps_r.vox_x_mm, ps_r.vox_y_mm, ps_r.vox_z_mm });
                        YK::Iter::threshold_inf_launch(
                            d_row_w_, view_n, min_vox / 2.f, stream);
                        YK::Iter::rcp_launch(d_row_w_, view_n, stream);
                    }

                    // Step1: A_i x
                    YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0,
                        view_n * sizeof(float), stream));
                    fp_.run(d_vol, p1, d_sino_fwd_, stream);

                    // Step2: r = b_i - A_i x
                    YK::Iter::residual_launch(
                        d_sino_meas + ia * view_n, d_sino_fwd_,
                        d_residual_, view_n, stream);

                    // Step2.5: r = W_i ⊙ r = r ⊘ R_i
                    YK::Iter::multiply_launch(
                        d_residual_, d_row_w_, view_n, stream);

                    // Step3: bp = A_i^T r
                    bp_.run(d_residual_, p1, d_bp_, stream, /*clear_vol=*/true);

                    // Step4: x += lambda * bp ⊘ C
                    YK::Iter::update_launch(
                        d_vol, d_bp_, d_col_w_,
                        lambda_cur_, cfg_.eps,
                        vol_n, stream);

                    // Step5: 约束
                    if (cfg_.use_min)
                        YK::Iter::clamp_min_launch(
                            d_vol, vol_n, cfg_.min_constraint, stream);
                    if (cfg_.use_max)
                        YK::Iter::clamp_max_launch(
                            d_vol, vol_n, cfg_.max_constraint, stream);
                }

                lambda_cur_ *= cfg_.lambda_red;
                YK_LOGI("[SART] iter {}/{} done", iter + 1, cfg_.n_iter);
            }
            return true;
        }

        void release()
        {
            if (d_sino_fwd_) { cudaFree(d_sino_fwd_); d_sino_fwd_ = nullptr; }
            if (d_residual_) { cudaFree(d_residual_); d_residual_ = nullptr; }
            if (d_row_w_) { cudaFree(d_row_w_);    d_row_w_ = nullptr; }
            if (d_bp_) { cudaFree(d_bp_);       d_bp_ = nullptr; }
            if (d_col_w_) { cudaFree(d_col_w_);    d_col_w_ = nullptr; }
            fp_.release();
            fp_lo_.release();
            bp_.release();
            is_initialized_ = false;
            lambda_cur_ = 1.0f;
        }

        ~SART() { release(); }

    private:
        bool        is_initialized_ = false;
        float       lambda_cur_ = 1.0f;
        SCBCTParams params_;
        SCBCTParams params_lo_;
        Config      cfg_;

        ForwardOperatorAdapter fp_;
        ForwardOperatorAdapter fp_lo_;
        BackOperatorAdapter bp_;

        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_row_w_ = nullptr;
        float* d_bp_ = nullptr;
        float* d_col_w_ = nullptr;
    };

    YK_INLINE bool sart_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t       stream,
        SART::Config       cfg = {})
    {
        SART sart;
        if (!sart.init(params, cfg, stream)) return false;
        return sart.run(d_sino_meas, d_vol, params, stream);
    }
};


namespace YK {

    class SIRT {
    public:
        struct Config {
            int   n_iter = 50;
            float lambda = 1.0f;
            float lambda_red = 1.0f;
            float eps = 1e-6f;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            int   n_batch = 8;
            bool  dump_debug = false;
            int   row_w_down = 2;
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_Joseph_v3;
        };

        bool init(const SCBCTParams& params, const Config& cfg,
            cudaStream_t stream, int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;
            lambda_cur_ = cfg.lambda;
            deviceId_ = deviceId;

            const int    Na = params.iPAng;
            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;
            const int    batch_Na = (Na + cfg.n_batch - 1) / cfg.n_batch;
            const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

            // ── 预切分 batch（连续切分）──────────────────────────────
            batches_.resize(cfg.n_batch);
            for (int b = 0; b < cfg.n_batch; ++b) {
                auto& bm = batches_[b];
                bm.start_angle = b * batch_Na;
                const int end = std::min(bm.start_angle + batch_Na, Na);
                bm.K = end - bm.start_angle;
                bm.sino_n = (size_t)bm.K * view_n;
                bm.pb = params;
                bm.pb.iPAng = bm.K;
                bm.pb.angle_list = std::vector<float>(
                    params.angle_list.begin() + bm.start_angle,
                    params.angle_list.begin() + end);
            }

            // ── 常驻缓冲 ──────────────────────────────────────────────
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

            fp_.init(params, cfg.fp_task, deviceId, stream);
            bp_.init(params, cfg.bp_task, deviceId, stream);

            // ── 预计算 C = A^T · 1_proj（缩小体积，全分辨率网格）─────
            {
                const float sVolX = params.iVX * params.vox_x_mm;
                const float sVolY = params.iVY * params.vox_y_mm;
                const float norm_xy = std::sqrt(sVolX * sVolX + sVolY * sVolY);
                const float scale = (norm_xy > 1e-8f)
                    ? std::max(sVolX, sVolY) / norm_xy * 0.9f
                    : 0.9f;

                SCBCTParams ps_c = params;
                ps_c.vox_x_mm *= scale;
                ps_c.vox_y_mm *= scale;
                ps_c.vox_z_mm *= scale;

                float* d_ones_sino = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_ones_sino,
                    max_batch_sino_n * sizeof(float)));

                for (int b = 0; b < cfg.n_batch; ++b) {
                    const auto& bm = batches_[b];
                    SCBCTParams pb_c = ps_c;
                    pb_c.iPAng = bm.K;
                    pb_c.angle_list = bm.pb.angle_list;

                    YK::Iter::fill_ones_launch(d_ones_sino, bm.sino_n, stream);
                    bp_.run(d_ones_sino, pb_c, d_col_w_, stream,
                        /*clear_vol=*/(b == 0));
                }
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                cudaFree(d_ones_sino);

                // C == 0 置 inf
                YK::Iter::threshold_inf_launch(d_col_w_, vol_n, 0.f, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // ── 构建低分辨率 R：扩大体积 10%，粗网格 2x2x2 ──────────
            {
                const float sVolX = params.iVX * params.vox_x_mm;
                const float sVolY = params.iVY * params.vox_y_mm;
                const float sVolZ = params.iVZ * params.vox_z_mm;
                const float sDetZ = params.iPV * params.dv_mm;

                params_lo_ = params;
                params_lo_.iVX = 2;
                params_lo_.iVY = 2;
                params_lo_.iVZ = 2;
                params_lo_.vox_x_mm = sVolX * 1.1f / 2.f;
                params_lo_.vox_y_mm = sVolY * 1.1f / 2.f;
                params_lo_.vox_z_mm = std::max(sDetZ, sVolZ) / 2.f;

                YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_lo_, 8 * sizeof(float)));
                YK::Iter::fill_ones_launch(d_ones_vol_lo_, 8, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));

                // The low-resolution R calculation uses the same FP operator
                // contract as the full-resolution solver, but with its own
                // volume geometry and therefore its own adapter instance.
                fp_lo_.init(params_lo_, cfg.fp_task, deviceId, stream);
            }

            // ── DEBUG ─────────────────────────────────────────────────
            if (cfg_.dump_debug)
                YK_LOGI("[SIRT] C dim = {}x{}x{}", params.iVX, params.iVY, params.iVZ);

            is_initialized_ = true;
            YK_LOGI("[SIRT] init OK: {} angles, {} batches; "
                "C TIGRE-style, R 2x2x2 粗网格{}",
                Na, cfg.n_batch,
                cfg_.dump_debug ? " [DUMP ON]" : "");
            return true;
        }

        bool iterate(
            const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream,
            unsigned int iterations)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;
            const int    Na = params.iPAng;
            const int    batch_Na = (Na + cfg_.n_batch - 1) / cfg_.n_batch;
            const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

            float* d_sino_batch = nullptr;
            float* d_work_batch = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_sino_batch,
                max_batch_sino_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_work_batch,
                max_batch_sino_n * sizeof(float)));

            struct Guard {
                float** a; float** b;
                ~Guard() { cudaFree(*a); cudaFree(*b); }
            } guard{ &d_sino_batch, &d_work_batch };

            bool dumped_R = false;

            for (unsigned int iter = 0; iter < iterations; ++iter)
            {
                std::string str = fmt::format("[SIRT] iter {}/{}", iter + 1, iterations);
                Util::CpuTimer timer(str.c_str());

                for (int b = 0; b < cfg_.n_batch; ++b)
                {
                    const auto& bm = batches_[b];

                    // 收集本批测量
                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_sino_batch,
                        d_sino_meas + bm.start_angle * view_n,
                        bm.sino_n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));

                    // Step1: 正投影 A_b x
                    YK_CUDA_CHECK(cudaMemsetAsync(
                        d_work_batch, 0, bm.sino_n * sizeof(float), stream));
                    fp_.run(d_vol, bm.pb, d_work_batch, stream);

                    // Step2: 残差 r = b - A_b x
                    YK::Iter::residual_launch(
                        d_sino_batch, d_work_batch,
                        d_sino_batch, bm.sino_n, stream);

                    // Step3: 现算本批 R = A_b · 1_vol_lo（2x2x2 粗网格）
                    {
                        SCBCTParams pb_lo = params_lo_;
                        pb_lo.iPAng = bm.K;
                        pb_lo.angle_list = bm.pb.angle_list;

                        YK_CUDA_CHECK(cudaMemsetAsync(
                            d_work_batch, 0, bm.sino_n * sizeof(float), stream));
                        fp_lo_.run(d_ones_vol_lo_, pb_lo, d_work_batch, stream);

                        // DEBUG: retain only lightweight geometry logging.
                        // Raw device dumping belongs to an application-level
                        // diagnostic hook, not the reconstruction core.
                        if (cfg_.dump_debug && !dumped_R && iter == 0 && b == 0) {
                            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                            YK_LOGI("[SIRT] R(b0) dim = {}x{}x{}",
                                params.iPU, params.iPV, bm.K);
                            dumped_R = true;
                        }

                        // R <= min(vox_lo)/2 置 inf
                        const float min_vox_lo = std::min({
                            params_lo_.vox_x_mm,
                            params_lo_.vox_y_mm,
                            params_lo_.vox_z_mm });
                        YK::Iter::threshold_inf_launch(
                            d_work_batch, bm.sino_n, min_vox_lo / 2.f, stream);
                    }

                    // Step4: R 行归一化 r /= (R + eps)
                    YK::Iter::divide_launch(
                        d_sino_batch, d_work_batch, cfg_.eps, bm.sino_n, stream);

                    // Step5: 反投影累加 d_bp_ += A_b^T r（首批清零）
                    bp_.run(d_sino_batch, bm.pb, d_bp_, stream,
                        /*clear_vol=*/(b == 0));
                }

                // Step6: 列归一化更新 x += lambda * bp / C
                YK::Iter::update_launch(
                    d_vol, d_bp_, d_col_w_,
                    lambda_cur_, cfg_.eps,
                    vol_n, stream);

                lambda_cur_ *= cfg_.lambda_red;

                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(
                        d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(
                        d_vol, vol_n, cfg_.max_constraint, stream);

                if ((iter + 1) % 5 == 0 || iter + 1 == iterations) {
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                    const size_t cx =
                        (size_t)(params.iVZ / 2) * params.iVY * params.iVX
                        + (size_t)(params.iVY / 2) * params.iVX
                        + params.iVX / 2;
                    float h_center = 0.f;
                    cudaMemcpy(&h_center, d_vol + cx,
                        sizeof(float), cudaMemcpyDeviceToHost);
                    YK_LOGI("[SIRT] iter {}/{} vol_center={:.6e} lambda={:.4e}",
                        iter + 1, iterations, h_center, lambda_cur_);
                }
            }
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t stream)
        {
            if (!is_initialized_) return false;
            return iterate(d_sino_meas, d_vol, params, stream, cfg_.n_iter);
        }

        void reset() { lambda_cur_ = cfg_.lambda; }

        void release()
        {
            if (d_bp_) { cudaFree(d_bp_);           d_bp_ = nullptr; }
            if (d_col_w_) { cudaFree(d_col_w_);        d_col_w_ = nullptr; }
            if (d_ones_vol_lo_) { cudaFree(d_ones_vol_lo_);  d_ones_vol_lo_ = nullptr; }

            batches_.clear();
            fp_.release();
            fp_lo_.release();
            bp_.release();
            is_initialized_ = false;
            lambda_cur_ = 1.0f;
        }

        ~SIRT() { release(); }

    private:
        struct BatchMeta {
            SCBCTParams pb;
            int         K;
            size_t      sino_n;
            int         start_angle;
        };

        bool         is_initialized_ = false;
        float        lambda_cur_ = 1.0f;
        int          deviceId_ = 0;
        SCBCTParams  params_;
        SCBCTParams  params_lo_;
        Config       cfg_;

        ForwardOperatorAdapter fp_;
        ForwardOperatorAdapter fp_lo_;
        BackOperatorAdapter bp_;

        std::vector<BatchMeta> batches_;

        float* d_bp_ = nullptr;
        float* d_col_w_ = nullptr;
        float* d_ones_vol_lo_ = nullptr;
    };

    YK_INLINE bool sirt_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t stream,
        SIRT::Config cfg = {})
    {
        SIRT recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }
} // namespace YK
