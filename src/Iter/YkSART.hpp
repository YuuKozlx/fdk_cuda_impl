// YkSART.hpp
#pragma once
#include "FP/YkFPRunner.hpp"
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

    class SART {
    public:
        struct Config {
            int   n_iter = 10;
            float lambda = 1.0f;
            float eps = 1e-6f;   // 防零除
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_Siddon_VoxDriven;
        };

        bool init(const SCBCTParams& params, const Config& cfg,
            cudaStream_t stream, int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

            // 分配工作缓冲区
            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, sino_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, sino_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_weight_, vol_n * sizeof(float)));

            // 初始化 FP/BP
            fp_.init(params, cfg.fp_task, deviceId);
            bp_.init(params, cfg.bp_task, deviceId);

            // 预计算归一化权重：全1正弦图反投影
            precomputeWeight_(params, stream, deviceId);

            is_initialized_ = true;
            return true;
        }

        // d_sino_meas：测量正弦图，GPU
        // d_vol：初始体积（通常全零），GPU，原地更新
        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t       stream)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_elems = (size_t)params.iPU * params.iPV;
            const int    Na = params.iPAng;

            for (int iter = 0; iter < cfg_.n_iter; ++iter)
            {
                // 逐角度 SART 更新
                for (int ia = 0; ia < Na; ++ia)
                {
                    // 构建单角度参数
                    SCBCTParams p1 = params;
                    p1.iPAng = 1;
                    p1.angle_list = { params.angle_list[ia] };

                    const float* d_sino_i = d_sino_meas + ia * view_elems;

                    // Step 1：正投影 Ax_i
                    YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0,
                        view_elems * sizeof(float), stream));
                    fp_.run(d_vol, p1, d_sino_fwd_, stream);

                    // Step 2：残差 r_i = sino_i - Ax_i
                    YK::Iter::residual_launch(d_sino_i, d_sino_fwd_,
                        d_residual_, view_elems, stream);

                    // Step 3：反投影 A^T r_i
                    YK_CUDA_CHECK(cudaMemsetAsync(d_bp_, 0,
                        vol_n * sizeof(float), stream));
                    bp_.run(d_residual_, p1, d_bp_, stream, /*clear_vol=*/true);

                    // Step 4：更新 x += λ * bp / w
                    YK::Iter::update_launch(d_vol, d_bp_, d_weight_,
                        cfg_.lambda, cfg_.eps,
                        vol_n, stream);
                }

                YK_LOGI("[SART] iter {}/{} done", iter + 1, cfg_.n_iter);
            }
            return true;
        }

        void release()
        {
            if (d_sino_fwd_) { cudaFree(d_sino_fwd_); d_sino_fwd_ = nullptr; }
            if (d_residual_) { cudaFree(d_residual_); d_residual_ = nullptr; }
            if (d_bp_) { cudaFree(d_bp_);       d_bp_ = nullptr; }
            if (d_weight_) { cudaFree(d_weight_);   d_weight_ = nullptr; }
            is_initialized_ = false;
        }

        ~SART() { release(); }

    private:
        void precomputeWeight_(const SCBCTParams& params,
            cudaStream_t stream, int deviceId)
        {
            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;

            // 全1正弦图
            float* d_ones = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_ones, sino_n * sizeof(float)));
            // thrust 或 kernel 填充为 1
            YK::Iter::fill_ones_launch(d_ones, sino_n, stream);

            // 反投影全1正弦图
            YK_CUDA_CHECK(cudaMemsetAsync(d_weight_, 0,
                vol_n * sizeof(float), stream));
            bp_.run(d_ones, params, d_weight_, stream, /*clear_vol=*/true);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaFree(d_ones);
            YK_LOGI("[SART] weight precomputed");
        }

        bool           is_initialized_ = false;
        SCBCTParams    params_;
        Config         cfg_;

        ConeProjector      fp_;
        ConeBackprojector  bp_;

        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_bp_ = nullptr;
        float* d_weight_ = nullptr;
    };

    // 便捷函数
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


 

        class SIRT {
        public:
            struct Config {
                int   n_iter = 50;
                float lambda = 1.0f;
                float lambda_red = 1.0f;  // SIRT 默认不衰减，需要时手动开启
                float eps = 1e-6f;
                bool  use_min = false;
                float min_constraint = 0.f;
                bool  use_max = false;
                float max_constraint = 1e30f;
                int   n_batch = 4;
                ETask fp_task = ETask::FP_Joseph;
                ETask bp_task = ETask::BP_Siddon_VoxDriven;
            };

            bool init(const SCBCTParams& params, const Config& cfg,
                cudaStream_t stream, int deviceId = 0)
            {
                params_ = params;
                cfg_ = cfg;
                lambda_cur_ = cfg.lambda;

                const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
                const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
                const int    Na = params.iPAng;
                const int    batch_Na = (Na + cfg.n_batch - 1) / cfg.n_batch;
                const size_t max_batch_sino_n = (size_t)batch_Na * params.iPU * params.iPV;

                YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_batch_sino_n * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_pixel_weight_, vol_n * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_line_weight_all_, sino_n * sizeof(float)));

                fp_.init(params, cfg.fp_task, deviceId);
                bp_.init(params, cfg.bp_task, deviceId);

                precomputeWeight_(params, stream);
                precomputeBatches_(params, stream);

                is_initialized_ = true;
                YK_LOGI("[SIRT] init OK: {} angles, {} batches", params.iPAng, cfg.n_batch);
                return true;
            }

            bool iterate(
                const float* d_sino_meas,
                float* d_vol,
                const SCBCTParams& params,
                cudaStream_t       stream,
                unsigned int       iterations)
            {
                if (!is_initialized_) return false;

                const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
                const size_t view_n = (size_t)params.iPU * params.iPV;
                const int    Na = params.iPAng;
                const int    batch_Na = (Na + cfg_.n_batch - 1) / cfg_.n_batch;
                const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

                float* d_sino_batch = nullptr;
                float* d_sino_fwd_batch = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_sino_batch, max_batch_sino_n * sizeof(float)));
                YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_batch, max_batch_sino_n * sizeof(float)));

                struct Guard {
                    float** a; float** b;
                    ~Guard() { cudaFree(*a); cudaFree(*b); }
                } guard{ &d_sino_batch, &d_sino_fwd_batch };

                for (unsigned int iter = 0; iter < iterations; ++iter)
                {
                    for (int b = 0; b < cfg_.n_batch; ++b)
                    {
                        const auto& bm = batches_[b];

                        // 本批测量正弦图
                        YK_CUDA_CHECK(cudaMemcpyAsync(
                            d_sino_batch,
                            d_sino_meas + bm.start_angle * view_n,
                            bm.sino_n * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream));

                        // Step1：正投影
                        YK_CUDA_CHECK(cudaMemsetAsync(
                            d_sino_fwd_batch, 0, bm.sino_n * sizeof(float), stream));
                        fp_.run(d_vol, bm.pb, d_sino_fwd_batch, stream);

                        // Step2：残差原地
                        YK::Iter::residual_launch(
                            d_sino_batch, d_sino_fwd_batch,
                            d_sino_fwd_batch, bm.sino_n, stream);

                        // Step3：行归一化，取全量 buffer 偏移指针，零拷贝
                        float* d_lw = d_line_weight_all_ + bm.start_angle * view_n;
                        YK::Iter::mul_launch(
                            d_sino_fwd_batch, d_lw, bm.sino_n, stream);

                        // Step4：反投影累加，第一批清零
                        bp_.run(d_sino_fwd_batch, bm.pb, d_bp_, stream,
                            /*clear_vol=*/(b == 0));
                    }

                    // Step5：体素更新
                    YK::Iter::addmul_launch(d_vol, d_bp_, d_pixel_weight_, vol_n, stream);

                    // lambda 衰减
                    lambda_cur_ *= cfg_.lambda_red;

                    // Step6：约束
                    if (cfg_.use_min)
                        YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
                    if (cfg_.use_max)
                        YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

                    if ((iter + 1) % 5 == 0 || iter + 1 == iterations) {
                        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                        const size_t cx =
                            (size_t)(params.iVZ / 2) * params.iVY * params.iVX
                            + (size_t)(params.iVY / 2) * params.iVX
                            + params.iVX / 2;
                        float h_center = 0.f;
                        cudaMemcpy(&h_center, d_vol + cx, sizeof(float),
                            cudaMemcpyDeviceToHost);
                        YK_LOGI("[SIRT] iter {}/{} vol_center={:.6e} lambda={:.4e}",
                            iter + 1, iterations, h_center, lambda_cur_);
                    }
                }
                return true;
            }

            bool run(const float* d_sino_meas,
                float* d_vol,
                const SCBCTParams& params,
                cudaStream_t       stream)
            {
                if (!is_initialized_) return false;
                return iterate(d_sino_meas, d_vol, params, stream, cfg_.n_iter);
            }

            void reset()
            {
                lambda_cur_ = cfg_.lambda;
            }

            void release()
            {
                if (d_sino_fwd_) { cudaFree(d_sino_fwd_);        d_sino_fwd_ = nullptr; }
                if (d_bp_) { cudaFree(d_bp_);              d_bp_ = nullptr; }
                if (d_pixel_weight_) { cudaFree(d_pixel_weight_);    d_pixel_weight_ = nullptr; }
                if (d_line_weight_all_) { cudaFree(d_line_weight_all_); d_line_weight_all_ = nullptr; }

                batches_.clear();

                fp_.release();
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

            void precomputeWeight_(const SCBCTParams& params, cudaStream_t stream)
            {
                const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
                const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;

                float* d_ones = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_ones, sino_n * sizeof(float)));
                YK::Iter::fill_ones_launch(d_ones, sino_n, stream);
                bp_.run(d_ones, params, d_pixel_weight_, stream, /*clear_vol=*/true);
                cudaFree(d_ones);

                YK::Iter::invert_scale_launch(
                    d_pixel_weight_, cfg_.lambda, cfg_.eps, vol_n, stream);

                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                YK_LOGI("[SIRT] pixel weight precomputed");
            }

            void precomputeBatches_(const SCBCTParams& params, cudaStream_t stream)
            {
                const size_t view_n = (size_t)params.iPU * params.iPV;
                const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
                const int    Na = params.iPAng;
                const int    n_batch = cfg_.n_batch;
                const int    batch_Na = (Na + n_batch - 1) / n_batch;

                batches_.resize(n_batch);

                float* d_vol_ones = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_vol_ones, vol_n * sizeof(float)));
                YK::Iter::fill_ones_launch(d_vol_ones, vol_n, stream);

                for (int b = 0; b < n_batch; ++b) {
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

                    float* dst = d_line_weight_all_ + bm.start_angle * view_n;
                    YK_CUDA_CHECK(cudaMemsetAsync(dst, 0, bm.sino_n * sizeof(float), stream));
                    fp_.run(d_vol_ones, bm.pb, dst, stream);
                    YK::Iter::invert_launch(dst, cfg_.eps, bm.sino_n, stream);
                }

                cudaFree(d_vol_ones);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                YK_LOGI("[SIRT] {} batches precomputed, line weights on GPU", n_batch);
            }

            bool        is_initialized_ = false;
            float       lambda_cur_ = 1.0f;
            SCBCTParams params_;
            Config      cfg_;

            ConeProjector     fp_;
            ConeBackprojector bp_;

            float* d_sino_fwd_ = nullptr;
            float* d_bp_ = nullptr;
            float* d_pixel_weight_ = nullptr;
            float* d_line_weight_all_ = nullptr;

            std::vector<BatchMeta> batches_;
        };

        YK_INLINE bool sirt_reconstruct(
            const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t       stream,
            SIRT::Config       cfg = {})
        {
            SIRT recon;
            if (!recon.init(params, cfg, stream)) return false;
            return recon.run(d_sino_meas, d_vol, params, stream);
        }



    // YkSIRT.hpp
    //class SIRT {
    //public:
    //    struct Config {
    //        int   n_iter = 50;    // SIRT 收敛慢，通常需要更多迭代
    //        float lambda = 1.0f;
    //        float eps = 1e-6f;
    //        bool  use_min = false;
    //        float min_constraint = 0.f;
    //        bool  use_max = false;
    //        float max_constraint = 1e30f;
    //        ETask fp_task = ETask::FP_Joseph;
    //        ETask bp_task = ETask::BP_Siddon_VoxDriven;
    //    };

    //    bool init(const SCBCTParams& params, const Config& cfg,
    //        cudaStream_t stream, int deviceId = 0)
    //    {
    //        params_ = params;
    //        cfg_ = cfg;

    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

    //        YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, sino_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_residual_, sino_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_weight_, vol_n * sizeof(float)));

    //        fp_.init(params, cfg.fp_task, deviceId);
    //        bp_.init(params, cfg.bp_task, deviceId);

    //        precomputeWeight_(params, stream, deviceId);

    //        is_initialized_ = true;
    //        return true;
    //    }

    //    bool iterate(
    //        const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t       stream,
    //        unsigned int       iterations)
    //    {
    //        if (!is_initialized_) return false;

    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;

    //        for (unsigned int iter = 0; iter < iterations; ++iter)
    //        {
    //            // ── Step1：全部角度正投影 A * x ──────────────────────
    //            YK_CUDA_CHECK(cudaMemsetAsync(
    //                d_sino_fwd_, 0, sino_n * sizeof(float), stream));

    //            size_t free_mem, total_mem;
    //            cudaMemGetInfo(&free_mem, &total_mem);
    //            YK_LOGI("[SIRT] before fp_.run: free={:.1f}MB total={:.1f}MB",
    //                free_mem / 1e6, total_mem / 1e6);

    //            {
    //                YK::Util::CudaTimer timer("sirt_forward_projection", stream);
    //                fp_.run(d_vol, params, d_sino_fwd_, stream);
    //            }


    //            {
    //                YK::Util::CudaTimer timer("sirt_residual", stream);
    //                // ── Step2：残差 r = sino - A * x ─────────────────────
    //                YK::Iter::residual_launch(
    //                    d_sino_meas, d_sino_fwd_,
    //                    d_residual_, sino_n, stream);
    //            }


    //            // ── Step3：反投影 bp = A^T * r ────────────────────────
    //            {
    //                YK::Util::CudaTimer timer("sirt_backprojection", stream);
    //                bp_.run(d_residual_, params, d_bp_, stream, /*clear_vol=*/true);
    //            }

    //            // ── Step4：更新 x += λ * bp / (w + eps) ──────────────
    //            {
    //                YK::Util::CudaTimer timer("sirt_update", stream);   
    //                YK::Iter::update_launch(
    //                    d_vol, d_bp_, d_weight_,
    //                    cfg_.lambda, cfg_.eps,
    //                    vol_n, stream);
    //            }
    //          

    //            // ── Step5：约束 ───────────────────────────────────────
    //            if (cfg_.use_min || cfg_.use_max) {
    //                YK::Iter::clamp_launch(d_vol, vol_n,
    //                    cfg_.use_min ? cfg_.min_constraint : -1e30f,
    //                    cfg_.use_max ? cfg_.max_constraint : 1e30f,
    //                    stream);
    //            }

    //            // 每5次同步一次打 log，不影响性能
    //            if ((iter + 1) % 5 == 0 || iter + 1 == iterations) {
    //                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //                YK_LOGI("[SIRT] iter {}/{}", iter + 1, iterations);
    //            }
    //        }
    //        return true;
    //    }

    //    bool run(const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t       stream)
    //    {
    //        if (!is_initialized_) return false;
    //        return iterate(d_sino_meas, d_vol, params, stream, cfg_.n_iter);
    //    }

    //    void release()
    //    {
    //        if (d_sino_fwd_) { cudaFree(d_sino_fwd_); d_sino_fwd_ = nullptr; }
    //        if (d_residual_) { cudaFree(d_residual_); d_residual_ = nullptr; }
    //        if (d_bp_) { cudaFree(d_bp_);       d_bp_ = nullptr; }
    //        if (d_weight_) { cudaFree(d_weight_);   d_weight_ = nullptr; }
    //        fp_.release();
    //        bp_.release();
    //        is_initialized_ = false;
    //    }

    //    ~SIRT() { release(); }

    //private:
    //    void precomputeWeight_(const SCBCTParams& params,
    //        cudaStream_t stream, int deviceId)
    //    {
    //        const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;

    //        float* d_ones = nullptr;
    //        YK_CUDA_CHECK(cudaMalloc(&d_ones, sino_n * sizeof(float)));
    //        YK::Iter::fill_ones_launch(d_ones, sino_n, stream);

    //        bp_.run(d_ones, params, d_weight_, stream, /*clear_vol=*/true);
    //        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

    //        cudaFree(d_ones);
    //        YK_LOGI("[SIRT] weight precomputed");
    //    }

    //    bool        is_initialized_ = false;
    //    SCBCTParams params_;
    //    Config      cfg_;

    //    ConeProjector     fp_;
    //    ConeBackprojector bp_;

    //    float* d_sino_fwd_ = nullptr;
    //    float* d_residual_ = nullptr;
    //    float* d_bp_ = nullptr;
    //    float* d_weight_ = nullptr;
    //};

    //YK_INLINE bool sirt_reconstruct(
    //    const float* d_sino_meas,
    //    float* d_vol,
    //    const SCBCTParams& params,
    //    cudaStream_t       stream,
    //    SIRT::Config       cfg = {})
    //{
    //    SIRT recon;
    //    if (!recon.init(params, cfg, stream)) return false;
    //    return recon.run(d_sino_meas, d_vol, params, stream);
    //}

} // namespace YK