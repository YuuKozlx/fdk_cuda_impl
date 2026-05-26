//// YkOSSART.hpp
//#pragma once
//#include "FP/YkFPRunner.hpp"
//#include "kernel/YkIterLaunch.cuh"
//#include "global/YkGlobals.h"
//#include "global/YkMacro.hpp"
//#include "common/YkVecGeo.hpp"
//#include "global/YkLog.h"
//#include "YKCBCT/interface/YkTaskTypes.hpp"
//
//#include <cuda_runtime.h>
//#include "BP/YkSiddonBPRunner.hpp"
//#include <global/YkCBCTParams.h>
//#include <vector>
//#include <numeric>
//
//namespace YK {
//
//    class OSSART {
//    public:
//        struct Config {
//            int   n_iter = 10;
//            int   n_subset = 20;
//            float lambda = 1.0f;
//            float lambda_red = 0.99f;  // 每次 subset 更新后衰减，1.0 表示不衰减
//            float eps = 1e-6f;
//            bool  use_min = false;
//            float min_constraint = 0.f;
//            bool  use_max = false;
//            float max_constraint = 1e30f;
//            ETask fp_task = ETask::FP_Joseph;
//            ETask bp_task = ETask::BP_Siddon_VoxDriven;
//        };
//
//        bool init(const SCBCTParams& params, const Config& cfg,
//            cudaStream_t stream, int deviceId = 0)
//        {
//            params_ = params;
//            cfg_ = cfg;
//            iteration_ = 0;
//            lambda_cur_ = cfg.lambda;
//
//            const int    Na = params.iPAng;
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//            const size_t view_n = (size_t)params.iPU * params.iPV;
//
//            // ── 构建子集（交错采样）并预缓存子集参数 ─────────────────────────────
//            // 子集0: 0, n_subset, 2*n_subset, ...
//            // 子集1: 1, n_subset+1, ...
//            subsets_.resize(cfg.n_subset);
//            subset_params_.resize(cfg.n_subset);
//
//            size_t max_K = 0;
//            for (int s = 0; s < cfg.n_subset; ++s) {
//                auto& idx = subsets_[s];
//                for (int i = s; i < Na; i += cfg.n_subset)
//                    idx.push_back(i);
//                max_K = std::max(max_K, idx.size());
//
//                auto& ps = subset_params_[s];
//                ps = params;
//                ps.iPAng = (int)idx.size();
//                ps.angle_list.resize(idx.size());
//                for (int i = 0; i < (int)idx.size(); ++i)
//                    ps.angle_list[i] = params.angle_list[idx[i]];
//            }
//
//            const size_t max_subset_sino = max_K * view_n;
//
//            YK_CUDA_CHECK(cudaMalloc(&d_sino_meas_sub_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_weight_, vol_n * sizeof(float)));
//
//            fp_.init(params, cfg.fp_task, deviceId);
//            bp_.init(params, cfg.bp_task, deviceId);
//
//            precomputeWeight_(params, stream);
//
//            is_initialized_ = true;
//            YK_LOGI("[OSSART] init OK: {} angles, {} subsets, max {}/subset",
//                Na, cfg.n_subset, max_K);
//            return true;
//        }
//
//        bool iterate(
//            const float* d_sino_meas,
//            float* d_vol,
//            const SCBCTParams& params,
//            cudaStream_t       stream,
//            unsigned int       iterations)
//        {
//            if (!is_initialized_) return false;
//
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//            const size_t view_n = (size_t)params.iPU * params.iPV;
//
//            for (unsigned int iter = 0; iter < iterations; ++iter)
//            {
//                const int subset_idx = iteration_ % cfg_.n_subset;
//                const std::vector<int>& subset = subsets_[subset_idx];
//                const int    K = (int)subset.size();
//                const size_t sino_n = (size_t)K * view_n;
//
//                const SCBCTParams& ps = subset_params_[subset_idx];
//
//                // ── 收集子集测量正弦图 ────────────────────────────────────────────
//                for (int i = 0; i < K; ++i) {
//                    YK_CUDA_CHECK(cudaMemcpyAsync(
//                        d_sino_meas_sub_ + i * view_n,
//                        d_sino_meas + subset[i] * view_n,
//                        view_n * sizeof(float),
//                        cudaMemcpyDeviceToDevice, stream));
//                }
//
//                // ── Step1：正投影 A_s * x ─────────────────────────────────────────
//                YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0, sino_n * sizeof(float), stream));
//                fp_.run(d_vol, ps, d_sino_fwd_, stream);
//
//                // ── Step2：残差 r = sino_s - A_s * x ─────────────────────────────
//                YK::Iter::residual_launch(
//                    d_sino_meas_sub_, d_sino_fwd_,
//                    d_residual_, sino_n, stream);
//
//                // ── Step3：反投影 bp = A_s^T * r ──────────────────────────────────
//                bp_.run(d_residual_, ps, d_bp_, stream, /*clear_vol=*/true);
//
//                // ── Step4：更新 x += λ * bp / (w + eps) ──────────────────────────
//                YK::Iter::update_launch(
//                    d_vol, d_bp_, d_weight_,
//                    lambda_cur_, cfg_.eps,
//                    vol_n, stream);
//
//                // ── lambda 衰减 ───────────────────────────────────────────────────
//                lambda_cur_ *= cfg_.lambda_red;
//
//                // ── Step5：约束 ───────────────────────────────────────────────────
//                if (cfg_.use_min)
//                    YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
//                if (cfg_.use_max)
//                    YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);
//
//                iteration_++;
//
//                YK_LOGD("[OSSART] iter={} subset={}/{} K={} lambda={:.4e}",
//                    iteration_, subset_idx + 1, cfg_.n_subset, K, lambda_cur_);
//            }
//            return true;
//        }
//
//        // n_iter 次完整迭代，每次遍历所有子集
//        bool run(const float* d_sino_meas,
//            float* d_vol,
//            const SCBCTParams& params,
//            cudaStream_t       stream)
//        {
//            if (!is_initialized_) return false;
//            return iterate(d_sino_meas, d_vol, params, stream,
//                cfg_.n_iter * cfg_.n_subset);
//        }
//
//        unsigned int totalIterations() const { return iteration_; }
//
//        void reset()
//        {
//            iteration_ = 0;
//            lambda_cur_ = cfg_.lambda;  // 恢复初始步长
//        }
//
//        void release()
//        {
//            if (d_sino_meas_sub_) { cudaFree(d_sino_meas_sub_); d_sino_meas_sub_ = nullptr; }
//            if (d_sino_fwd_) { cudaFree(d_sino_fwd_);      d_sino_fwd_ = nullptr; }
//            if (d_residual_) { cudaFree(d_residual_);      d_residual_ = nullptr; }
//            if (d_bp_) { cudaFree(d_bp_);            d_bp_ = nullptr; }
//            if (d_weight_) { cudaFree(d_weight_);        d_weight_ = nullptr; }
//
//            subsets_.clear();
//            subset_params_.clear();
//
//            fp_.release();
//            bp_.release();
//            is_initialized_ = false;
//            iteration_ = 0;
//            lambda_cur_ = 1.0f;
//        }
//
//        ~OSSART() { release(); }
//
//    private:
//        void precomputeWeight_(const SCBCTParams& params, cudaStream_t stream)
//        {
//            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//
//            // 近似列归一化：(A^T * 1) / n_subset
//            // 显存只占 1 × vol_n，避免 n_subset × vol_n 压满显存
//            float* d_ones = nullptr;
//            YK_CUDA_CHECK(cudaMalloc(&d_ones, sino_n * sizeof(float)));
//            YK::Iter::fill_ones_launch(d_ones, sino_n, stream);
//            bp_.run(d_ones, params, d_weight_, stream, /*clear_vol=*/true);
//            cudaFree(d_ones);
//
//            YK::Iter::invert_scale_launch(
//                d_weight_, 1.0f / cfg_.n_subset, cfg_.eps, vol_n, stream);
//
//            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//            YK_LOGI("[OSSART] weight precomputed (approx 1/n_subset)");
//        }
//
//        bool         is_initialized_ = false;
//        unsigned int iteration_ = 0;
//        float        lambda_cur_ = 1.0f;
//        SCBCTParams  params_;
//        Config       cfg_;
//
//        ConeProjector     fp_;
//        ConeBackprojector bp_;
//
//        std::vector<std::vector<int>> subsets_;
//        std::vector<SCBCTParams>      subset_params_;
//
//        float* d_sino_meas_sub_ = nullptr;
//        float* d_sino_fwd_ = nullptr;
//        float* d_residual_ = nullptr;
//        float* d_bp_ = nullptr;
//        float* d_weight_ = nullptr;
//    };
//
//    // ── 便捷函数 ─────────────────────────────────────────────────────────────────
//    YK_INLINE bool ossart_reconstruct(
//        const float* d_sino_meas,
//        float* d_vol,
//        const SCBCTParams& params,
//        cudaStream_t       stream,
//        OSSART::Config     cfg = {})
//    {
//        OSSART recon;
//        if (!recon.init(params, cfg, stream)) return false;
//        return recon.run(d_sino_meas, d_vol, params, stream);
//    }
//
//} // namespace YK
// 



//// YkOSSART.hpp
//#pragma once
//#include "FP/YkFPRunner.hpp"
//#include "kernel/YkIterLaunch.cuh"
//#include "global/YkGlobals.h"
//#include "global/YkMacro.hpp"
//#include "common/YkVecGeo.hpp"
//#include "global/YkLog.h"
//#include "YKCBCT/interface/YkTaskTypes.hpp"
//
//#include <cuda_runtime.h>
//#include "BP/YkSiddonBPRunner.hpp"
//#include <global/YkCBCTParams.h>
//#include <vector>
//#include <numeric>
//
//namespace YK {
//
//    class OSSART {
//    public:
//        struct Config {
//            int   n_iter = 10;
//            int   n_subset = 20;
//            float lambda = 1.0f;
//            float lambda_red = 0.99f;  // 每次 subset 更新后衰减，1.0 表示不衰减
//            float eps = 1e-6f;
//            bool  use_min = false;
//            float min_constraint = 0.f;
//            bool  use_max = false;
//            float max_constraint = 1e30f;
//            ETask fp_task = ETask::FP_Joseph;
//            ETask bp_task = ETask::BP_Siddon_VoxDriven;
//        };
//
//        bool init(const SCBCTParams& params, const Config& cfg,
//            cudaStream_t stream, int deviceId = 0)
//        {
//            params_ = params;
//            cfg_ = cfg;
//            iteration_ = 0;
//
//            const int    Na = params.iPAng;
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//            const size_t view_n = (size_t)params.iPU * params.iPV;
//
//            // ── 构建子集（交错采样）────────────────────────────────
//            // 子集0: 0, n_subset, 2*n_subset, ...
//            // 子集1: 1, n_subset+1, ...
//            subsets_.resize(cfg.n_subset);
//            for (int s = 0; s < cfg.n_subset; ++s)
//                for (int i = s; i < Na; i += cfg.n_subset)
//                    subsets_[s].push_back(i);
//
//            // 最大子集角度数（用于分配 buffer）
//            size_t max_K = 0;
//            for (auto& sub : subsets_)
//                max_K = std::max(max_K, sub.size());
//
//            const size_t max_subset_sino = max_K * view_n;
//
//            YK_CUDA_CHECK(cudaMalloc(&d_sino_meas_sub_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_subset_sino * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
//            YK_CUDA_CHECK(cudaMalloc(&d_weight_, vol_n * sizeof(float)));
//
//            fp_.init(params, cfg.fp_task, deviceId);
//            bp_.init(params, cfg.bp_task, deviceId);
//
//            precomputeWeight_(params, stream, deviceId);
//
//            is_initialized_ = true;
//            YK_LOGI("[OSSART] init OK: {} angles, {} subsets, max {}/subset",
//                Na, cfg.n_subset, max_K);
//            return true;
//        }
//
//        bool iterate(
//            const float* d_sino_meas,
//            float* d_vol,
//            const SCBCTParams& params,
//            cudaStream_t       stream,
//            unsigned int       iterations)
//        {
//            if (!is_initialized_) return false;
//
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//            const size_t view_n = (size_t)params.iPU * params.iPV;
//
//            for (unsigned int iter = 0; iter < iterations; ++iter)
//            {
//                const int subset_idx = iteration_ % cfg_.n_subset;
//                const std::vector<int>& subset = subsets_[subset_idx];
//                const int K = (int)subset.size();
//
//                // ── 构建子集参数 ──────────────────────────────────
//                SCBCTParams ps = params;
//                ps.iPAng = K;
//                ps.angle_list.resize(K);
//                for (int i = 0; i < K; ++i)
//                    ps.angle_list[i] = params.angle_list[subset[i]];
//
//                // ── 收集子集测量正弦图 ────────────────────────────
//                for (int i = 0; i < K; ++i) {
//                    YK_CUDA_CHECK(cudaMemcpyAsync(
//                        d_sino_meas_sub_ + i * view_n,
//                        d_sino_meas + subset[i] * view_n,
//                        view_n * sizeof(float),
//                        cudaMemcpyDeviceToDevice, stream));
//                }
//
//                // ── Step1：正投影 A_s * x ─────────────────────────
//                YK_CUDA_CHECK(cudaMemsetAsync(
//                    d_sino_fwd_, 0, K * view_n * sizeof(float), stream));
//                fp_.run(d_vol, ps, d_sino_fwd_, stream);
//
//                // ── Step2：残差 r = sino_s - A_s * x ─────────────
//                YK::Iter::residual_launch(
//                    d_sino_meas_sub_, d_sino_fwd_,
//                    d_residual_, K * view_n, stream);
//
//                // ── Step3：反投影 bp = A_s^T * r ──────────────────
//                bp_.run(d_residual_, ps, d_bp_, stream, /*clear_vol=*/true);
//
//                // ── Step4：更新 x += λ * bp / (w + eps) ──────────
//                YK::Iter::update_launch(
//                    d_vol, d_bp_, d_weight_,
//                    cfg_.lambda, cfg_.eps,
//                    vol_n, stream);
//
//                // ── Step5：约束 ───────────────────────────────────
//                if (cfg_.use_min || cfg_.use_max) {
//                    YK::Iter::clamp_launch(d_vol, vol_n,
//                        cfg_.use_min ? cfg_.min_constraint : -1e30f,
//                        cfg_.use_max ? cfg_.max_constraint : 1e30f,
//                        stream);
//                }
//
//
//                // 第一次子集更新后加，在 iteration_++ 之前
//                if (iteration_ == 0) {
//                    cudaStreamSynchronize(stream);
//
//                    // 检查子集测量正弦图
//                    std::vector<float> h_tmp(K * view_n);
//                    cudaMemcpy(h_tmp.data(), d_sino_meas_sub_, K * view_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] sino_meas_sub: min={:.4e} max={:.4e}",
//                        *std::min_element(h_tmp.begin(), h_tmp.end()),
//                        *std::max_element(h_tmp.begin(), h_tmp.end()));
//
//                    // 检查正投影结果
//                    cudaMemcpy(h_tmp.data(), d_sino_fwd_, K * view_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] sino_fwd: min={:.4e} max={:.4e}",
//                        *std::min_element(h_tmp.begin(), h_tmp.end()),
//                        *std::max_element(h_tmp.begin(), h_tmp.end()));
//
//                    // 检查残差
//                    cudaMemcpy(h_tmp.data(), d_residual_, K * view_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] residual: min={:.4e} max={:.4e}",
//                        *std::min_element(h_tmp.begin(), h_tmp.end()),
//                        *std::max_element(h_tmp.begin(), h_tmp.end()));
//
//                    // 检查权重
//                    std::vector<float> h_vol(vol_n);
//                    cudaMemcpy(h_vol.data(), d_weight_, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] weight: min={:.4e} max={:.4e}",
//                        *std::min_element(h_vol.begin(), h_vol.end()),
//                        *std::max_element(h_vol.begin(), h_vol.end()));
//
//                    // 检查反投影
//                    cudaMemcpy(h_vol.data(), d_bp_, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] bp: min={:.4e} max={:.4e}",
//                        *std::min_element(h_vol.begin(), h_vol.end()),
//                        *std::max_element(h_vol.begin(), h_vol.end()));
//
//                    // 检查更新后体积
//                    cudaMemcpy(h_vol.data(), d_vol, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
//                    YK_LOGI("[diag] vol after 1st update: min={:.4e} max={:.4e}",
//                        *std::min_element(h_vol.begin(), h_vol.end()),
//                        *std::max_element(h_vol.begin(), h_vol.end()));
//                }
//
//                iteration_++;
//                YK_LOGD("[OSSART] iter={} subset={}/{} K={}",
//                    iteration_, subset_idx + 1, cfg_.n_subset, K);
//            }
//            return true;
//        }
//
//        // ── run：n_iter 次完整迭代，每次遍历所有子集 ────────────
//        bool run(const float* d_sino_meas,
//            float* d_vol,
//            const SCBCTParams& params,
//            cudaStream_t       stream)
//        {
//            if (!is_initialized_) return false;
//            return iterate(d_sino_meas, d_vol, params, stream,
//                cfg_.n_iter * cfg_.n_subset);
//        }
//
//        unsigned int totalIterations() const { return iteration_; }
//        void reset() { iteration_ = 0; }
//
//        void release()
//        {
//            if (d_sino_meas_sub_) { cudaFree(d_sino_meas_sub_); d_sino_meas_sub_ = nullptr; }
//            if (d_sino_fwd_) { cudaFree(d_sino_fwd_);      d_sino_fwd_ = nullptr; }
//            if (d_residual_) { cudaFree(d_residual_);      d_residual_ = nullptr; }
//            if (d_bp_) { cudaFree(d_bp_);            d_bp_ = nullptr; }
//            if (d_weight_) { cudaFree(d_weight_);        d_weight_ = nullptr; }
//            fp_.release();
//            bp_.release();
//            is_initialized_ = false;
//            iteration_ = 0;
//        }
//
//        ~OSSART() { release(); }
//
//    private:
//        void precomputeWeight_(const SCBCTParams& params,
//            cudaStream_t stream, int deviceId)
//        {
//            const size_t sino_n = (size_t)params.iPAng * params.iPU * params.iPV;
//            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
//
//            float* d_ones = nullptr;
//            YK_CUDA_CHECK(cudaMalloc(&d_ones, sino_n * sizeof(float)));
//            YK::Iter::fill_ones_launch(d_ones, sino_n, stream);
//
//            bp_.run(d_ones, params, d_weight_, stream, /*clear_vol=*/true);
//            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
//
//            cudaFree(d_ones);
//            YK_LOGI("[OSSART] weight precomputed");
//        }
//
//        bool         is_initialized_ = false;
//        unsigned int iteration_ = 0;
//        SCBCTParams  params_;
//        Config       cfg_;
//
//        ConeProjector     fp_;
//        ConeBackprojector bp_;
//
//        std::vector<std::vector<int>> subsets_;
//
//        float* d_sino_meas_sub_ = nullptr;
//        float* d_sino_fwd_ = nullptr;
//        float* d_residual_ = nullptr;
//        float* d_bp_ = nullptr;
//        float* d_weight_ = nullptr;
//    };
//
//    // ====================================================================
//    // 便捷函数
//    // ====================================================================
//    YK_INLINE bool ossart_reconstruct(
//        const float* d_sino_meas,
//        float* d_vol,
//        const SCBCTParams& params,
//        cudaStream_t       stream,
//        OSSART::Config     cfg = {})
//    {
//        OSSART recon;
//        if (!recon.init(params, cfg, stream)) return false;
//        return recon.run(d_sino_meas, d_vol, params, stream);
//    }
//
//} // namespace YK


// YkOSSART.hpp
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

    // ================================================================
    // OS-SART, 含完整 R 行归一化 + C 列归一化。
    //
    // 显存优化: R/C 权重不预存 (那样需 n_subset * vol_n ~ 8GB),
    // 改为每子集迭代时现算。代价是每次迭代多一次 FP(算 row) +
    // 一次 BP(算 col), 单子集很快, 换来常驻显存仅 ~2GB。
    //
    // 更新公式 (对每个子集 s):
    //   r   = b_s - A_s x
    //   r  /= (A_s · 1_vol  + eps)                  <-- R 行归一化
    //   bp  = A_s^T r
    //   x  += lambda * bp / (A_s^T · 1_proj + eps)  <-- C 列归一化
    //
    // 与 TIGRE 的 OS-SART 一致: FP/BP 即使非精确伴随,
    // 只要 R/C 都用算子真实算出, 迭代仍能干净收敛。
    // ================================================================
    class OSSART {
    public:
        struct Config {
            int   n_iter = 10;
            int   n_subset = 20;
            float lambda = 1.0f;
            float lambda_red = 1.0f;   // 1.0=不衰减; 想衰减用 0.99
            float eps = 1e-6f;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            ETask fp_task = ETask::FP_Joseph;
            // 快的 voxel-driven (BP_Siddon_VoxDriven) 或精确伴随的 Bp_Joseph。
            // 配了正确 R/C 后, unmatched 的快 BP 通常也能收敛。
            ETask bp_task = ETask::BP_Siddon_VoxDriven;
        };

        bool init(const SCBCTParams& params, const Config& cfg,
            cudaStream_t stream, int deviceId = 0)
        {
            params_ = params;
            cfg_ = cfg;
            iteration_ = 0;
            lambda_cur_ = cfg.lambda;

            const int    Na = params.iPAng;
            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;

            // ── 构建子集 (交错采样) 并预缓存子集参数 ─────────────────
            subsets_.resize(cfg.n_subset);
            subset_params_.resize(cfg.n_subset);

            size_t max_K = 0;
            for (int s = 0; s < cfg.n_subset; ++s) {
                auto& idx = subsets_[s];
                for (int i = s; i < Na; i += cfg.n_subset)
                    idx.push_back(i);
                max_K = std::max(max_K, idx.size());

                auto& ps = subset_params_[s];
                ps = params;
                ps.iPAng = (int)idx.size();
                ps.angle_list.resize(idx.size());
                for (int i = 0; i < (int)idx.size(); ++i)
                    ps.angle_list[i] = params.angle_list[idx[i]];
            }

            const size_t max_subset_sino = max_K * view_n;

            // ── 常驻缓冲 (约 2GB @512^3) ──────────────────────────────
            //   sino 类 3 个 + row 权重 1 个 = 4 * max_subset_sino  (~400MB)
            //   vol  类: d_bp_ / d_ones_vol_ / d_col_w_ = 3 * vol_n   (~1.26GB)
            YK_CUDA_CHECK(cudaMalloc(&d_sino_meas_sub_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_row_w_, max_subset_sino * sizeof(float))); // 现算 R
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));          // 现算 C

            fp_.init(params, cfg.fp_task, deviceId);
            bp_.init(params, cfg.bp_task, deviceId);

            // 全 1 体只需一次, 反复用于算每子集的行权重
            YK::Iter::fill_ones_launch(d_ones_vol_, vol_n, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            is_initialized_ = true;
            YK_LOGI("[OSSART] init OK: {} angles, {} subsets, max {}/subset; weights computed on-the-fly",
                Na, cfg.n_subset, max_K);
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

            for (unsigned int iter = 0; iter < iterations; ++iter)
            {
                const int subset_idx = iteration_ % cfg_.n_subset;
                const std::vector<int>& subset = subsets_[subset_idx];
                const int    K = (int)subset.size();
                const size_t sino_n = (size_t)K * view_n;

                const SCBCTParams& ps = subset_params_[subset_idx];

                // ── 现算本子集 R 行权重: d_row_w_ = A_s · 1_vol ─────
                YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_ones_vol_, ps, d_row_w_, stream);

                // ── 现算本子集 C 列权重: d_col_w_ = A_s^T · 1_proj ──
                //   用 d_residual_ 暂存全 1 正弦图 (随后会被残差覆盖)
                YK::Iter::fill_ones_launch(d_residual_, sino_n, stream);
                bp_.run(d_residual_, ps, d_col_w_, stream, /*clear_vol=*/true);

                // ── 收集子集测量正弦图 ────────────────────────────────
                for (int i = 0; i < K; ++i) {
                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_sino_meas_sub_ + i * view_n,
                        d_sino_meas + subset[i] * view_n,
                        view_n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }

                // ── Step1: 正投影 A_s * x ─────────────────────────────
                YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_vol, ps, d_sino_fwd_, stream);

                // ── Step2: 残差 r = b_s - A_s x ──────────────────────
                YK::Iter::residual_launch(
                    d_sino_meas_sub_, d_sino_fwd_,
                    d_residual_, sino_n, stream);

                // ── Step2.5: R 行归一化  r /= (A_s·1_vol + eps) ─────
                YK::Iter::divide_launch(
                    d_residual_, d_row_w_, cfg_.eps, sino_n, stream);

                // ── Step3: 反投影 bp = A_s^T * r ─────────────────────
                bp_.run(d_residual_, ps, d_bp_, stream, /*clear_vol=*/true);

                // ── Step4: 更新 x += lambda * bp / (A_s^T·1 + eps) ──
                YK::Iter::update_launch(
                    d_vol, d_bp_, d_col_w_,
                    lambda_cur_, cfg_.eps,
                    vol_n, stream);

                lambda_cur_ *= cfg_.lambda_red;

                // ── Step5: 约束 ───────────────────────────────────────
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

                if (iteration_ == 0) diagnose_(d_vol, sino_n, vol_n, stream);

                iteration_++;
                YK_LOGD("[OSSART] iter={} subset={}/{} K={} lambda={:.4e}",
                    iteration_, subset_idx + 1, cfg_.n_subset, K, lambda_cur_);
            }
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t       stream)
        {
            if (!is_initialized_) return false;
            return iterate(d_sino_meas, d_vol, params, stream,
                cfg_.n_iter * cfg_.n_subset);
        }

        unsigned int totalIterations() const { return iteration_; }

        void reset() { iteration_ = 0; lambda_cur_ = cfg_.lambda; }

        void release()
        {
            if (d_sino_meas_sub_) { cudaFree(d_sino_meas_sub_); d_sino_meas_sub_ = nullptr; }
            if (d_sino_fwd_) { cudaFree(d_sino_fwd_); d_sino_fwd_ = nullptr; }
            if (d_residual_) { cudaFree(d_residual_); d_residual_ = nullptr; }
            if (d_row_w_) { cudaFree(d_row_w_); d_row_w_ = nullptr; }
            if (d_bp_) { cudaFree(d_bp_); d_bp_ = nullptr; }
            if (d_ones_vol_) { cudaFree(d_ones_vol_); d_ones_vol_ = nullptr; }
            if (d_col_w_) { cudaFree(d_col_w_); d_col_w_ = nullptr; }

            subsets_.clear();
            subset_params_.clear();

            fp_.release();
            bp_.release();
            is_initialized_ = false;
            iteration_ = 0;
            lambda_cur_ = 1.0f;
        }

        ~OSSART() { release(); }

    private:
        void diagnose_(const float* d_vol, size_t sino_n, size_t vol_n, cudaStream_t stream)
        {
            cudaStreamSynchronize(stream);
            std::vector<float> h_tmp(sino_n), h_vol(vol_n);

            cudaMemcpy(h_tmp.data(), d_row_w_, sino_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] row_w (A_s.1_vol): min={:.4e} max={:.4e}",
                *std::min_element(h_tmp.begin(), h_tmp.end()),
                *std::max_element(h_tmp.begin(), h_tmp.end()));

            cudaMemcpy(h_vol.data(), d_col_w_, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] col_w (A_s^T.1_proj): min={:.4e} max={:.4e}",
                *std::min_element(h_vol.begin(), h_vol.end()),
                *std::max_element(h_vol.begin(), h_vol.end()));

            cudaMemcpy(h_tmp.data(), d_sino_meas_sub_, sino_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] sino_meas_sub: min={:.4e} max={:.4e}",
                *std::min_element(h_tmp.begin(), h_tmp.end()),
                *std::max_element(h_tmp.begin(), h_tmp.end()));

            cudaMemcpy(h_tmp.data(), d_sino_fwd_, sino_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] sino_fwd: min={:.4e} max={:.4e}",
                *std::min_element(h_tmp.begin(), h_tmp.end()),
                *std::max_element(h_tmp.begin(), h_tmp.end()));

            cudaMemcpy(h_tmp.data(), d_residual_, sino_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] residual (after R-norm): min={:.4e} max={:.4e}",
                *std::min_element(h_tmp.begin(), h_tmp.end()),
                *std::max_element(h_tmp.begin(), h_tmp.end()));

            cudaMemcpy(h_vol.data(), d_bp_, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] bp: min={:.4e} max={:.4e}",
                *std::min_element(h_vol.begin(), h_vol.end()),
                *std::max_element(h_vol.begin(), h_vol.end()));

            cudaMemcpy(h_vol.data(), d_vol, vol_n * sizeof(float), cudaMemcpyDeviceToHost);
            YK_LOGI("[diag] vol after 1st update: min={:.4e} max={:.4e}",
                *std::min_element(h_vol.begin(), h_vol.end()),
                *std::max_element(h_vol.begin(), h_vol.end()));
        }

        bool         is_initialized_ = false;
        unsigned int iteration_ = 0;
        float        lambda_cur_ = 1.0f;
        SCBCTParams  params_;
        Config       cfg_;

        ConeProjector     fp_;
        ConeBackprojector bp_;

        std::vector<std::vector<int>> subsets_;
        std::vector<SCBCTParams>      subset_params_;

        float* d_sino_meas_sub_ = nullptr;
        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_row_w_ = nullptr;     // 现算: A_s · 1_vol
        float* d_bp_ = nullptr;
        float* d_ones_vol_ = nullptr;  // 常驻全 1 体
        float* d_col_w_ = nullptr;     // 现算: A_s^T · 1_proj
    };

    // ── 便捷函数 ────────────────────────────────────────────────────
    YK_INLINE bool ossart_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t       stream,
        OSSART::Config     cfg = {})
    {
        OSSART recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }

} // namespace YK