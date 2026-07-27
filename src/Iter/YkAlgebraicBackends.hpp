// YkOSSART.hpp
#pragma once
#include "common/YkProjectionOperators.hpp"
#include "kernels/YkIterLaunch.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkLog.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"

#include <cuda_runtime.h>
#include <global/YkCBCTParams.h>
#include <vector>
#include <numeric>



// ============================================================================
// OSSART_TIGRE : 对齐 TIGRE OS_SART 语义;Config / 公共接口与 OSSART_v0 完全一致
//
// Config 字段沿用 v0 语义映射:
//   n_subset   → 块数。内部换算 blocksize = ceil(Na / n_subset),
//                连续分块 (= TIGRE order_subsets 'ordered'),按顺序遍历
//   n_iter     → 外层迭代数;run() = 扫 n_iter 轮全部块
//   lambda_red → 每扫完一整轮所有块衰减一次(TIGRE MATLAB 语义;=1 时无影响)
//   eps        → 传入更新步分母。TIGRE 无 eps(用 V=inf),严格对齐请设 0;
//                默认 1e-6 相对 V 的量级可忽略
//   use_min    → 对齐 TIGRE 默认行为 (noneg=True) 需调用方设
//                use_min=true, min_constraint=0;每块更新后 clamp
//
// 与 v0 的内部差异:
//   - W:2x2x2 粗网格,全角度一次 FP 预计算,块取连续切片(零拷贝)
//   - V:init 预计算,BP 全 1 后沿 Z 取平均 → 每块一张 2D 图 (= .mean(axis=0))
//   - 测量数据:块连续 → 指针偏移,d_sino_meas_sub_ 移除
//
// 依赖两个新 Iter kernel(实现放 Iter kernels .cu,声明并入 Iter 头):
//   mean_z_to_2d_launch / update_v2d_launch(见文末注释)
//
// 未自动对齐:TIGRE 算子对为 Siddon FP + FDK 加权 BP,如有对应 ETask 请在
// Config 传入。体素布局假定 idx = z*(iVX*iVY) + y*iVX + x。
// ============================================================================

// ============================================================================
// OSSART_TIGRE : 对齐 TIGRE OS_SART 语义;Config / 公共接口与 OSSART_v0 完全一致
//
// Config 字段沿用 v0 语义映射:
//   n_subset   → 块数。内部换算 blocksize = ceil(Na / n_subset),
//                连续分块 (= TIGRE order_subsets 'ordered'),按顺序遍历
//   n_iter     → 外层迭代数;run() = 扫 n_iter 轮全部块
//   lambda_red → 每扫完一整轮所有块衰减一次(TIGRE MATLAB 语义;=1 时无影响)
//   eps        → 传入更新步分母。TIGRE 无 eps(用 V=inf),严格对齐请设 0;
//                默认 1e-6 相对 V 的量级可忽略
//   use_min    → 对齐 TIGRE 默认行为 (noneg=True) 需调用方设
//                use_min=true, min_constraint=0;每块更新后 clamp
//
// 与 v0 的内部差异:
//   - W:2x2x2 粗网格,全角度一次 FP 预计算,块取连续切片(零拷贝)
//   - V:init 预计算,BP 全 1 后沿 Z 取平均 → 每块一张 2D 图 (= .mean(axis=0))
//   - 测量数据:块连续 → 指针偏移,d_sino_meas_sub_ 移除
//
// 依赖两个新 Iter kernel(实现放 Iter kernels .cu,声明并入 Iter 头):
//   mean_z_to_2d_launch / update_v2d_launch(见文末注释)
//
// 未自动对齐:TIGRE 算子对为 Siddon FP + FDK 加权 BP,如有对应 ETask 请在
// Config 传入。体素布局假定 idx = z*(iVX*iVY) + y*iVX + x。
// ============================================================================

namespace YK {

    class AlgebraicTigreBackend {
    public:
        struct Config {
            int   n_iter = 10;
            int   n_subset = 5;
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
            std::vector<SConeProjGeomVec> geometry;
            detail::buildCircularViews(params, geometry);
            return init(params, cfg, geometry, stream, deviceId);
        }

        bool init(const SCBCTParams& params, const Config& cfg,
            const std::vector<SConeProjGeomVec>& geometry,
            cudaStream_t stream, int deviceId = 0)
        {
            if (static_cast<int>(geometry.size()) != params.iPAng) {
                YK_LOGE("[AlgebraicTigreBackend] geometry size mismatch");
                return false;
            }
            params_ = params;
            cfg_ = cfg;
            iteration_ = 0;
            lambda_cur_ = cfg.lambda;

            const int    Na = params.iPAng;
            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t slice_n = (size_t)params.iVX * params.iVY;
            const size_t view_n = (size_t)params.iPU * params.iPV;

            // ── 连续分块 (= TIGRE 'ordered'):blocksize = ceil(Na/n_subset) ──
            const int blocksize = (Na + cfg.n_subset - 1) / cfg.n_subset;
            n_block_ = (Na + blocksize - 1) / blocksize;

            block_start_.resize(n_block_);
            block_size_.resize(n_block_);
            block_params_.resize(n_block_);

            size_t max_K = 0;
            for (int b = 0; b < n_block_; ++b) {
                const int start = b * blocksize;
                const int K = std::min(blocksize, Na - start);
                block_start_[b] = start;
                block_size_[b] = K;
                max_K = std::max(max_K, (size_t)K);

                auto& ps = block_params_[b];
                ps = params;
                ps.iPAng = K;
                ps.angle_list.assign(params.angle_list.begin() + start,
                    params.angle_list.begin() + start + K);
            }

            const size_t max_sino = max_K * view_n;

            // ── 常驻缓冲(块连续 → 无需 d_sino_meas_sub_)────────────────
            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));

            fp_.init(params, geometry, cfg.fp_task, deviceId, stream);
            bp_.init(params, geometry, cfg.bp_task, deviceId, stream);

            // W is projected from a 2x2x2 temporary volume.  Keep a distinct
            // FP operator so texture dimensions always match that buffer.
            const float sVolX = params.iVX * params.vox_x_mm;
            const float sVolY = params.iVY * params.vox_y_mm;
            const float sVolZ = params.iVZ * params.vox_z_mm;
            const float sDetZ = params.iPV * params.dv_mm;
            params_lo_ = params;
            params_lo_.iVX = 2; params_lo_.iVY = 2; params_lo_.iVZ = 2;
            params_lo_.vox_x_mm = sVolX * 1.1f / 2.f;
            params_lo_.vox_y_mm = sVolY * 1.1f / 2.f;
            params_lo_.vox_z_mm = std::max(sDetZ, sVolZ) / 2.f;
            fp_lo_.init(params_lo_, geometry, cfg.fp_task, deviceId, stream);

            // ── W:全角度一次预计算 (= TIGRE set_w) ─────────────────────
            // 粗网格 2x2x2;x/y 扩 1.1;z = max(探测器高, 体积高),不扩
            {
                const float sVolX = params.iVX * params.vox_x_mm;
                const float sVolY = params.iVY * params.vox_y_mm;
                const float sVolZ = params.iVZ * params.vox_z_mm;
                const float sDetZ = params.iPV * params.dv_mm;

                SCBCTParams ps_w = params_lo_;        // 全角度列表

                const size_t w_n = (size_t)Na * view_n;
                YK_CUDA_CHECK(cudaMalloc(&d_row_w_full_, w_n * sizeof(float)));
                YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_full_, 0,
                    w_n * sizeof(float), stream));

                float* d_ones8 = nullptr;
                YK_CUDA_CHECK(cudaMalloc(&d_ones8, 8 * sizeof(float)));
                YK::Iter::fill_ones_launch(d_ones8, 8, stream);

                fp_lo_.run(d_ones8, ps_w, d_row_w_full_, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                YK_CUDA_CHECK(cudaFree(d_ones8));

                // W[W <= min(真实体素)/2] = inf; W = 1/W (inf → 0)
                const float real_min_vox = std::min({
                    params.vox_x_mm, params.vox_y_mm, params.vox_z_mm });
                YK::Iter::threshold_inf_launch(
                    d_row_w_full_, w_n, real_min_vox / 2.f, stream);
                YK::Iter::rcp_launch(d_row_w_full_, w_n, stream);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // ── V:每块预计算 2D 图 (= TIGRE set_v) ─────────────────────
            // 体积三轴统一缩 max(sx,sy)/hypot(sx,sy)*0.9,BP 全 1,
            // 沿 Z 取平均 → 2D,零值 → inf
            d_col_w_all_.resize(n_block_, nullptr);
            {
                const float sVolX = params.iVX * params.vox_x_mm;
                const float sVolY = params.iVY * params.vox_y_mm;
                const float norm_xy = std::sqrt(sVolX * sVolX + sVolY * sVolY);
                const float scale = (norm_xy > 1e-8f)
                    ? std::max(sVolX, sVolY) / norm_xy * 0.9f
                    : 0.9f;

                for (int b = 0; b < n_block_; ++b) {
                    const int    K = block_size_[b];
                    const size_t sino_n = (size_t)K * view_n;

                    SCBCTParams ps_c = block_params_[b];
                    ps_c.vox_x_mm *= scale;
                    ps_c.vox_y_mm *= scale;
                    ps_c.vox_z_mm *= scale;

                    YK::Iter::fill_ones_launch(d_residual_, sino_n, stream);
                    bp_.run(d_residual_, ps_c, d_bp_, stream, /*clear_vol=*/true);

                    YK_CUDA_CHECK(cudaMalloc(&d_col_w_all_[b],
                        slice_n * sizeof(float)));
                    YK::Iter::mean_z_to_2d_launch(
                        d_bp_, d_col_w_all_[b],
                        params.iVX, params.iVY, params.iVZ, stream);

                    // V[V == 0] = inf
                    YK::Iter::threshold_inf_launch(
                        d_col_w_all_[b], slice_n, 0.f, stream);
                }
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            is_initialized_ = true;
            YK_LOGI("[OSSART-TIGRE] init OK: {} angles, {} blocks (blocksize={})",
                Na, n_block_, blocksize);
            return true;
        }

        // iterations = 子集级迭代数(与 v0 语义一致,每次处理一个块)
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
                // ── 按输入顺序取块 (= TIGRE 'ordered' 遍历) ───────────────
                const int b = iteration_ % n_block_;

                const int    start = block_start_[b];
                const int    K = block_size_[b];
                const size_t sino_n = (size_t)K * view_n;
                const SCBCTParams& ps = block_params_[b];

                // 连续切片:测量与 W 直接指针偏移,零拷贝
                const float* d_meas_blk = d_sino_meas + (size_t)start * view_n;
                const float* d_w_blk = d_row_w_full_ + (size_t)start * view_n;

                // Step1: A_b x
                YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0,
                    sino_n * sizeof(float), stream));
                fp_.run(d_vol, ps, d_sino_fwd_, stream);

                // Step2: r = b_b - A_b x
                YK::Iter::residual_launch(
                    d_meas_blk, d_sino_fwd_, d_residual_, sino_n, stream);

                // Step2.5: r = W .* r
                YK::Iter::multiply_launch(
                    d_residual_, d_w_blk, sino_n, stream);

                // Step3: bp = A_b^T r
                bp_.run(d_residual_, ps, d_bp_, stream, /*clear_vol=*/true);

                // Step4: x += lambda * bp / (V_b + eps)  (V 为 2D,沿 Z 广播)
                YK::Iter::update_v2d_launch(
                    d_vol, d_bp_, d_col_w_all_[b],
                    lambda_cur_, cfg_.eps,
                    params.iVX, params.iVY, params.iVZ, stream);

                // Step5: 约束(每块更新后;= TIGRE noneg 位置)
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

                iteration_++;

                // lambda 衰减:每扫完一整轮所有块执行一次(TIGRE MATLAB 语义)
                if (iteration_ % n_block_ == 0)
                    lambda_cur_ *= cfg_.lambda_red;

                YK_LOGD("[OSSART-TIGRE] iter={} block={} K={} lambda={:.4e}",
                    iteration_, b, K, lambda_cur_);
            }
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            const SCBCTParams& params,
            cudaStream_t       stream)
        {
            if (!is_initialized_) return false;
            // n_iter 轮完整扫描(n_block_ 通常 == n_subset,
            // Na 不整除时可能更少,以实际块数为准)
            return iterate(d_sino_meas, d_vol, params, stream,
                cfg_.n_iter * n_block_);
        }

        unsigned int totalIterations() const { return iteration_; }
        int actualSubsetCount() const { return n_block_; }

        void reset() { iteration_ = 0; lambda_cur_ = cfg_.lambda; }

        void release()
        {
            if (d_sino_fwd_) { cudaFree(d_sino_fwd_);   d_sino_fwd_ = nullptr; }
            if (d_residual_) { cudaFree(d_residual_);   d_residual_ = nullptr; }
            if (d_bp_) { cudaFree(d_bp_);         d_bp_ = nullptr; }
            if (d_row_w_full_) { cudaFree(d_row_w_full_); d_row_w_full_ = nullptr; }

            for (auto& p : d_col_w_all_)
                if (p) { cudaFree(p); p = nullptr; }
            d_col_w_all_.clear();

            block_start_.clear();
            block_size_.clear();
            block_params_.clear();

            fp_.release();
            fp_lo_.release();
            bp_.release();
            is_initialized_ = false;
            iteration_ = 0;
            lambda_cur_ = 1.0f;
        }

        ~AlgebraicTigreBackend() { release(); }

    private:
        bool         is_initialized_ = false;
        unsigned int iteration_ = 0;
        float        lambda_cur_ = 1.0f;
        SCBCTParams  params_;
        SCBCTParams  params_lo_;
        Config       cfg_;

        ForwardOperatorAdapter fp_;
        ForwardOperatorAdapter fp_lo_;
        BackOperatorAdapter bp_;

        int                      n_block_ = 0;
        std::vector<int>         block_start_;
        std::vector<int>         block_size_;
        std::vector<SCBCTParams> block_params_;

        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_bp_ = nullptr;
        float* d_row_w_full_ = nullptr;   // 全角度 W,块用连续切片

        std::vector<float*> d_col_w_all_; // 每块一张 2D V 图 (iVX*iVY)
    };


    YK_INLINE bool algebraic_tigre_backend_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t       stream,
        AlgebraicTigreBackend::Config cfg = {})
    {
        AlgebraicTigreBackend recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }

} // namespace YK

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
   // ================================================================
// OS-SART，含 R 行归一化 + C 列归一化 + 黄金角子集遍历顺序。
//
// 子集内：交错采样（角度均匀散布全程）。
// 子集间：黄金角顺序遍历，相邻迭代角度尽量正交，收敛更快更稳。
//
// 更新公式 (对每个子集 s):
//   r   = b_s - A_s x
//   r  /= (A_s · 1_vol  + eps)                  <-- R 行归一化
//   bp  = A_s^T r
//   x  += lambda * bp / (A_s^T · 1_proj + eps)  <-- C 列归一化
// ================================================================
    class AlgebraicGoldenSubsetBackend {
    public:
        struct Config {
            int   n_iter = 10;
            int   n_subset = 5;
            float lambda = 1.0f;
            float lambda_red = 1.0f;
            float eps = 1e-6f;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            // FP/BP 应为匹配对：Joseph FP ↔ Joseph_v2 BP（推荐）。
            // 不要用 BP_FDK（仅适合单遍 FDK，进迭代会产生棋盘格/星状伪影）。
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_FDK_matched;
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

            // ── 生成子集遍历顺序：黄金角步进，相邻迭代角度尽量正交 ──
            subset_order_.resize(cfg.n_subset);
            {
                const int n = cfg.n_subset;
                if (n <= 1) {
                    subset_order_[0] = 0;
                }
                else {
                    int step = (int)std::round((double)n / 1.6180339887498949);
                    if (step < 1) step = 1;
                    while (std::gcd(step, n) != 1) {   // 与 n 互质才能遍历全部子集
                        ++step;
                        if (step >= n) { step = 1; break; }
                    }
                    int cur = 0;
                    for (int k = 0; k < n; ++k) {
                        subset_order_[k] = cur;
                        cur = (cur + step) % n;
                    }
                }
                std::string ord;
                for (int k = 0; k < cfg.n_subset; ++k)
                    ord += std::to_string(subset_order_[k]) + " ";
                YK_LOGI("[OSSART] subset order (golden-angle): {}", ord);
            }

            const size_t max_subset_sino = max_K * view_n;

            // ── 常驻缓冲 ──────────────────────────────────────────────
            YK_CUDA_CHECK(cudaMalloc(&d_sino_meas_sub_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_row_w_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

            fp_.init(params, cfg.fp_task, deviceId, stream);
            bp_.init(params, cfg.bp_task, deviceId, stream);

            YK::Iter::fill_ones_launch(d_ones_vol_, vol_n, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            is_initialized_ = true;
            YK_LOGI("[OSSART] init OK: {} angles, {} subsets, max {}/subset",
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
                // ── 按黄金角顺序取子集 ──
                const int order_pos = iteration_ % cfg_.n_subset;
                const int subset_idx = subset_order_[order_pos];

                const std::vector<int>& subset = subsets_[subset_idx];
                const int    K = (int)subset.size();
                const size_t sino_n = (size_t)K * view_n;

                const SCBCTParams& ps = subset_params_[subset_idx];

                // ── 现算 R: d_row_w_ = A_s · 1_vol ──
                YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_ones_vol_, ps, d_row_w_, stream);

                // ── 现算 C: d_col_w_ = A_s^T · 1_proj ──
                YK::Iter::fill_ones_launch(d_residual_, sino_n, stream);
                bp_.run(d_residual_, ps, d_col_w_, stream, /*clear_vol=*/true);

                // ── 收集子集测量 ──
                for (int i = 0; i < K; ++i) {
                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_sino_meas_sub_ + i * view_n,
                        d_sino_meas + subset[i] * view_n,
                        view_n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }

                // Step1: A_s x
                YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_vol, ps, d_sino_fwd_, stream);

                // Step2: r = b_s - A_s x
                YK::Iter::residual_launch(
                    d_sino_meas_sub_, d_sino_fwd_,
                    d_residual_, sino_n, stream);

                // Step2.5: R 行归一化
                YK::Iter::divide_launch(
                    d_residual_, d_row_w_, cfg_.eps, sino_n, stream);

                // Step3: bp = A_s^T r
                bp_.run(d_residual_, ps, d_bp_, stream, /*clear_vol=*/true);

                // Step4: x += lambda * bp / (C + eps)
                YK::Iter::update_launch(
                    d_vol, d_bp_, d_col_w_,
                    lambda_cur_, cfg_.eps,
                    vol_n, stream);

                lambda_cur_ *= cfg_.lambda_red;

                // Step5: 约束
                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

                iteration_++;
                YK_LOGD("[OSSART] iter={} order_pos={} subset={} K={} lambda={:.4e}",
                    iteration_, order_pos, subset_idx, K, lambda_cur_);
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
            subset_order_.clear();

            fp_.release();
            bp_.release();
            is_initialized_ = false;
            iteration_ = 0;
            lambda_cur_ = 1.0f;
        }

        ~AlgebraicGoldenSubsetBackend() { release(); }

    private:
        bool         is_initialized_ = false;
        unsigned int iteration_ = 0;
        float        lambda_cur_ = 1.0f;
        SCBCTParams  params_;
        Config       cfg_;

        ForwardOperatorAdapter fp_;
        BackOperatorAdapter bp_;

        std::vector<std::vector<int>> subsets_;
        std::vector<SCBCTParams>      subset_params_;
        std::vector<int>              subset_order_;   // 黄金角遍历顺序

        float* d_sino_meas_sub_ = nullptr;
        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_row_w_ = nullptr;
        float* d_bp_ = nullptr;
        float* d_ones_vol_ = nullptr;
        float* d_col_w_ = nullptr;
    };


    // ── 便捷函数 ────────────────────────────────────────────────────
    YK_INLINE bool algebraic_golden_subset_backend_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        cudaStream_t       stream,
        AlgebraicGoldenSubsetBackend::Config cfg = {})
    {
        AlgebraicGoldenSubsetBackend recon;
        if (!recon.init(params, cfg, stream)) return false;
        return recon.run(d_sino_meas, d_vol, params, stream);
    }

} // namespace YK

namespace YK {

    class AlgebraicDetailedWeightBackend {
    public:
        struct Config {
            int   n_iter = 10;
            int   n_subset = 20;
            float lambda = 1.0f;
            float lambda_red = 1.0f;
            float eps = 1e-6f;
            bool  use_min = false;
            float min_constraint = 0.f;
            bool  use_max = false;
            float max_constraint = 1e30f;
            ETask fp_task = ETask::FP_Joseph;
            ETask bp_task = ETask::BP_Joseph_v2;
            bool golden_subset_order = false;
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
            iteration_ = 0;
            lambda_cur_ = cfg.lambda;
            deviceId_ = deviceId;

            if ((int)h_views.size() != params.iPAng) {
                YK_LOGE("[AlgebraicDetailedWeightBackend] geometry size mismatch");
                return false;
            }

            const int    Na = params.iPAng;
            const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
            const size_t view_n = (size_t)params.iPU * params.iPV;

            // 构建子集
            subsets_.resize(cfg.n_subset);
            subset_params_.resize(cfg.n_subset);
            subset_views_.resize(cfg.n_subset);
            subset_order_.resize(cfg.n_subset);

            size_t max_K = 0;
            for (int s = 0; s < cfg.n_subset; ++s) {
                auto& idx = subsets_[s];
                for (int i = s; i < Na; i += cfg.n_subset)
                    idx.push_back(i);
                max_K = std::max(max_K, idx.size());

                // 子集参数
                auto& ps = subset_params_[s];
                ps = params;
                ps.iPAng = (int)idx.size();
                ps.angle_list.resize(idx.size());
                for (int i = 0; i < (int)idx.size(); ++i)
                    ps.angle_list[i] = params.angle_list[idx[i]];

                // 子集几何
                auto& sv = subset_views_[s];
                sv.resize(idx.size());
                for (int i = 0; i < (int)idx.size(); ++i)
                    sv[i] = h_views[idx[i]];
            }

            if (!cfg.golden_subset_order || cfg.n_subset <= 1) {
                std::iota(subset_order_.begin(), subset_order_.end(), 0);
            }
            else {
                int step = static_cast<int>(std::round(
                    static_cast<double>(cfg.n_subset) / 1.6180339887498949));
                step = std::max(step, 1);
                while (std::gcd(step, cfg.n_subset) != 1) {
                    ++step;
                    if (step >= cfg.n_subset) { step = 1; break; }
                }
                int current = 0;
                for (int i = 0; i < cfg.n_subset; ++i) {
                    subset_order_[i] = current;
                    current = (current + step) % cfg.n_subset;
                }
            }

            const size_t max_subset_sino = max_K * view_n;

            YK_CUDA_CHECK(cudaMalloc(&d_sino_meas_sub_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_sino_fwd_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_residual_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_row_w_, max_subset_sino * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_, vol_n * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

            if (!fp_.init(params, h_views_, cfg.fp_task, deviceId, stream) ||
                !bp_.init(params, h_views_, cfg.bp_task, deviceId, stream)) {
                release();
                return false;
            }

            YK::Iter::fill_ones_launch(d_ones_vol_, vol_n, stream);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            is_initialized_ = true;
            YK_LOGI("[AlgebraicDetailedWeightBackend] init OK: {} angles, "
                "{} subsets, max {}/subset",
                Na, cfg.n_subset, max_K);
            return true;
        }

        bool iterate(
            const float* d_sino_meas,
            float* d_vol,
            cudaStream_t stream,
            unsigned int iterations)
        {
            if (!is_initialized_) return false;

            const size_t vol_n = (size_t)params_.iVX * params_.iVY * params_.iVZ;
            const size_t view_n = (size_t)params_.iPU * params_.iPV;

            for (unsigned int iter = 0; iter < iterations; ++iter)
            {
                const int order_pos = iteration_ % cfg_.n_subset;
                const int subset_idx = subset_order_[order_pos];
                const auto& idx = subsets_[subset_idx];
                const int   K = (int)idx.size();
                const size_t sino_n = (size_t)K * view_n;

                const SCBCTParams& ps = subset_params_[subset_idx];
                const std::vector<SConeProjGeomVec>& sv = subset_views_[subset_idx];

                // 行权重：A_s · 1_vol
                YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_ones_vol_, ps, d_row_w_, stream);

                // 列权重：A_s^T · 1_proj
                YK::Iter::fill_ones_launch(d_residual_, sino_n, stream);
                bp_.run(d_residual_, ps, d_col_w_, stream, true);

                // 收集子集正弦图
                for (int i = 0; i < K; ++i)
                    YK_CUDA_CHECK(cudaMemcpyAsync(
                        d_sino_meas_sub_ + i * view_n,
                        d_sino_meas + idx[i] * view_n,
                        view_n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));

                // 正投影
                YK_CUDA_CHECK(cudaMemsetAsync(d_sino_fwd_, 0, sino_n * sizeof(float), stream));
                fp_.run(d_vol, ps, d_sino_fwd_, stream);

                // 残差
                YK::Iter::residual_launch(
                    d_sino_meas_sub_, d_sino_fwd_, d_residual_, sino_n, stream);

                // R行归一化
                YK::Iter::divide_launch(d_residual_, d_row_w_, cfg_.eps, sino_n, stream);

                // 反投影
                bp_.run(d_residual_, ps, d_bp_, stream, true);

                // 更新
                YK::Iter::update_launch(
                    d_vol, d_bp_, d_col_w_,
                    lambda_cur_, cfg_.eps, vol_n, stream);

                lambda_cur_ *= cfg_.lambda_red;

                if (cfg_.use_min)
                    YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
                if (cfg_.use_max)
                    YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

                iteration_++;
                YK_LOGD("[AlgebraicDetailedWeightBackend] update={} subset={}/{} "
                    "K={} lambda={:.4e}",
                    iteration_, subset_idx + 1, cfg_.n_subset, K, lambda_cur_);
            }
            return true;
        }

        bool run(const float* d_sino_meas,
            float* d_vol,
            cudaStream_t stream)
        {
            if (!is_initialized_) return false;
            return iterate(d_sino_meas, d_vol, stream,
                cfg_.n_iter * cfg_.n_subset);
        }

        unsigned int totalIterations() const { return iteration_; }
        int actualSubsetCount() const { return cfg_.n_subset; }
        void reset() { iteration_ = 0; lambda_cur_ = cfg_.lambda; }

        void release()
        {
            if (d_sino_meas_sub_) { cudaFree(d_sino_meas_sub_); d_sino_meas_sub_ = nullptr; }
            if (d_sino_fwd_) { cudaFree(d_sino_fwd_);      d_sino_fwd_ = nullptr; }
            if (d_residual_) { cudaFree(d_residual_);      d_residual_ = nullptr; }
            if (d_row_w_) { cudaFree(d_row_w_);         d_row_w_ = nullptr; }
            if (d_bp_) { cudaFree(d_bp_);            d_bp_ = nullptr; }
            if (d_ones_vol_) { cudaFree(d_ones_vol_);      d_ones_vol_ = nullptr; }
            if (d_col_w_) { cudaFree(d_col_w_);         d_col_w_ = nullptr; }

            subsets_.clear();
            subset_params_.clear();
            subset_views_.clear();
            subset_order_.clear();

            fp_.release();
            bp_.release();
            is_initialized_ = false;
            iteration_ = 0;
            lambda_cur_ = 1.0f;
        }

        ~AlgebraicDetailedWeightBackend() { release(); }

    private:
        bool         is_initialized_ = false;
        unsigned int iteration_ = 0;
        float        lambda_cur_ = 1.0f;
        int          deviceId_ = 0;
        SCBCTParams  params_;
        Config       cfg_;

        std::vector<SConeProjGeomVec>        h_views_;
        std::vector<std::vector<int>>        subsets_;
        std::vector<SCBCTParams>             subset_params_;
        std::vector<std::vector<SConeProjGeomVec>> subset_views_;
        std::vector<int> subset_order_;

        // Same operator contract as circular iterative solvers; only the
        // GeometryContext construction differs.
        ForwardOperatorAdapter fp_;
        BackOperatorAdapter bp_;

        float* d_sino_meas_sub_ = nullptr;
        float* d_sino_fwd_ = nullptr;
        float* d_residual_ = nullptr;
        float* d_row_w_ = nullptr;
        float* d_bp_ = nullptr;
        float* d_ones_vol_ = nullptr;
        float* d_col_w_ = nullptr;
    };

    YK_INLINE bool algebraic_ex_backend_reconstruct(
        const float* d_sino_meas,
        float* d_vol,
        const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& h_views,
        cudaStream_t                         stream,
        AlgebraicDetailedWeightBackend::Config cfg = {})
    {
        AlgebraicDetailedWeightBackend recon;
        if (!recon.init(params, cfg, h_views, stream)) return false;
        return recon.run(d_sino_meas, d_vol, stream);
    }
} // namespace YK
