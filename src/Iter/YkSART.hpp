// YkSART.hpp
#pragma once
#include "common/YkProjectionOperators.hpp"
#include "kernel/YkIterLaunch.cuh"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkLog.h"
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
                        const float sVolX = params.iVX * params.vox_x_mm;
                        const float sVolY = params.iVY * params.vox_y_mm;
                        const float sVolZ = params.iVZ * params.vox_z_mm;
                        const float sDetZ = params.iPV * params.dv_mm;

                        SCBCTParams ps_r = p1;
                        ps_r.iVX = 2; ps_r.iVY = 2; ps_r.iVZ = 2;
                        ps_r.vox_x_mm = sVolX * 1.1f / 2.f;
                        ps_r.vox_y_mm = sVolY * 1.1f / 2.f;
                        ps_r.vox_z_mm = std::max(sDetZ, sVolZ) / 2.f;

                        float* d_ones_coarse = nullptr;
                        YK_CUDA_CHECK(cudaMalloc(&d_ones_coarse, 8 * sizeof(float)));
                        YK::Iter::fill_ones_launch(d_ones_coarse, 8, stream);

                        YK_CUDA_CHECK(cudaMemsetAsync(d_row_w_, 0,
                            view_n * sizeof(float), stream));
                        fp_.run(d_ones_coarse, ps_r, d_row_w_, stream);
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
            bp_.release();
            is_initialized_ = false;
            lambda_cur_ = 1.0f;
        }

        ~SART() { release(); }

    private:
        bool        is_initialized_ = false;
        float       lambda_cur_ = 1.0f;
        SCBCTParams params_;
        Config      cfg_;

        ForwardOperatorAdapter fp_;
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
    //// ================================================================
    //// SIRT (Simultaneous Iterative Reconstruction Technique)
    ////
    //// 等价于 OS-SART 取 n_subset=1：每次迭代用全部角度做一次更新。
    //// 与之前基于 TIGRE 2x2x2 行权重 + Z 均值列权重的近似版不同，
    //// 这里 R/C 全部用全分辨率真实算子算出，数值干净、锥束下不发散。
    ////
    //// 更新公式 (每轮，全角度)：
    ////   r   = b - A x
    ////   r  /= (A · 1_vol  + eps)                 <-- R 行归一化 (全分辨率)
    ////   bp  = Aᵀ r
    ////   x  += lambda * bp / (Aᵀ · 1_proj + eps)  <-- C 列归一化 (3D, 不压 Z)
    ////
    //// 显存策略 (sino 类分块)：
    ////   - C = Aᵀ·1_proj 与 x 无关 → init 预计算一次常驻 (3D vol 大小)。
    ////   - R = A·1_vol   与 x 无关，但整张 sino 太大不常驻
    ////         → 按 batch 现算到一个复用的 batch sino 缓冲。
    ////   常驻：d_bp_ + d_ones_vol_ + d_col_w_ = 3×vol；
    ////         迭代临时：2×(batch sino) (meas + fwd/residual/R 复用)。
    //// ================================================================
    //class SIRT {
    //public:
    //    struct Config {
    //        int   n_iter = 50;
    //        float lambda = 1.0f;
    //        float lambda_red = 1.0f;   // 1.0=不衰减；想衰减用 0.99
    //        float eps = 1e-6f;
    //        bool  use_min = false;
    //        float min_constraint = 0.f;
    //        bool  use_max = false;
    //        float max_constraint = 1e30f;
    //        int   n_batch = 4;         // 仅用于 sino 分块，不影响数值结果
    //        ETask fp_task = ETask::FP_Joseph;
    //        ETask bp_task = ETask::BP_Siddon_VoxDriven;
    //    };

    //    bool init(const SCBCTParams& params, const Config& cfg,
    //        cudaStream_t stream, int deviceId = 0)
    //    {
    //        params_ = params;
    //        cfg_ = cfg;
    //        lambda_cur_ = cfg.lambda;
    //        deviceId_ = deviceId;

    //        const int    Na = params.iPAng;
    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t view_n = (size_t)params.iPU * params.iPV;
    //        const int    batch_Na = (Na + cfg.n_batch - 1) / cfg.n_batch;
    //        const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

    //        // ── 预切分 batch（纯显存分块，连续切分即可，SIRT 单次更新与顺序无关）──
    //        batches_.resize(cfg.n_batch);
    //        for (int b = 0; b < cfg.n_batch; ++b) {
    //            auto& bm = batches_[b];
    //            bm.start_angle = b * batch_Na;
    //            const int end = std::min(bm.start_angle + batch_Na, Na);
    //            bm.K = end - bm.start_angle;
    //            bm.sino_n = (size_t)bm.K * view_n;
    //            bm.pb = params;
    //            bm.pb.iPAng = bm.K;
    //            bm.pb.angle_list = std::vector<float>(
    //                params.angle_list.begin() + bm.start_angle,
    //                params.angle_list.begin() + end);
    //        }

    //        // ── 常驻缓冲 ──
    //        YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_, vol_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

    //        fp_.init(params, cfg.fp_task, deviceId, stream);
    //        bp_.init(params, cfg.bp_task, deviceId, stream);

    //        // ── 预计算 C 列权重：d_col_w_ = Aᵀ·1_proj（全角度，3D，不压 Z）──
    //        //   分块累加：每块对一个全 1 batch sino 做 BP，累加到 d_col_w_。
    //        {
    //            float* d_ones_sino = nullptr;
    //            YK_CUDA_CHECK(cudaMalloc(&d_ones_sino, max_batch_sino_n * sizeof(float)));
    //            for (int b = 0; b < cfg.n_batch; ++b) {
    //                const auto& bm = batches_[b];
    //                YK::Iter::fill_ones_launch(d_ones_sino, bm.sino_n, stream);
    //                // 第一块清零体积，其余累加
    //                bp_.run(d_ones_sino, bm.pb, d_col_w_, stream, /*clear_vol=*/(b == 0));
    //            }
    //            cudaFree(d_ones_sino);
    //        }
    //        // d_col_w_ 保持原始 Aᵀ·1（不取倒数），更新时用 bp/(col_w+eps)

    //        // 全 1 体常驻，迭代里反复用于现算 R 行权重
    //        YK::Iter::fill_ones_launch(d_ones_vol_, vol_n, stream);

    //        YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //        is_initialized_ = true;
    //        YK_LOGI("[SIRT] init OK: {} angles, {} batches; C precomputed, R on-the-fly",
    //            Na, cfg.n_batch);
    //        return true;
    //    }

    //    bool iterate(
    //        const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t stream,
    //        unsigned int iterations)
    //    {
    //        if (!is_initialized_) return false;

    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t view_n = (size_t)params.iPU * params.iPV;
    //        const int    Na = params.iPAng;
    //        const int    batch_Na = (Na + cfg_.n_batch - 1) / cfg_.n_batch;
    //        const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

    //        float* d_sino_batch = nullptr;   // 子集测量 / 残差
    //        float* d_work_batch = nullptr;   // fwd / R 行权重复用
    //        YK_CUDA_CHECK(cudaMalloc(&d_sino_batch, max_batch_sino_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_work_batch, max_batch_sino_n * sizeof(float)));

    //        struct Guard {
    //            float** a; float** b;
    //            ~Guard() { cudaFree(*a); cudaFree(*b); }
    //        } guard{ &d_sino_batch, &d_work_batch };

    //        for (unsigned int iter = 0; iter < iterations; ++iter)
    //        {
    //            std::string str = fmt::format("[SIRT] iter {}/{}", iter + 1, iterations);
    //            Util::CpuTimer timer(str.c_str());

    //            // ── 全角度累加反投影 d_bp_ = Aᵀ ( R ⊙ (b - A x) ) ──
    //            for (int b = 0; b < cfg_.n_batch; ++b)
    //            {
    //                const auto& bm = batches_[b];

    //                // 取本块测量
    //                YK_CUDA_CHECK(cudaMemcpyAsync(
    //                    d_sino_batch,
    //                    d_sino_meas + bm.start_angle * view_n,
    //                    bm.sino_n * sizeof(float),
    //                    cudaMemcpyDeviceToDevice, stream));

    //                // Step1: 正投影 A_b x
    //                YK_CUDA_CHECK(cudaMemsetAsync(
    //                    d_work_batch, 0, bm.sino_n * sizeof(float), stream));
    //                fp_.run(d_vol, bm.pb, d_work_batch, stream);

    //                // Step2: 残差 r = b - A_b x （写回 d_sino_batch）
    //                YK::Iter::residual_launch(
    //                    d_sino_batch, d_work_batch,
    //                    d_sino_batch, bm.sino_n, stream);

    //                // Step3: 现算本块 R 行权重 d_work_batch = A_b·1_vol
    //                YK_CUDA_CHECK(cudaMemsetAsync(
    //                    d_work_batch, 0, bm.sino_n * sizeof(float), stream));
    //                fp_.run(d_ones_vol_, bm.pb, d_work_batch, stream);

    //                // Step4: R 行归一化 r /= (A_b·1_vol + eps)
    //                YK::Iter::divide_launch(
    //                    d_sino_batch, d_work_batch, cfg_.eps, bm.sino_n, stream);

    //                // Step5: 反投影累加 d_bp_ += A_bᵀ r （首块清零）
    //                bp_.run(d_sino_batch, bm.pb, d_bp_, stream, /*clear_vol=*/(b == 0));
    //            }

    //            // ── Step6: 列归一化更新 x += lambda * bp / (Aᵀ·1 + eps) ──
    //            YK::Iter::update_launch(
    //                d_vol, d_bp_, d_col_w_,
    //                lambda_cur_, cfg_.eps,
    //                vol_n, stream);

    //            lambda_cur_ *= cfg_.lambda_red;

    //            if (cfg_.use_min)
    //                YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
    //            if (cfg_.use_max)
    //                YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

    //            if ((iter + 1) % 5 == 0 || iter + 1 == iterations) {
    //                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //                const size_t cx =
    //                    (size_t)(params.iVZ / 2) * params.iVY * params.iVX
    //                    + (size_t)(params.iVY / 2) * params.iVX
    //                    + params.iVX / 2;
    //                float h_center = 0.f;
    //                cudaMemcpy(&h_center, d_vol + cx, sizeof(float), cudaMemcpyDeviceToHost);
    //                YK_LOGI("[SIRT] iter {}/{} vol_center={:.6e} lambda={:.4e}",
    //                    iter + 1, iterations, h_center, lambda_cur_);
    //            }
    //        }
    //        return true;
    //    }

    //    bool run(const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t stream)
    //    {
    //        if (!is_initialized_) return false;
    //        return iterate(d_sino_meas, d_vol, params, stream, cfg_.n_iter);
    //    }

    //    void reset() { lambda_cur_ = cfg_.lambda; }

    //    void release()
    //    {
    //        if (d_bp_) { cudaFree(d_bp_);        d_bp_ = nullptr; }
    //        if (d_ones_vol_) { cudaFree(d_ones_vol_);  d_ones_vol_ = nullptr; }
    //        if (d_col_w_) { cudaFree(d_col_w_);     d_col_w_ = nullptr; }

    //        batches_.clear();
    //        fp_.release();
    //        bp_.release();
    //        is_initialized_ = false;
    //        lambda_cur_ = 1.0f;
    //    }

    //    ~SIRT() { release(); }

    //private:
    //    struct BatchMeta {
    //        SCBCTParams pb;
    //        int         K;
    //        size_t      sino_n;
    //        int         start_angle;
    //    };

    //    bool         is_initialized_ = false;
    //    float        lambda_cur_ = 1.0f;
    //    int          deviceId_ = 0;
    //    SCBCTParams  params_;
    //    Config       cfg_;

    //    ForwardOperatorAdapter fp_;
    //    BackOperatorAdapter bp_;

    //    std::vector<BatchMeta> batches_;

    //    float* d_bp_ = nullptr;        // Aᵀ(R⊙r) 累加
    //    float* d_ones_vol_ = nullptr;  // 常驻全 1 体，用于现算 R
    //    float* d_col_w_ = nullptr;     // 预计算 C = Aᵀ·1_proj (3D)
    //};

    //YK_INLINE bool sirt_reconstruct(
    //    const float* d_sino_meas,
    //    float* d_vol,
    //    const SCBCTParams& params,
    //    cudaStream_t stream,
    //    SIRT::Config cfg = {})
    //{
    //    SIRT recon;
    //    if (!recon.init(params, cfg, stream)) return false;
    //    return recon.run(d_sino_meas, d_vol, params, stream);
    //}


        // ================================================================
        // SIRT — 8GB 显存友好版
        //
        //   C 列权重 = Aᵀ·1_proj : 全分辨率 3D，init 预计算一次常驻。
        //                          (轴向均匀性靠它，必须全分辨率、不压 Z)
        //   R 行权重 = A·1_vol    : 与 x 无关的射线穿越长度，对体素分辨率不敏感
        //                          → 用低分辨率全 1 体现算 (FOV 不变，体素更大)。
        //                          低分辨率体仅几 MB，FP 成本接近忽略。
        //
        // 与之前崩掉的近似版区别：
        //   - 旧版同时叠了 [2x2x2 R] + [Z均值2D C] + [×1.1/×0.9 缩放] 三个近似。
        //   - 本版 C 完全精确 (3D全分辨率)，仅 R 用适度低分辨率，且 FOV 严格对齐，
        //     无缩放魔法系数。数值干净，锥束轴向不发散。
        //
        // 更新 (每轮，全角度)：
        //   r  = b - A x
        //   r /= (A·1_vol_lo + eps)                  <-- R (低分辨率体现算)
        //   x += lambda * Aᵀr / (Aᵀ·1_proj + eps)    <-- C (全分辨率 3D)
        //
        // 显存 (8GB)：常驻 d_bp_ + d_col_w_ = 2×vol (~0.84GB)
        //            + d_ones_vol_lo_ (几MB) ；
        //            迭代临时 2×batch_sino，靠加大 n_batch 压小。
        //            R 不整张常驻 (那要 ~2GB，8GB 放不下)。
        // ================================================================

    static inline void yk_dump_raw(const float* d_ptr, size_t n, const char* path)
    {
        std::vector<float> h(n);
        YK_CUDA_CHECK(cudaMemcpy(h.data(), d_ptr, n * sizeof(float),
            cudaMemcpyDeviceToHost));
        FILE* fp = std::fopen(path, "wb");
        if (!fp) { YK_LOGE("[dump] fopen fail: {}", path); return; }
        std::fwrite(h.data(), sizeof(float), n, fp);
        std::fclose(fp);

        float vmin = h[0], vmax = h[0];
        size_t nzero = 0;
        for (float v : h) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
            if (v == 0.f) ++nzero;
        }
        YK_LOGI("[dump] {} : n={} min={:.6e} max={:.6e} zeros={} ({:.2f}%)",
            path, n, vmin, vmax, nzero, 100.0 * nzero / n);
    }
    //class SIRT {
    //public:
    //    struct Config {
    //        int   n_iter = 50;
    //        float lambda = 1.0f;
    //        float lambda_red = 1.0f;
    //        float eps = 1e-6f;
    //        bool  use_min = false;
    //        float min_constraint = 0.f;
    //        bool  use_max = false;
    //        float max_constraint = 1e30f;
    //        int   n_batch = 8;
    //        int   row_w_down = 2;       // R 体降采样倍数 (FOV 不变)
    //        bool  dump_debug = false;   // ← 打开导出权重 raw
    //        ETask fp_task = ETask::FP_Joseph;
    //        ETask bp_task = ETask::BP_Joseph_v3;
    //    };

    //    bool init(const SCBCTParams& params, const Config& cfg,
    //        cudaStream_t stream, int deviceId = 0)
    //    {
    //        params_ = params;
    //        cfg_ = cfg;
    //        lambda_cur_ = cfg.lambda;
    //        deviceId_ = deviceId;

    //        const int    Na = params.iPAng;
    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t view_n = (size_t)params.iPU * params.iPV;
    //        const int    batch_Na = (Na + cfg.n_batch - 1) / cfg.n_batch;
    //        const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

    //        // ── 预切分 batch (连续切分) ──
    //        batches_.resize(cfg.n_batch);
    //        for (int b = 0; b < cfg.n_batch; ++b) {
    //            auto& bm = batches_[b];
    //            bm.start_angle = b * batch_Na;
    //            const int end = std::min(bm.start_angle + batch_Na, Na);
    //            bm.K = end - bm.start_angle;
    //            bm.sino_n = (size_t)bm.K * view_n;
    //            bm.pb = params;
    //            bm.pb.iPAng = bm.K;
    //            bm.pb.angle_list = std::vector<float>(
    //                params.angle_list.begin() + bm.start_angle,
    //                params.angle_list.begin() + end);
    //        }

    //        // ── 常驻缓冲 (2×vol) ──
    //        YK_CUDA_CHECK(cudaMalloc(&d_bp_, vol_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_col_w_, vol_n * sizeof(float)));

    //        fp_.init(params, cfg.fp_task, deviceId, stream);
    //        bp_.init(params, cfg.bp_task, deviceId, stream);

    //        // ── 预计算 C = Aᵀ·1_proj (全分辨率 3D，分块累加) ──
    //        {
    //            float* d_ones_sino = nullptr;
    //            YK_CUDA_CHECK(cudaMalloc(&d_ones_sino, max_batch_sino_n * sizeof(float)));
    //            for (int b = 0; b < cfg.n_batch; ++b) {
    //                const auto& bm = batches_[b];
    //                YK::Iter::fill_ones_launch(d_ones_sino, bm.sino_n, stream);
    //                bp_.run(d_ones_sino, bm.pb, d_col_w_, stream, /*clear_vol=*/(b == 0));
    //            }
    //            cudaFree(d_ones_sino);
    //        }

    //        // ── 构建低分辨率 R 体：体素数降采样，FOV 严格不变 ──
    //        const int dn = std::max(1, cfg.row_w_down);
    //        params_lo_ = params;
    //        params_lo_.iVX = std::max(1, params.iVX / dn);
    //        params_lo_.iVY = std::max(1, params.iVY / dn);
    //        params_lo_.iVZ = std::max(1, params.iVZ / dn);
    //        params_lo_.vox_x_mm = (params.iVX * params.vox_x_mm) / params_lo_.iVX;
    //        params_lo_.vox_y_mm = (params.iVY * params.vox_y_mm) / params_lo_.iVY;
    //        params_lo_.vox_z_mm = (params.iVZ * params.vox_z_mm) / params_lo_.iVZ;

    //        const size_t vol_lo_n =
    //            (size_t)params_lo_.iVX * params_lo_.iVY * params_lo_.iVZ;
    //        YK_CUDA_CHECK(cudaMalloc(&d_ones_vol_lo_, vol_lo_n * sizeof(float)));
    //        YK::Iter::fill_ones_launch(d_ones_vol_lo_, vol_lo_n, stream);

    //        fp_lo_.init(params_lo_, cfg.fp_task, deviceId);

    //        YK_CUDA_CHECK(cudaStreamSynchronize(stream));

    //        // ── DEBUG: dump C 与 1/(C+eps) ──
    //        if (cfg_.dump_debug) {
    //            yk_dump_raw(d_col_w_, vol_n, "dbg_C_colweight.raw");

    //            float* d_invc = nullptr;
    //            YK_CUDA_CHECK(cudaMalloc(&d_invc, vol_n * sizeof(float)));
    //            YK_CUDA_CHECK(cudaMemcpyAsync(d_invc, d_col_w_,
    //                vol_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    //            // invert_scale: x = (x>eps) ? 1/x : 0
    //            YK::Iter::invert_scale_launch(d_invc, 1.0f, cfg_.eps, vol_n, stream);
    //            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //            yk_dump_raw(d_invc, vol_n, "dbg_invC.raw");
    //            cudaFree(d_invc);

    //            YK_LOGI("[SIRT] C dim = {}x{}x{} (W x H x slices), "
    //                "ImageJ: 32-bit Real, then Reslice for XZ/YZ",
    //                params.iVX, params.iVY, params.iVZ);
    //        }

    //        is_initialized_ = true;
    //        YK_LOGI("[SIRT] init OK: {} angles, {} batches; "
    //            "C full-res precomputed, R lo-res {}x{}x{} (down {}){}",
    //            Na, cfg.n_batch,
    //            params_lo_.iVX, params_lo_.iVY, params_lo_.iVZ, dn,
    //            cfg_.dump_debug ? " [DUMP ON]" : "");
    //        return true;
    //    }

    //    bool iterate(
    //        const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t stream,
    //        unsigned int iterations)
    //    {
    //        if (!is_initialized_) return false;

    //        const size_t vol_n = (size_t)params.iVX * params.iVY * params.iVZ;
    //        const size_t view_n = (size_t)params.iPU * params.iPV;
    //        const int    Na = params.iPAng;
    //        const int    batch_Na = (Na + cfg_.n_batch - 1) / cfg_.n_batch;
    //        const size_t max_batch_sino_n = (size_t)batch_Na * view_n;

    //        float* d_sino_batch = nullptr;
    //        float* d_work_batch = nullptr;
    //        YK_CUDA_CHECK(cudaMalloc(&d_sino_batch, max_batch_sino_n * sizeof(float)));
    //        YK_CUDA_CHECK(cudaMalloc(&d_work_batch, max_batch_sino_n * sizeof(float)));

    //        struct Guard {
    //            float** a; float** b;
    //            ~Guard() { cudaFree(*a); cudaFree(*b); }
    //        } guard{ &d_sino_batch, &d_work_batch };

    //        bool dumped_R = false;

    //        for (unsigned int iter = 0; iter < iterations; ++iter)
    //        {
    //            std::string str = fmt::format("[SIRT] iter {}/{}", iter + 1, iterations);
    //            Util::CpuTimer timer(str.c_str());

    //            for (int b = 0; b < cfg_.n_batch; ++b)
    //            {
    //                const auto& bm = batches_[b];

    //                YK_CUDA_CHECK(cudaMemcpyAsync(
    //                    d_sino_batch,
    //                    d_sino_meas + bm.start_angle * view_n,
    //                    bm.sino_n * sizeof(float),
    //                    cudaMemcpyDeviceToDevice, stream));

    //                // Step1: 正投影 A_b x
    //                YK_CUDA_CHECK(cudaMemsetAsync(
    //                    d_work_batch, 0, bm.sino_n * sizeof(float), stream));
    //                fp_.run(d_vol, bm.pb, d_work_batch, stream);

    //                // Step2: 残差 r = b - A_b x
    //                YK::Iter::residual_launch(
    //                    d_sino_batch, d_work_batch,
    //                    d_sino_batch, bm.sino_n, stream);

    //                // Step3: 现算本块 R = A_b·1_vol_lo (低分辨率)
    //                SCBCTParams pb_lo = params_lo_;
    //                pb_lo.iPAng = bm.K;
    //                pb_lo.angle_list = bm.pb.angle_list;
    //                YK_CUDA_CHECK(cudaMemsetAsync(
    //                    d_work_batch, 0, bm.sino_n * sizeof(float), stream));
    //                fp_lo_.run(d_ones_vol_lo_, pb_lo, d_work_batch, stream);

    //                // ── DEBUG: dump 首轮首块 R ──
    //                if (cfg_.dump_debug && !dumped_R && iter == 0 && b == 0) {
    //                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //                    yk_dump_raw(d_work_batch, bm.sino_n, "dbg_R_lineweight_b0.raw");
    //                    YK_LOGI("[SIRT] R(b0) dim = {}x{}x{} (iPU x iPV x K)",
    //                        params.iPU, params.iPV, bm.K);
    //                    dumped_R = true;
    //                }

    //                // Step4: R 行归一化 r /= (R + eps)
    //                YK::Iter::divide_launch(
    //                    d_sino_batch, d_work_batch, cfg_.eps, bm.sino_n, stream);

    //                // Step5: 反投影累加 d_bp_ += A_bᵀ r (首块清零)
    //                bp_.run(d_sino_batch, bm.pb, d_bp_, stream, /*clear_vol=*/(b == 0));
    //            }

    //            // Step6: 列归一化更新 x += lambda * bp / (C + eps)
    //            YK::Iter::update_launch(
    //                d_vol, d_bp_, d_col_w_,
    //                lambda_cur_, cfg_.eps,
    //                vol_n, stream);

    //            lambda_cur_ *= cfg_.lambda_red;

    //            if (cfg_.use_min)
    //                YK::Iter::clamp_min_launch(d_vol, vol_n, cfg_.min_constraint, stream);
    //            if (cfg_.use_max)
    //                YK::Iter::clamp_max_launch(d_vol, vol_n, cfg_.max_constraint, stream);

    //            if ((iter + 1) % 5 == 0 || iter + 1 == iterations) {
    //                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
    //                const size_t cx =
    //                    (size_t)(params.iVZ / 2) * params.iVY * params.iVX
    //                    + (size_t)(params.iVY / 2) * params.iVX
    //                    + params.iVX / 2;
    //                float h_center = 0.f;
    //                cudaMemcpy(&h_center, d_vol + cx, sizeof(float), cudaMemcpyDeviceToHost);
    //                YK_LOGI("[SIRT] iter {}/{} vol_center={:.6e} lambda={:.4e}",
    //                    iter + 1, iterations, h_center, lambda_cur_);
    //            }
    //        }
    //        return true;
    //    }

    //    bool run(const float* d_sino_meas,
    //        float* d_vol,
    //        const SCBCTParams& params,
    //        cudaStream_t stream)
    //    {
    //        if (!is_initialized_) return false;
    //        return iterate(d_sino_meas, d_vol, params, stream, cfg_.n_iter);
    //    }

    //    void reset() { lambda_cur_ = cfg_.lambda; }

    //    void release()
    //    {
    //        if (d_bp_) { cudaFree(d_bp_);            d_bp_ = nullptr; }
    //        if (d_col_w_) { cudaFree(d_col_w_);         d_col_w_ = nullptr; }
    //        if (d_ones_vol_lo_) { cudaFree(d_ones_vol_lo_);   d_ones_vol_lo_ = nullptr; }

    //        batches_.clear();
    //        fp_.release();
    //        fp_lo_.release();
    //        bp_.release();
    //        is_initialized_ = false;
    //        lambda_cur_ = 1.0f;
    //    }

    //    ~SIRT() { release(); }

    //private:
    //    struct BatchMeta {
    //        SCBCTParams pb;
    //        int         K;
    //        size_t      sino_n;
    //        int         start_angle;
    //    };

    //    bool         is_initialized_ = false;
    //    float        lambda_cur_ = 1.0f;
    //    int          deviceId_ = 0;
    //    SCBCTParams  params_;
    //    SCBCTParams  params_lo_;
    //    Config       cfg_;

    //    ForwardOperatorAdapter fp_;
    //    ConeProjector     fp_lo_;
    //    BackOperatorAdapter bp_;

    //    std::vector<BatchMeta> batches_;

    //    float* d_bp_ = nullptr;
    //    float* d_col_w_ = nullptr;
    //    float* d_ones_vol_lo_ = nullptr;
    //};

    //YK_INLINE bool sirt_reconstruct(
    //    const float* d_sino_meas,
    //    float* d_vol,
    //    const SCBCTParams& params,
    //    cudaStream_t stream,
    //    SIRT::Config cfg = {})
    //{
    //    SIRT recon;
    //    if (!recon.init(params, cfg, stream)) return false;
    //    return recon.run(d_sino_meas, d_vol, params, stream);
    //}


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
            if (cfg_.dump_debug) {
                yk_dump_raw(d_col_w_, vol_n, "dbg_C_colweight.raw");
                YK_LOGI("[SIRT] C dim = {}x{}x{}",
                    params.iVX, params.iVY, params.iVZ);
            }

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

                        // DEBUG: dump 首轮首批 R
                        if (cfg_.dump_debug && !dumped_R && iter == 0 && b == 0) {
                            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                            yk_dump_raw(d_work_batch, bm.sino_n,
                                "dbg_R_lineweight_b0.raw");
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
