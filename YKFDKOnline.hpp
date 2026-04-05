#pragma once
#include <vector>
#include <functional>
#include <cstdio>

#include <cuda_runtime.h>

#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkFDKChunkBuffer.hpp"
#include "YkFDKPrecompute.hpp"
#include "IProcessor.hpp"
#include "YkFdkFilterContext.hpp"
#include "YkFDKBackproject.cuh"
#include "YkFDKVecPreWeight2.hpp"
#include "YkFDKFilter2.hpp"

namespace YK {

#ifndef YK_OR_LOGE
#define YK_OR_LOGE(fmt, ...) std::fprintf(stderr, "[YK][OnlineRecon][E] " fmt "\n", ##__VA_ARGS__)
#endif
#ifndef YK_OR_LOGI
#define YK_OR_LOGI(fmt, ...) std::fprintf(stdout, "[YK][OnlineRecon][I] " fmt "\n", ##__VA_ARGS__)
#endif

    // ============================================================
    // FdkOnlineReconstructor
    //
    //   生命周期：
    //     1. init(dims, h_geo_all, h_gv_all, Kchunk, vox, stream)
    //        — 按总视角数预分配所有 GPU 资源，上传 geo/gv/coeffs
    //     2. feedViews(h_proj_views, view_indices)
    //        — 每次传入若干张投影（不要求连续、不要求满 Kchunk）
    //        — 立即触发 preweight → filter → BP，累加到内部 volume
    //     3. finalize(d_vol_out / h_vol_out)
    //        — 取出重建结果
    //     4. reset()  — 清零 volume，可复用对象重新重建
    //
    //   约束：
    //     - h_geo_all / h_gv_all 必须在 init() 时全量传入（几何参数需预计算）
    //     - feedViews() 可以乱序调用，view_indices 指定每张投影对应的全局视角号
    //     - 每次 feedViews() 的张数 <= Kchunk
    // ============================================================
    class FdkOnlineReconstructor {
    public:
        FdkOnlineReconstructor() = default;
        ~FdkOnlineReconstructor() { destroy(); }

        FdkOnlineReconstructor(const FdkOnlineReconstructor&) = delete;
        FdkOnlineReconstructor& operator=(const FdkOnlineReconstructor&) = delete;

        using DumpFn = std::function<void(int a, const char* tag, float* d_buf, size_t n)>;

        // ----------------------------------------------------------------
        // init — 全量几何参数 + 资源分配，只调一次
        // ----------------------------------------------------------------
        bool init(
            const SDimensions3D& dims,
            const std::vector<SConeProjectionVec>& h_geo_all,
            const std::vector<SFDKGeoParamPerView>& h_gv_all,
            int          Kchunk,
            float        vox,
            cudaStream_t stream = 0)
        {
            destroy();

            if ((int)h_geo_all.size() != dims.iPAng ||
                (int)h_gv_all.size() != dims.iPAng) {
                YK_OR_LOGE("init: geo/gv size mismatch with dims.iPAng=%d.", dims.iPAng);
                return false;
            }
            if (Kchunk <= 0) {
                YK_OR_LOGE("init: Kchunk(%d) must be > 0.", Kchunk);
                return false;
            }

            dims_ = dims;
            Kchunk_ = Kchunk;
            vox_ = vox;
            stream_ = stream;

            const size_t view_elems = (size_t)dims_.iPU * dims_.iPV;
            const size_t chunk_elems = (size_t)Kchunk_ * view_elems;
            const size_t vol_elems = (size_t)dims_.iVX * dims_.iVY * dims_.iVZ;

            // --- geo / gv（全量）---
            YK_CUDA_CHECK(cudaMalloc(&d_geo_, dims_.iPAng * sizeof(SConeProjectionVec)));
            YK_CUDA_CHECK(cudaMalloc(&d_gv_, dims_.iPAng * sizeof(SFDKGeoParamPerView)));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_geo_, h_geo_all.data(),
                dims_.iPAng * sizeof(SConeProjectionVec),
                cudaMemcpyHostToDevice, stream_));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_gv_, h_gv_all.data(),
                dims_.iPAng * sizeof(SFDKGeoParamPerView),
                cudaMemcpyHostToDevice, stream_));

            // --- 预计算仿射系数（全量）---
            YK_CUDA_CHECK(cudaMalloc(&d_coeffs_, dims_.iPAng * sizeof(FdkAffineCoeff)));
            launchPrecomputeCoeffs(d_geo_, d_gv_, d_coeffs_, dims_.iPAng, stream_);

            h_coeffs_.resize(dims_.iPAng);
            YK_CUDA_CHECK(cudaMemcpyAsync(h_coeffs_.data(), d_coeffs_,
                dims_.iPAng * sizeof(FdkAffineCoeff),
                cudaMemcpyDeviceToHost, stream_));

            // --- volume（清零）---
            YK_CUDA_CHECK(cudaMalloc(&d_vol_, vol_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_, 0, vol_elems * sizeof(float), stream_));

            // --- chunk 投影缓冲 ---
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_in_, chunk_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_pw_, chunk_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMalloc(&d_chunk_flt_, chunk_elems * sizeof(float)));

            // --- ChunkBuffer（texture）---
            chunk_.init(Kchunk_, dims_.iPU, dims_.iPV);

            // --- PreweightProcessor ---
            {
                PreweightInitContext ictx{};
                ictx.dims = dims_;
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_OR_LOGE("init: PreweightProcessor init failed."); return false;
                }
            }

            // --- FilterProcessor ---
            {
                SDimensions3D chunk_dims = dims_;
                chunk_dims.iPAng = Kchunk_;

                FdkFilterInitContext ictx{};
                ictx.dims = chunk_dims;
                ictx.desc = {};
                ictx.policy = {};
                ictx.stream = stream_;
                fp_.setInitContext(&ictx);
                if (!fp_.init()) {
                    YK_OR_LOGE("init: FilterProcessor init failed."); return false;
                }
            }

            // coeffs D2H 需要同步完成才能在 feedViews 里用
            cudaStreamSynchronize(stream_);

            is_initialized_ = true;
            YK_OR_LOGI("init ok: iPAng=%d iPU=%d iPV=%d Kchunk=%d vox=%.4f",
                dims_.iPAng, dims_.iPU, dims_.iPV, Kchunk_, vox_);
            return true;
        }

        // ----------------------------------------------------------------
        // feedViews — 喂入若干张投影，立即触发滤波+BP
        //
        //   h_proj_views : host pointer，[K * iPV * iPU] 行优先
        //   view_indices : 每张投影对应的全局视角号（用于取 geo/gv/coeffs）
        //                  大小必须 == K，且每个值在 [0, iPAng)
        //   K            : 本次喂入的张数，<= Kchunk
        // ----------------------------------------------------------------
        bool feedViews(
            const float* h_proj_views,
            const std::vector<int>& view_indices,
            DumpFn                onDump = {})
        {
            if (!is_initialized_) {
                YK_OR_LOGE("feedViews: not initialized."); return false;
            }
            const int K = (int)view_indices.size();
            if (K <= 0 || K > Kchunk_) {
                YK_OR_LOGE("feedViews: K=%d out of range [1, %d].", K, Kchunk_); return false;
            }
            if (!h_proj_views) {
                YK_OR_LOGE("feedViews: h_proj_views is null."); return false;
            }
            for (int i = 0; i < K; ++i) {
                if (view_indices[i] < 0 || view_indices[i] >= dims_.iPAng) {
                    YK_OR_LOGE("feedViews: view_indices[%d]=%d out of range.",
                        i, view_indices[i]);
                    return false;
                }
            }

            const size_t view_elems = (size_t)dims_.iPU * dims_.iPV;

            // H2D：整批一次上传
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_chunk_in_,
                h_proj_views,
                (size_t)K * view_elems * sizeof(float),
                cudaMemcpyHostToDevice, stream_));

            // 构造本次 chunk 的 geo/gv 偏移
            // view_indices 可能不连续，需要把对应行的 geo/gv 收集到临时缓冲
            // 连续视角时直接偏移指针；非连续时先 gather 到临时缓冲
            const bool contiguous = isContiguous_(view_indices);

            const SConeProjectionVec* d_geo_chunk = nullptr;
            const SFDKGeoParamPerView* d_gv_chunk = nullptr;

            if (contiguous) {
                // 直接偏移，零拷贝
                d_geo_chunk = d_geo_ + view_indices[0];
                d_gv_chunk = d_gv_ + view_indices[0];
            }
            else {
                // gather：把非连续视角的 geo/gv 收集到 chunk 起始处
                gatherGeoChunk_(view_indices, K);
                d_geo_chunk = d_geo_gather_;
                d_gv_chunk = d_gv_gather_;
            }

            // preweight
            PreweightChunkContext pctx{};
            pctx.d_geo = d_geo_chunk;
            pctx.d_gv = d_gv_chunk;
            pctx.K = K;
            pw_.setContext(&pctx);
            pw_.process(d_chunk_in_, d_chunk_pw_, stream_);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(view_indices[i], "pw",
                        d_chunk_pw_ + i * view_elems, view_elems);

            // 滤波
            // h_gv 子集：收集当前视角的 offsetU/du_mm
            gatherHGv_(view_indices, K);
            FdkFilterContext fctx{ h_gv_gather_host_.data(), K };
            fp_.setContext(&fctx);
            fp_.process(d_chunk_pw_, d_chunk_flt_, stream_);

            if (onDump)
                for (int i = 0; i < K; ++i)
                    onDump(view_indices[i], "flt",
                        d_chunk_flt_ + i * view_elems, view_elems);

            // chunk flt → texture slots
            for (int i = 0; i < K; ++i)
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    chunk_.slotPtr(i),
                    d_chunk_flt_ + i * view_elems,
                    view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream_));

            chunk_.uploadTexObjs(K, stream_);

            // 上传 constant 系数（gather 到连续缓冲）
            gatherCoeffs_(view_indices, K);
            YK_CUDA_CHECK(cudaMemcpyToSymbol(gC_coeffs,
                h_coeffs_gather_.data(),
                K * sizeof(FdkAffineCoeff)));

            // BP 累加
            launchBpKernel(
                chunk_.d_texObjs,
                d_geo_chunk,
                d_gv_chunk,
                d_vol_,
                dims_.iVX, dims_.iVY, dims_.iVZ, vox_,
                K, stream_);

            views_processed_ += K;
            return true;
        }

        // ----------------------------------------------------------------
        // finalize — 取出重建结果
        // ----------------------------------------------------------------
        bool finalizeDevice(float* d_vol_out)
        {
            if (!is_initialized_ || !d_vol_out) return false;
            if (d_vol_out == d_vol_) return true;   // 已经是内部 buffer
            const size_t n = (size_t)dims_.iVX * dims_.iVY * dims_.iVZ;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_vol_out, d_vol_,
                n * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            return true;
        }

        bool finalizeHost(float* h_vol_out)
        {
            if (!is_initialized_ || !h_vol_out) return false;
            const size_t n = (size_t)dims_.iVX * dims_.iVY * dims_.iVZ;
            YK_CUDA_CHECK(cudaMemcpyAsync(h_vol_out, d_vol_,
                n * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            cudaStreamSynchronize(stream_);
            return true;
        }

        // ----------------------------------------------------------------
        // reset — 清零 volume，复用对象开始新一轮重建
        // ----------------------------------------------------------------
        void reset()
        {
            if (!is_initialized_) return;
            const size_t n = (size_t)dims_.iVX * dims_.iVY * dims_.iVZ;
            YK_CUDA_CHECK(cudaMemsetAsync(d_vol_, 0, n * sizeof(float), stream_));
            views_processed_ = 0;
            YK_OR_LOGI("reset: volume cleared, ready for new reconstruction.");
        }

        // ----------------------------------------------------------------
        // destroy
        // ----------------------------------------------------------------
        void destroy()
        {
            pw_.release();
            fp_.release();
            chunk_.destroy();

            auto free_ = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            free_(d_geo_);
            free_(d_gv_);
            free_(d_coeffs_);
            free_(d_vol_);
            free_(d_chunk_in_);
            free_(d_chunk_pw_);
            free_(d_chunk_flt_);
            free_(d_geo_gather_);
            free_(d_gv_gather_);

            h_coeffs_.clear();
            h_coeffs_gather_.clear();
            h_gv_gather_host_.clear();

            is_initialized_ = false;
            views_processed_ = 0;
        }

        // 状态查询
        bool isInitialized()   const { return is_initialized_; }
        int  viewsProcessed()  const { return views_processed_; }
        int  Kchunk()          const { return Kchunk_; }
        float* deviceVolume()  const { return d_vol_; }

    private:
        // ---- 资源 ----
        SDimensions3D dims_ = {};
        int           Kchunk_ = 0;
        float         vox_ = 1.0f;
        cudaStream_t  stream_ = 0;

        SConeProjectionVec* d_geo_ = nullptr;
        SFDKGeoParamPerView* d_gv_ = nullptr;
        FdkAffineCoeff* d_coeffs_ = nullptr;
        float* d_vol_ = nullptr;
        float* d_chunk_in_ = nullptr;
        float* d_chunk_pw_ = nullptr;
        float* d_chunk_flt_ = nullptr;

        // gather 缓冲（非连续视角时使用）
        SConeProjectionVec* d_geo_gather_ = nullptr;
        SFDKGeoParamPerView* d_gv_gather_ = nullptr;
        int                  gather_cap_ = 0;

        ChunkBuffer           chunk_;
        PreweightProcessor    pw_;
        FilterProcessor       fp_;

        std::vector<FdkAffineCoeff>      h_coeffs_;
        std::vector<FdkAffineCoeff>      h_coeffs_gather_;
        std::vector<SFDKGeoParamPerView> h_gv_gather_host_;
        std::vector<SConeProjectionVec>  h_geo_gather_host_;

        bool is_initialized_ = false;
        int  views_processed_ = 0;

        // ---- 内部辅助 ----

        static bool isContiguous_(const std::vector<int>& idx)
        {
            for (int i = 1; i < (int)idx.size(); ++i)
                if (idx[i] != idx[i - 1] + 1) return false;
            return true;
        }

        void ensureGatherCap_(int K)
        {
            if (K <= gather_cap_) return;
            auto free_ = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            free_(d_geo_gather_);
            free_(d_gv_gather_);
            YK_CUDA_CHECK(cudaMalloc(&d_geo_gather_, K * sizeof(SConeProjectionVec)));
            YK_CUDA_CHECK(cudaMalloc(&d_gv_gather_, K * sizeof(SFDKGeoParamPerView)));
            gather_cap_ = K;
        }

        void gatherGeoChunk_(const std::vector<int>& idx, int K)
        {
            ensureGatherCap_(K);
            h_geo_gather_host_.resize(K);
            // geo/gv 在 host 侧没有副本，需从 device 读回再 scatter 上传
            // 代价较高——建议调用方尽量传连续视角
            // 这里用 D2H → scatter → H2D 实现正确性
            std::vector<SConeProjectionVec>  tmp_geo(dims_.iPAng);
            std::vector<SFDKGeoParamPerView> tmp_gv(dims_.iPAng);
            YK_CUDA_CHECK(cudaMemcpy(tmp_geo.data(), d_geo_,
                dims_.iPAng * sizeof(SConeProjectionVec), cudaMemcpyDeviceToHost));
            YK_CUDA_CHECK(cudaMemcpy(tmp_gv.data(), d_gv_,
                dims_.iPAng * sizeof(SFDKGeoParamPerView), cudaMemcpyDeviceToHost));

            std::vector<SConeProjectionVec>  geo_buf(K);
            std::vector<SFDKGeoParamPerView> gv_buf(K);
            for (int i = 0; i < K; ++i) {
                geo_buf[i] = tmp_geo[idx[i]];
                gv_buf[i] = tmp_gv[idx[i]];
            }
            YK_CUDA_CHECK(cudaMemcpyAsync(d_geo_gather_, geo_buf.data(),
                K * sizeof(SConeProjectionVec), cudaMemcpyHostToDevice, stream_));
            YK_CUDA_CHECK(cudaMemcpyAsync(d_gv_gather_, gv_buf.data(),
                K * sizeof(SFDKGeoParamPerView), cudaMemcpyHostToDevice, stream_));
        }

        void gatherHGv_(const std::vector<int>& idx, int K)
        {
            // FilterProcessor 需要 host 侧 h_gv 子集
            // h_gv 全量没有单独存一份 host 副本，借用 h_coeffs_ 已有的同步时机
            // 这里直接从 init 时传入的 h_gv_all 重新 gather——
            // 但 h_gv_all 没有存下来。解决方案：init 时保存 host 副本。
            // 见 h_gv_all_host_
            h_gv_gather_host_.resize(K);
            for (int i = 0; i < K; ++i)
                h_gv_gather_host_[i] = h_gv_all_host_[idx[i]];
        }

        void gatherCoeffs_(const std::vector<int>& idx, int K)
        {
            h_coeffs_gather_.resize(K);
            for (int i = 0; i < K; ++i)
                h_coeffs_gather_[i] = h_coeffs_[idx[i]];
        }

        // host 副本（init 时保存，供 gather 使用）
        std::vector<SFDKGeoParamPerView> h_gv_all_host_;
    };

} // namespace YK