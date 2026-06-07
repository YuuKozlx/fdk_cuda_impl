#pragma once
#include <algorithm>
#include <cstdio>
#include <functional>
#include <vector>
#include <vector_functions.hpp>
#include <vector_types.h>
#include "YkBackProjectProcessor.hpp"
#include "YkFDKFilterProcessor.hpp"
#include "YkFDKGpuContext.hpp"
#include "YkFDKVecGeoDerived.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "FDK/YkFDKParkerWeightProcessor.hpp"
#include "FDK/YkFDKPreWeightProcessor.hpp"
#include "FDK/YkFdkPipelineContext.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "util/YkVecOperation.hpp"

namespace YK {

    // dump 回调的 payload，由内部构造，外部只读
    struct FdkDumpPayload {
        int          viewIdx;
        const char* stage;
        void* d_buf;
        size_t       n;
        cudaStream_t stream;
        void* userdata;
    };

    using FdkDumpCallback = std::function<void(void*)>;


    // ================================================================
    // FdkReconstructor
    // ================================================================
    // ================================================================
    // FdkReconstructor
    //
    // 优化一：消除 O(N²) 的全量 ProjGeom的计算。
    //   原版每次 feed 重建 new_total 个 geo 然后只用尾部，
    //   改为只建当前 batch 的 geo，增量上传到 gpu_ctx_ 的
    //   prev_total 偏移位置。
    //   gpu_ctx_ 容量改为 iPAngTotal（在 init 时传入）。
    //   其余逻辑（chunk 循环、processor、FdkGpuContext 结构）完全不变。
    // ================================================================
    class FdkReconstructor {
    public:
        FdkReconstructor() = default;

        void reset() {
            total_received_ = 0;
        }

        int  totalReceived()  const { return total_received_; }
        bool isInitialized()  const { return is_initialized_; }

        void release()
        {
            pw_.release();
            pkw_.release();
            flt_.release();
            bp_.release();
            gpu_ctx_.release();  // 释放 proj buffer 和 geo buffer
            reset();
            Kchunk_ = 0;
            bParker_ = false;
            is_initialized_ = false;
        }

        // ----------------------------------------------------------------
        // init：固定参数确定后调一次。
        // gpu_ctx_ 按 iPAngTotal 分配 geo/gv/coeffs 容量，
        // proj buffer 仍按 Kchunk 分配。
        // ----------------------------------------------------------------
        bool init(const SCBCTParams& params, int Kchunk, cudaStream_t stream, int device_id = 0)
        {
            Kchunk = std::min(Kchunk, kMaxChunkAng); // clip Kchunk 上限
            YK_LOGD("Kchunk is reset to {} (max {})", Kchunk, kMaxChunkAng);

            Kchunk_ = Kchunk;
            bParker_ = params.bShortScan;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;  // geo buffer 容量

            // PreweightProcessor
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] PreweightProcessor init failed\n");
                    return false;
                }
            }

            // ParkerWeightProcessor
            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                ictx.nDirSign = params.nDirSign;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] ParkerWeightProcessor init failed\n");
                    return false;
                }
            }

            // FilterProcessor
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    fprintf(stderr, "[FdkReconstructor] FilterProcessor init failed\n");
                    return false;
                }
            }

            // BpProcessor
            {
                SVolGeom vol_geom = SVolGeom::make_centered(
                    iVX, iVY, iVZ, params.vox_x_mm, params.vox_y_mm, params.vox_z_mm);
                vol_geom.center = make_float3(
                    params.vol_offset_x_mm,
                    params.vol_offset_y_mm,
                    params.vol_offset_z_mm);

                BpInitContext ictx{};
                ictx.vol_geom = vol_geom;
                ictx.use_precomputed = true;
                bp_.setInitContext(&ictx);
                if (!bp_.init()) {
                    fprintf(stderr, "[FdkReconstructor] BpProcessor init failed\n");
                    return false;
                }
            }

            // gpu_ctx_：
            //   geo/gv/coeffs 按 iPAngTotal 分配（支持增量上传）
            //   proj buffer 按 Kchunk 分配（每次 chunk 覆盖写）
            {
                // iPAng 字段传 iPA_total，让 geo buffer 按总视图数分配
                SProjDims dims{ iPU, iPV, iPA_total };
                // 空 geo/gv，只分配不上传
                gpu_ctx_.init(dims, iPA_total, stream, device_id);
            }

            is_initialized_ = true;
            return true;
        }

        bool reinitProcessors(const SCBCTParams& params, int K, cudaStream_t stream)
        {
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] PreweightProcessor reinit failed\n");
                    return false;
                }
            }

            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                ictx.nDirSign = params.nDirSign;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    fprintf(stderr, "[FdkReconstructor] ParkerWeightProcessor reinit failed\n");
                    return false;
                }
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    fprintf(stderr, "[FdkReconstructor] FilterProcessor reinit failed\n");
                    return false;
                }
            }

            return true;
        }



        // feed
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol = false,
            TaskDumpCallback onDump = nullptr, void* userdata = nullptr)
        {
            return feed_impl(h_proj_batch, params, Kchunk_, stream,
                d_vol_out, clear_vol, onDump, userdata);
        }

    private:
        int  Kchunk_ = 0;
        bool bParker_ = false;
        bool is_initialized_ = false;

        int total_received_ = 0;

        Fdk::PreweightProcessor    pw_;
        Fdk::ParkerWeightProcessor pkw_;
        Fdk::FilterProcessor       flt_;
        Fdk::BpProcessor           bp_;
        FdkGpuContext              gpu_ctx_;   // 持久成员，init() 时分配

        bool feed_impl(
            const float* h_proj_batch,
            const SCBCTParams& params,
            int                Kchunk,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol,
            TaskDumpCallback    onDump = nullptr, void* userdata = nullptr)
        {
            if (!is_initialized_) {
                YK_LOGE("[FdkReconstructor] not initialized, call init() first");
                return false;
            }
            if (Kchunk > kMaxChunkAng) {
                YK_LOGE("[FdkReconstructor] Kchunk={} exceeds kMaxChunkAng={}", Kchunk, kMaxChunkAng);
                return false;
            }
            if (!h_proj_batch || params.iPAng <= 0) {
                YK_LOGE("[FdkReconstructor] invalid input");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[FdkReconstructor] angle_list size mismatch");
                return false;
            }

            const int batch_count = params.iPAng;
            const int prev_total = total_received_;   // 本次 feed 前已接收的视图数
            total_received_ += batch_count;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            // -------------------------------------------------------
            // [优化一] 只建当前 batch 的 geo，O(batch_count)
            // 原版：h_geo_full(new_total) 重建全量，O(N²)
            // -------------------------------------------------------
            std::vector<SConeProjGeomVec>    h_geo(batch_count);
            std::vector<SFDKGeoParamPerView> h_gv(batch_count);

            build_circular_vec_geometry_from_theta(
                h_geo, params.angle_list, batch_count,
                iPU, iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_geo, h_gv);

            // -------------------------------------------------------
            // [优化一] 增量上传：写入 gpu_ctx_ 的 prev_total 偏移位置
            // geo/gv/coeffs buffer 容量 = iPAngTotal，足以容纳全部视图
            // -------------------------------------------------------
            gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, prev_total, batch_count, stream);

            if (clear_vol) {
                const size_t n = (size_t)iVX * iVY * iVZ;
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
            }

            const size_t view_elems = (size_t)iPU * iPV;
            float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
            float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
            float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

            bool filter_dirty = false;

            // chunk 循环开头定义一次


            for (int base = 0; base < batch_count; base += Kchunk) {
                const int K = std::min(Kchunk, batch_count - base);



                // 尾包时重建所有 processor
                if (K != Kchunk_) {
                    if (!reinitProcessors(params, K, stream)) return false;
                    filter_dirty = true;
                }

                // geo/gv/coeffs 偏移 = prev_total + base（全局位置）
                const int global_base = prev_total + base;


                auto triggerDump = [&](const char* stage, float* d_base) {
                    if (!onDump) return;
                    cudaStreamSynchronize(stream);  // 只在有 dump 时同步
                    for (int i = 0; i < K; ++i) {
                        DumpPayload payload{
                            global_base + i, stage,
                            static_cast<void*>(d_base + i * view_elems),
                            view_elems, stream, userdata
                        };
                        onDump(&payload);
                    }
                    };

                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + global_base, K, stream);
                gpu_ctx_.proj.uploadProjChunk(
                    h_proj_batch + (size_t)base * view_elems, K, stream);

                // Preweight
                PreweightChunkContext pctx{};
                pctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                pctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                pctx.K = K;
                pw_.setContext(&pctx);
                pw_.process(d_chunk_in, d_chunk_pw, stream);

                triggerDump("pw", d_chunk_pw);

                // ParkerWeight
                if (bParker_) {
                    ParkerWeightChunkContext pkctx{};
                    pkctx.h_angles = params.angle_list.data() + base;
                    pkctx.K = K;
                    pkw_.setContext(&pkctx);
                    pkw_.process(d_chunk_pw, d_chunk_pw, stream);

                    triggerDump("parker", d_chunk_pw);
                }

                // Filter
                FdkFilterContext fctx{ h_gv.data() + base, K };
                flt_.setContext(&fctx);
                flt_.process(d_chunk_pw, d_chunk_flt, stream);
                triggerDump("flt", d_chunk_flt);

                // Backprojection
                BpChunkContext bctx{};
                bctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                bctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                bctx.K = K;
                bp_.setContext(&bctx);
                bp_.process(gpu_ctx_.proj.d_texObjs(), d_vol_out, stream);
            }

            // 恢复标准尺寸供下次 feed 使用
            if (filter_dirty) {
                if (!reinitProcessors(params, Kchunk_, stream)) return false;
            }

            return true;
        }
    };

    // ================================================================
    // 全局便捷函数（单次全量重建）
    // ================================================================
    YK_INLINE bool fdk_recon(
        const float* h_proj,
        float* d_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        bool               clear_vol = true,
        TaskDumpCallback onDump = nullptr, void* userdata = nullptr)
    {
        FdkReconstructor recon;
        if (!recon.init(params, Kchunk, stream))
            return false;
        return recon.feed(h_proj, params, stream,
            d_vol_out, clear_vol, onDump, userdata);
    }

} // namespace YK


namespace YK {

    class FdkReconstructorEx {
    public:
        FdkReconstructorEx() = default;

        void reset() {
            total_received_ = 0;
        }

        int  totalReceived()  const { return total_received_; }
        bool isInitialized()  const { return is_initialized_; }

        void release()
        {
            pw_.release();
            pkw_.release();
            flt_.release();
            bp_.release();
            gpu_ctx_.release();
            reset();
            Kchunk_ = 0;
            bParker_ = false;
            is_initialized_ = false;
        }

        bool init(const SCBCTParams& params, int Kchunk, cudaStream_t stream, int device_id = 0)
        {
            // 与 FdkReconstructor::init 完全相同
            Kchunk = std::min(Kchunk, kMaxChunkAng);
            Kchunk_ = Kchunk;
            bParker_ = params.bShortScan;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;

            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) return false;
            }
            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                ictx.nDirSign = params.nDirSign;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) return false;
            }
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) return false;
            }
            {
                SVolGeom vol_geom = SVolGeom::make_centered(
                    iVX, iVY, iVZ, params.vox_x_mm, params.vox_z_mm);
                vol_geom.center = make_float3(
                    params.vol_offset_x_mm,
                    params.vol_offset_y_mm,
                    params.vol_offset_z_mm);
                BpInitContext ictx{};
                ictx.vol_geom = vol_geom;
                ictx.use_precomputed = true;
                bp_.setInitContext(&ictx);
                if (!bp_.init()) return false;
            }
            {
                SProjDims dims{ iPU, iPV, iPA_total };
                gpu_ctx_.init(dims, iPA_total, stream, device_id);
            }

            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // feed：接受外部预建几何，不再内部 build_circular
        // h_views_ext.size() == params.iPAng（当前batch）
        // ----------------------------------------------------------------
        bool feed(
            const float* h_proj_batch,
            const SCBCTParams& params,
            const std::vector<SConeProjGeomVec>& h_views_ext,
            cudaStream_t                         stream,
            float* d_vol_out,
            bool                                 clear_vol = false,
            TaskDumpCallback                     onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!is_initialized_) {
                YK_LOGE("[FdkReconstructorEx] not initialized"); return false;
            }
            if (!h_proj_batch || params.iPAng <= 0) {
                YK_LOGE("[FdkReconstructorEx] invalid input"); return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[FdkReconstructorEx] angle_list size mismatch"); return false;
            }
            if ((int)h_views_ext.size() != params.iPAng) {
                YK_LOGE("[FdkReconstructorEx] h_views_ext size {} != params.iPAng {}",
                    h_views_ext.size(), params.iPAng);
                return false;
            }

            const int batch_count = params.iPAng;
            const int prev_total = total_received_;
            total_received_ += batch_count;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;

            // ---- 唯一差别：geo 直接用外部传入 ----
            std::vector<SFDKGeoParamPerView> h_gv(batch_count);
            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_views_ext, h_gv);

            gpu_ctx_.uploadGeoIncremental(h_views_ext, h_gv, prev_total, batch_count, stream);

            if (clear_vol) {
                const size_t n = (size_t)iVX * iVY * iVZ;
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0, n * sizeof(float), stream));
            }

            const size_t view_elems = (size_t)iPU * iPV;
            float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
            float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
            float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

            bool filter_dirty = false;

            for (int base = 0; base < batch_count; base += Kchunk_) {
                const int K = std::min(Kchunk_, batch_count - base);

                if (K != Kchunk_) {
                    if (!reinitProcessors(params, K, stream)) return false;
                    filter_dirty = true;
                }

                const int global_base = prev_total + base;

                auto triggerDump = [&](const char* stage, float* d_base) {
                    if (!onDump) return;
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < K; ++i) {
                        DumpPayload payload{
                            global_base + i, stage,
                            static_cast<void*>(d_base + i * view_elems),
                            view_elems, stream, userdata
                        };
                        onDump(&payload);
                    }
                    };

                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + global_base, K, stream);
                gpu_ctx_.proj.uploadProjChunk(
                    h_proj_batch + (size_t)base * view_elems, K, stream);

                PreweightChunkContext pctx{};
                pctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                pctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                pctx.K = K;
                pw_.setContext(&pctx);
                pw_.process(d_chunk_in, d_chunk_pw, stream);
                triggerDump("pw", d_chunk_pw);

                if (bParker_) {
                    ParkerWeightChunkContext pkctx{};
                    pkctx.h_angles = params.angle_list.data() + base;
                    pkctx.K = K;
                    pkw_.setContext(&pkctx);
                    pkw_.process(d_chunk_pw, d_chunk_pw, stream);
                    triggerDump("parker", d_chunk_pw);
                }

                FdkFilterContext fctx{ h_gv.data() + base, K };
                flt_.setContext(&fctx);
                flt_.process(d_chunk_pw, d_chunk_flt, stream);
                triggerDump("flt", d_chunk_flt);

                BpChunkContext bctx{};
                bctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                bctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                bctx.K = K;
                bp_.setContext(&bctx);
                bp_.process(gpu_ctx_.proj.d_texObjs(), d_vol_out, stream);
            }

            if (filter_dirty) {
                if (!reinitProcessors(params, Kchunk_, stream)) return false;
            }

            return true;
        }

    private:
        int  Kchunk_ = 0;
        bool bParker_ = false;
        bool is_initialized_ = false;
        int  total_received_ = 0;

        Fdk::PreweightProcessor    pw_;
        Fdk::ParkerWeightProcessor pkw_;
        Fdk::FilterProcessor       flt_;
        Fdk::BpProcessor           bp_;
        FdkGpuContext              gpu_ctx_;

        bool reinitProcessors(const SCBCTParams& params, int K, cudaStream_t stream)
        {
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) return false;
            }
            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                ictx.nDirSign = params.nDirSign;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) return false;
            }
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ params.iPU, params.iPV, K };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) return false;
            }
            return true;
        }
    };

    // 全局便捷函数
    YK_INLINE bool fdk_recon_ex(
        const float* h_proj,
        float* d_vol_out,
        const SCBCTParams& params,
        const std::vector<SConeProjGeomVec>& h_views_ext,
        int                                  Kchunk,
        cudaStream_t                         stream,
        bool                                 clear_vol = true,
        TaskDumpCallback                     onDump = nullptr,
        void* userdata = nullptr)
    {
        FdkReconstructorEx recon;
        if (!recon.init(params, Kchunk, stream))
            return false;
        return recon.feed(h_proj, params, h_views_ext, stream,
            d_vol_out, clear_vol, onDump, userdata);
    }

};