#pragma once
#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include <vector_functions.hpp>
#include <vector_types.h>

#include "FDK/YkBackProjectProcessor.hpp"
#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/YkFDKGpuContext.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "util/YkVecOperation.hpp"
#include "FDK/YkFDKParkerWeightProcessor.hpp"
#include "FDK/YkFDKPreWeightProcessor.hpp"
#include "FDK/YkFdkPipelineContext.hpp"
#include "common/YkVecGeo.hpp"

namespace YK {

    class FdkZSlabReconstructor {
    public:
        FdkZSlabReconstructor() = default;

        void reset()
        {
            total_received_ = 0;
            h_vol_buffer_ = nullptr;
        }

        bool isInitialized() const { return is_initialized_; }
        int  totalReceived() const { return total_received_; }

        void release()
        {
            pw_.release();
            pkw_.release();
            flt_.release();

            for (auto& bp : slab_bps_)
                if (bp) bp->release();
            slab_bps_.clear();
            slabs_.clear();

            if (d_vol_slab_) {
                cudaFree(d_vol_slab_);
                d_vol_slab_ = nullptr;
            }

            gpu_ctx_.release();
            h_vol_buffer_ = nullptr;
            h_vol_slab_buffer_.clear();

            reset();
            Kchunk_ = 0;
            bParker_ = false;
            is_initialized_ = false;
        }

        bool init(const SCBCTParams& params, int Kchunk,
            cudaStream_t stream, int z_block_size, int device_id = 0)
        {
            Kchunk_ = std::min(Kchunk, kMaxChunkAng);
            bParker_ = params.bShortScan;
            params_ = params;
            stream_ = stream;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;

            z_block_size_ = std::min(z_block_size, iVZ);

            // ---- PreweightProcessor ----
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk_ };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_LOGE("[FdkZSlabReconstructor] PreweightProcessor init failed");
                    return false;
                }
            }

            // ---- ParkerWeightProcessor ----
            if (bParker_) {
                ParkerWeightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk_ };
                ictx.fDetUSize = params.du_mm;
                ictx.fSrcOrigin = params.SID;
                ictx.fDetOrigin = params.SDD - params.SID;
                ictx.iPAnglesTotal = params.iPAngTotal;
                ictx.fScanRangeRad = params.scan_range_rad;
                ictx.fStartAngleRad = params.scan_start_angle_rad;
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    YK_LOGE("[FdkZSlabReconstructor] ParkerWeightProcessor init failed");
                    return false;
                }
            }

            // ---- FilterProcessor ----
            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, Kchunk_ };
                ictx.desc = params.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    YK_LOGE("[FdkZSlabReconstructor] FilterProcessor init failed");
                    return false;
                }
            }

            // ---- gpu_ctx_ ----
            {
                SProjDims dims{ iPU, iPV, iPA_total };
                gpu_ctx_.init(dims, iPA_total, stream, device_id);
            }

            // ---- 设备端单块 vol slab buffer ----
            {
                const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
                YK_CUDA_CHECK(cudaMalloc(&d_vol_slab_, slab_elems_max * sizeof(float)));
            }

            // ---- 按 z_block_size 分段 ----
            // SlabContext 是 POD，存 vector 无问题。
            // BpProcessor 不可拷贝也不可移动，用 unique_ptr 持有，
            // vector 扩容时只移动指针，BpProcessor 对象原地不动。
            {
                const float vox_z = params.vox_z_mm;
                const float full_center_z = params.vol_offset_z_mm;

                const int num_slabs = (iVZ + z_block_size_ - 1) / z_block_size_;
                slabs_.reserve(num_slabs);
                slab_bps_.reserve(num_slabs);

                for (int z_start = 0; z_start < iVZ; z_start += z_block_size_) {
                    const int z_count = std::min(z_block_size_, iVZ - z_start);

                    SlabContext slab{};
                    slab.z_start = z_start;
                    slab.z_count = z_count;
                    slabs_.push_back(slab);

                    SVolGeom sub_geom = SVolGeom::make_centered(
                        iVX, iVY, z_count, params.vox_x_mm, vox_z);

                    const float slab_center_vox = z_start + z_count * 0.5f;
                    const float full_center_vox = iVZ * 0.5f;
                    const float z_offset_mm = (slab_center_vox - full_center_vox) * vox_z;

                    sub_geom.center = make_float3(
                        params.vol_offset_x_mm,
                        params.vol_offset_y_mm,
                        full_center_z + z_offset_mm);

                    // unique_ptr：堆上原地构造，vector 扩容只移动指针
                    auto bp_ptr = std::make_unique<Fdk::BpProcessor>();
                    BpInitContext bctx{};
                    bctx.vol_geom = sub_geom;
                    bctx.use_precomputed = false;
                    bp_ptr->setInitContext(&bctx);
                    if (!bp_ptr->init()) {
                        YK_LOGE("[FdkZSlabReconstructor] BpProcessor init failed for slab z={}",
                            z_start);
                        return false;
                    }
                    slab_bps_.push_back(std::move(bp_ptr));
                }
            }

            // ---- CPU 完整体积累积 buffer ----
            const size_t vol_elems = (size_t)iVX * iVY * iVZ;


            // ---- CPU 单段回读 buffer ----
            const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
            h_vol_slab_buffer_.resize(slab_elems_max);

            is_initialized_ = true;
            YK_LOGI("[FdkZSlabReconstructor] init OK: {}x{}x{} vox, {} slabs, z_block={}",
                iVX, iVY, iVZ, (int)slabs_.size(), z_block_size_);
            return true;
        }

        bool feed(const float* h_proj_batch,
            const SCBCTParams& params,
            float* h_vol_out,
            cudaStream_t       stream)
        {
            if (!is_initialized_) {
                YK_LOGE("[FdkZSlabReconstructor] not initialized");
                return false;
            }
            if (!h_proj_batch || params.iPAng <= 0) {
                YK_LOGE("[FdkZSlabReconstructor] invalid input");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[FdkZSlabReconstructor] angle_list size mismatch");
                return false;
            }

            h_vol_buffer_ = h_vol_out;
            const int batch_count = params.iPAng;
            const int prev_total = total_received_;
            total_received_ += batch_count;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            // ---- 建当前 batch 的 geo ----
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

            gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, prev_total, batch_count, stream);

            const size_t view_elems = (size_t)iPU * iPV;
            float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
            float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
            float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

            bool filter_dirty = false;

            // ---- chunk 循环 ----
            for (int base = 0; base < batch_count; base += Kchunk_) {
                const int K = std::min(Kchunk_, batch_count - base);

                if (K != Kchunk_) {
                    if (!reinitProcessors(params, K, stream)) return false;
                    filter_dirty = true;
                }

                const int global_base = prev_total + base;

                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + global_base, K, stream);
                gpu_ctx_.proj.uploadProjChunk(
                    h_proj_batch + (size_t)base * view_elems, K, stream);

                // preweight
                PreweightChunkContext pctx{};
                pctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                pctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                pctx.K = K;
                pw_.setContext(&pctx);
                pw_.process(d_chunk_in, d_chunk_pw, stream);

                // parker
                if (bParker_) {
                    ParkerWeightChunkContext pkctx{};
                    pkctx.h_angles = params.angle_list.data() + base;
                    pkctx.K = K;
                    pkw_.setContext(&pkctx);
                    pkw_.process(d_chunk_pw, d_chunk_pw, stream);
                }

                // filter
                FdkFilterContext fctx{ h_gv.data() + base, K };
                flt_.setContext(&fctx);
                flt_.process(d_chunk_pw, d_chunk_flt, stream);

                // ---- 对每个 slab：清零 → 反投影 → 回读 → CPU 累加 ----
                for (int si = 0; si < (int)slabs_.size(); ++si) {
                    const SlabContext& slab = slabs_[si];
                    Fdk::BpProcessor& bp = *slab_bps_[si];

                    const size_t slab_elems = (size_t)iVX * iVY * slab.z_count;

                    // a. 清零
                    YK_CUDA_CHECK(cudaMemsetAsync(
                        d_vol_slab_, 0, slab_elems * sizeof(float), stream));

                    // b. 反投影
                    BpChunkContext bctx{};
                    bctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                    bctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                    bctx.K = K;
                    bp.setContext(&bctx);
                    bp.process(gpu_ctx_.proj.d_texObjs(), d_vol_slab_, stream);

                    // c. 同步后回读
                    YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                    YK_CUDA_CHECK(cudaMemcpy(
                        h_vol_slab_buffer_.data(), d_vol_slab_,
                        slab_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

                    // d. 累加到 h_vol_buffer_ 对应段
                    const bool is_first_chunk = (prev_total == 0 && base == 0);
                    accumulateSlab_(slab.z_start, slab.z_count, iVX, iVY, is_first_chunk);
                }
            }

            if (filter_dirty) {
                if (!reinitProcessors(params, Kchunk_, stream)) return false;
            }

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
                    YK_LOGE("[FdkZSlabReconstructor] PreweightProcessor reinit failed");
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
                pkw_.setInitContext(&ictx);
                if (!pkw_.init()) {
                    YK_LOGE("[FdkZSlabReconstructor] ParkerWeightProcessor reinit failed");
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
                    YK_LOGE("[FdkZSlabReconstructor] FilterProcessor reinit failed");
                    return false;
                }
            }
            return true;
        }

    private:

        struct SlabContext {
            int z_start = 0;
            int z_count = 0;
        };

        void accumulateSlab_(int z_start, int z_count, int iVX, int iVY, bool is_first)
        {
            const size_t slab_elems = (size_t)iVX * iVY * z_count;
            const size_t z_offset = (size_t)iVX * iVY * z_start;
            float* dst = h_vol_buffer_ + z_offset;
            const float* src = h_vol_slab_buffer_.data();

            if (is_first)
                std::memcpy(dst, src, slab_elems * sizeof(float));
            else
                for (size_t i = 0; i < slab_elems; ++i)
                    dst[i] += src[i];
        }

        int          Kchunk_ = 0;
        bool         bParker_ = false;
        bool         is_initialized_ = false;
        int          total_received_ = 0;
        int          z_block_size_ = 0;

        SCBCTParams  params_{};
        cudaStream_t stream_ = nullptr;

        Fdk::PreweightProcessor    pw_;
        Fdk::ParkerWeightProcessor pkw_;
        Fdk::FilterProcessor       flt_;
        FdkGpuContext              gpu_ctx_;

        float* d_vol_slab_ = nullptr;

        std::vector<SlabContext>                        slabs_;
        std::vector<std::unique_ptr<Fdk::BpProcessor>>  slab_bps_;  // 指针可移动，对象原地不动
        float* h_vol_buffer_ = nullptr;  // 不持有，指向外部
        std::vector<float>                              h_vol_slab_buffer_;
    };

    // ================================================================
    // 便携函数
    // ================================================================
    YK_INLINE bool fdk_zslab_recon(
        const float* h_proj,
        float* h_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        int                z_block_size)
    {
        FdkZSlabReconstructor recon;
        if (!recon.init(params, Kchunk, stream, z_block_size))
            return false;
        return recon.feed(h_proj, params, h_vol_out, stream);
    }

} // namespace YK