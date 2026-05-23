#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "FDK/YkBackProjectProcessor.hpp"
#include "FDK/YkFDKFilterProcessor.hpp"
#include "FDK/YkFDKGpuContext.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"
#include "FDK/YkFDKPreWeightProcessor.hpp"
#include "FDK/YkFdkPipelineContext.hpp"
#include "Heli/YkHelicalGeo.hpp"
#include "Heli/YkHeliCTParams.h"
#include "Heli/YkHelicalParkerProcessor.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "util/YkCudaTimer.hpp"

namespace YK {

    class HelicalReconstructor {
    public:

        bool init(const SHeliCTParam& param,
            const std::vector<SConeProjGeomVec>& geo,
            cudaStream_t                          stream,
            int                                   device_id = 0)
        {
            param_ = param;
            geo_ = geo;
            stream_ = stream;
            device_id_ = device_id;

            const float z_vol_half =
                param.iVZ * param.vox_z_mm * 0.5f;
            z_vol_start_ = param.vol_offset_z_mm - z_vol_half;
            z_vol_end_ = param.vol_offset_z_mm + z_vol_half;

            slabs_ = buildHelicalSlabs(
                param_, computeViewHalf_(), geo_);

            if (slabs_.empty()) {
                YK_LOGE("[HelicalReconstructor] no slabs generated");
                return false;
            }

            const size_t vol_elems =
                (size_t)param.iVX * param.iVY * param.iVZ;
            h_vol_accum_.assign(vol_elems, 0.f);
            h_weight_accum_.assign(vol_elems, 0.f);

            if (!initPipeline_()) return false;

            is_initialized_ = true;
            YK_LOGI("[HelicalReconstructor] init OK: "
                "{}x{}x{} vox  {} slabs  "
                "z_block={:.1f}mm  z_step={:.1f}mm  "
                "parker={}",
                param.iVX, param.iVY, param.iVZ,
                (int)slabs_.size(),
                param.z_block_mm, param.z_step_mm,
                param.bShortScan ? "on" : "off");
            return true;
        }

        void release()
        {
            pw_.release();
            flt_.release();
            hpw_.release();
            gpu_ctx_.release();

            geo_.clear();
            slabs_.clear();
            h_vol_accum_.clear();
            h_weight_accum_.clear();

            is_initialized_ = false;
            pipeline_initialized_ = false;
        }

        bool reconstruct(const float* h_proj,
            float* h_vol_out,
            cudaStream_t stream)
        {
            if (!is_initialized_) {
                YK_LOGE("[HelicalReconstructor] not initialized");
                return false;
            }

            const int    iPU = param_.iPU;
            const int    iPV = param_.iPV;
            const int    iVX = param_.iVX;
            const int    iVY = param_.iVY;
            const size_t view_elems = (size_t)iPU * iPV;
            const size_t vol_elems =
                (size_t)iVX * param_.iVY * param_.iVZ;

            std::fill(h_vol_accum_.begin(), h_vol_accum_.end(), 0.f);
            std::fill(h_weight_accum_.begin(), h_weight_accum_.end(), 0.f);

            int max_z_count = 0;
            for (auto& s : slabs_)
                max_z_count = std::max(max_z_count, s.z_count_vox);

            float* d_vol_slab = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_vol_slab,
                (size_t)iVX * iVY * max_z_count * sizeof(float)));

            for (int si = 0; si < (int)slabs_.size(); ++si) {
                const auto& slab = slabs_[si];
                const int   K = (int)slab.views.indices.size();

                YK_LOGI("[helical recon] slab {}/{} "
                    "z0={:.2f}mm views={}  "
                    "angle=[{:.1f},{:.1f}]deg",
                    si + 1, (int)slabs_.size(),
                    slab.z_center, K,
                    slab.views.angles.front() * 180.f / CUDA_PI,
                    slab.views.angles.back() * 180.f / CUDA_PI);

                // 提取该段投影
                std::vector<float> h_proj_slab(K * view_elems);
                for (int i = 0; i < K; ++i) {
                    std::memcpy(
                        h_proj_slab.data() + (size_t)i * view_elems,
                        h_proj + (size_t)slab.views.indices[i] * view_elems,
                        view_elems * sizeof(float));
                }

                // 从 geo_ 取该段几何
                std::vector<SConeProjGeomVec>    h_geo(K);
                std::vector<SFDKGeoParamPerView> h_gv(K);
                for (int i = 0; i < K; ++i)
                    h_geo[i] = geo_[slab.views.indices[i]];

                GeoDerivedManagerVec{}.build_geo_params(
                    iPU, iPV,
                    slab.views.angles.back() - slab.views.angles.front(),
                    h_geo, h_gv);

                gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, 0, K, stream);

                // BpProcessor
                SVolGeom sub_geom = SVolGeom::make_centered(
                    iVX, iVY, slab.z_count_vox,
                    param_.vox_x_mm, param_.vox_z_mm);
                sub_geom.center = make_float3(
                    param_.vol_offset_x_mm,
                    param_.vol_offset_y_mm,
                    slab.z_center);

                auto bp = std::make_unique<Fdk::BpProcessor>();
                {
                    BpInitContext bctx{};
                    bctx.vol_geom = sub_geom;
                    bctx.use_precomputed = true;
                    bp->setInitContext(&bctx);
                    if (!bp->init()) {
                        YK_LOGE("[HelicalReconstructor] "
                            "BpProcessor init failed slab {}", si);
                        cudaFree(d_vol_slab);
                        return false;
                    }
                }

                // Parker reinit
                if (param_.bShortScan) {
                    if (!reinitHelicalParker_(iPU, iPV,
                        std::min(param_.Kchunk, K))) {
                        cudaFree(d_vol_slab);
                        return false;
                    }
                }

                const size_t slab_elems =
                    (size_t)iVX * iVY * slab.z_count_vox;
                YK_CUDA_CHECK(cudaMemsetAsync(
                    d_vol_slab, 0, slab_elems * sizeof(float), stream));

                float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
                float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
                float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

                bool filter_dirty = false;

                for (int base = 0; base < K; base += param_.Kchunk) {
                    const int Kc = std::min(param_.Kchunk, K - base);

                    if (Kc != param_.Kchunk) {
                        if (!reinitChunkProcessors_(iPU, iPV, Kc, stream)) {
                            cudaFree(d_vol_slab);
                            return false;
                        }
                        filter_dirty = true;
                    }

                    gpu_ctx_.proj.uploadProjChunk(
                        h_proj_slab.data() + (size_t)base * view_elems,
                        Kc, stream);
                    gpu_ctx_.geo.uploadCoeffsChunk(
                        gpu_ctx_.geo.d_coeffs() + base, Kc, stream);

                    // preweight
                    PreweightChunkContext pctx{};
                    pctx.d_geo = gpu_ctx_.geo.d_geo() + base;
                    pctx.d_gv = gpu_ctx_.geo.d_gv() + base;
                    pctx.K = Kc;
                    pw_.setContext(&pctx);
                    pw_.process(d_chunk_in, d_chunk_pw, stream);

                    // Parker
                    if (param_.bShortScan) {
                        Helical::HelicalParkerChunkContext hpkctx{};
                        hpkctx.h_angles =
                            slab.views.angles.data() + base;
                        hpkctx.K = Kc;
                        hpkctx.angle_base =
                            slab.views.angles.front();
                        hpw_.setContext(&hpkctx);
                        hpw_.process(d_chunk_pw, stream);
                    }

                    // filter
                    FdkFilterContext fctx{ h_gv.data() + base, Kc };
                    flt_.setContext(&fctx);
                    flt_.process(d_chunk_pw, d_chunk_flt, stream);

                    // backproject
                    BpChunkContext bctx{};
                    bctx.d_geo = gpu_ctx_.geo.d_geo() + base;
                    bctx.d_gv = gpu_ctx_.geo.d_gv() + base;
                    bctx.K = Kc;
                    bp->setContext(&bctx);
                    bp->process(gpu_ctx_.proj.d_texObjs(),
                        d_vol_slab, stream);
                }

                if (filter_dirty)
                    reinitChunkProcessors_(
                        iPU, iPV, param_.Kchunk, stream);

                std::vector<float> h_slab(slab_elems);
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                YK_CUDA_CHECK(cudaMemcpy(
                    h_slab.data(), d_vol_slab,
                    slab_elems * sizeof(float),
                    cudaMemcpyDeviceToHost));

                mergeSlabToVolume_(h_slab.data(), slab);
                bp->release();
            }

            cudaFree(d_vol_slab);

            for (size_t i = 0; i < vol_elems; ++i) {
                h_vol_out[i] = (h_weight_accum_[i] > 1e-6f)
                    ? h_vol_accum_[i] / h_weight_accum_[i]
                    : 0.f;
            }

            return true;
        }

        int   totalSlabs() const { return (int)slabs_.size(); }
        float zVolStart()  const { return z_vol_start_; }
        float zVolEnd()    const { return z_vol_end_; }

    private:

        float computeViewHalf_() const
        {
            if (param_.bShortScan) {
                const float half_fan = std::atan(
                    param_.iPU * 0.5f * param_.du_mm / param_.SDD);
                return (CUDA_PI + 2.f * half_fan) * 0.5f;
            }
            return CUDA_PI;
        }

        bool initPipeline_()
        {
            const int iPU = param_.iPU;
            const int iPV = param_.iPV;

            int max_K = 0;
            for (auto& s : slabs_)
                max_K = std::max(max_K, (int)s.views.indices.size());

            const int chunk = std::min(param_.Kchunk, max_K);

            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_LOGE("[HelicalReconstructor] pw init failed");
                    return false;
                }
            }

            if (param_.bShortScan) {
                if (!reinitHelicalParker_(iPU, iPV, chunk))
                    return false;
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.desc = param_.desc;
                ictx.policy = {};
                ictx.stream = stream_;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    YK_LOGE("[HelicalReconstructor] flt init failed");
                    return false;
                }
            }

            {
                SProjDims dims{ iPU, iPV, max_K };
                gpu_ctx_.init(dims, max_K, stream_, device_id_);
            }

            pipeline_initialized_ = true;
            return true;
        }

        bool reinitHelicalParker_(int iPU, int iPV, int K)
        {
            Helical::HelicalParkerInitContext hpctx{};
            hpctx.dims = SProjDims{ iPU, iPV, K };
            hpctx.fDetUSize = param_.du_mm;
            hpctx.fSrcOrigin = param_.SID;
            hpctx.fDetOrigin = param_.SDD - param_.SID;
            hpw_.setInitContext(&hpctx);
            if (!hpw_.init()) {
                YK_LOGE("[HelicalReconstructor] Parker init failed");
                return false;
            }
            return true;
        }

        bool reinitChunkProcessors_(int iPU, int iPV, int K,
            cudaStream_t stream)
        {
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) return false;
            }

            if (param_.bShortScan) {
                if (!reinitHelicalParker_(iPU, iPV, K))
                    return false;
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.desc = param_.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) return false;
            }

            return true;
        }

        void mergeSlabToVolume_(const float* h_slab,
            const HelicalSlabConfig& slab)
        {
            const int iVX = param_.iVX;
            const int iVY = param_.iVY;

            const float z_slab_start =
                slab.z_center - param_.z_block_mm * 0.5f;
            const float z_slab_end =
                slab.z_center + param_.z_block_mm * 0.5f;

            for (int iz = 0; iz < slab.z_count_vox; ++iz) {
                const int vol_iz = slab.z_start_vox + iz;
                if (vol_iz < 0 || vol_iz >= param_.iVZ) continue;

                const float z_world = z_vol_start_
                    + (vol_iz + 0.5f) * param_.vox_z_mm;

                const float t = (z_world - z_slab_start)
                    / (z_slab_end - z_slab_start);
                const float cw = cosf(CUDA_PI * (t - 0.5f));
                const float w2 = cw * cw;

                const size_t slab_z_off = (size_t)iz * iVX * iVY;
                const size_t vol_z_off = (size_t)vol_iz * iVX * iVY;

                for (size_t xy = 0; xy < (size_t)iVX * iVY; ++xy) {
                    h_vol_accum_[vol_z_off + xy] +=
                        w2 * h_slab[slab_z_off + xy];
                    h_weight_accum_[vol_z_off + xy] += w2;
                }
            }
        }

        bool         is_initialized_ = false;
        bool         pipeline_initialized_ = false;

        SHeliCTParam  param_{};
        cudaStream_t  stream_ = nullptr;
        int           device_id_ = 0;

        float z_vol_start_ = 0.f;
        float z_vol_end_ = 0.f;

        std::vector<SConeProjGeomVec>  geo_;
        std::vector<HelicalSlabConfig> slabs_;

        Fdk::PreweightProcessor         pw_;
        Helical::HelicalParkerProcessor hpw_;
        Fdk::FilterProcessor            flt_;
        FdkGpuContext                   gpu_ctx_;

        std::vector<float> h_vol_accum_;
        std::vector<float> h_weight_accum_;
    };

}; // namespace YK


namespace YK {

    // ============================================================
//  HelicalOnlineReconstructor
//
//  边扫描边重建：投影分批送入，积累到足够一个 slab 时立即重建。
//
//  使用流程：
//    1. init()          — 用标称参数提前建 slabs + 初始化 pipeline
//    2. feedBatch()     — 每批投影 + 实测角度，可多次调用
//    3. feedDone()      — 通知数据送完，flush 尾部不完整 slab
//    4. getVolume()     — 归一化后拷贝到输出缓冲
//    5. reset()         — 清空积累状态，复用 pipeline（下次扫描）
// ============================================================
    class HelicalOnlineReconstructor {
    public:

        // --------------------------------------------------------
        //  init
        //  param      : 螺旋CT参数（含标称角速度，用于 buildHelicalSlabs）
        //  stream     : CUDA stream
        //  device_id  : GPU device
        // --------------------------------------------------------
        bool init(const SHeliCTParam& param,
            cudaStream_t        stream,
            int                 device_id = 0)
        {
            param_ = param;
            stream_ = stream;
            device_id_ = device_id;

            const float z_vol_half =
                param_.iVZ * param_.vox_z_mm * 0.5f;
            z_vol_start_ = param_.vol_offset_z_mm - z_vol_half;
            z_vol_end_ = param_.vol_offset_z_mm + z_vol_half;

            // 用标称几何（匀速假设）提前建 slabs
            // buildHelicalSlabs 只需要参数和标称几何，不依赖实测角度
            const float view_half = computeViewHalf_();
            slabs_ = buildHelicalSlabsFromParam_(param_, view_half);

            if (slabs_.empty()) {
                YK_LOGE("[HelicalOnlineReconstructor] no slabs generated");
                return false;
            }

            // 累积体积缓冲
            const size_t vol_elems =
                (size_t)param_.iVX * param_.iVY * param_.iVZ;
            h_vol_accum_.assign(vol_elems, 0.f);
            h_weight_accum_.assign(vol_elems, 0.f);

            // 初始化 GPU pipeline
            if (!initPipeline_()) return false;

            // 预分配投影积累缓冲（预留 2 圈容量）
            const size_t view_elems = (size_t)param_.iPU * param_.iPV;
            const size_t reserve_views =
                static_cast<size_t>(param_.views_per_rot) * 2;
            proj_buf_.reserve(view_elems * reserve_views);
            geo_buf_.reserve(reserve_views);

            is_initialized_ = true;

            YK_LOGI("[HelicalOnlineReconstructor] init OK: "
                "{}x{}x{} vox  {} slabs  "
                "z_block={:.1f}mm  z_step={:.1f}mm  parker={}",
                param_.iVX, param_.iVY, param_.iVZ,
                (int)slabs_.size(),
                param_.z_block_mm, param_.z_step_mm,
                param_.bShortScan ? "on" : "off");
            return true;
        }

        // --------------------------------------------------------
        //  feedBatch
        //  h_proj    : [Kbatch * iPU * iPV] float，行优先连续存储
        //  h_angles  : [Kbatch] float，每帧实测旋转角度（rad）
        //  Kbatch    : 本批视角数
        // --------------------------------------------------------
        void feedBatch(const float* h_proj,
            const float* h_angles,
            int          Kbatch)
        {
            if (!is_initialized_) {
                YK_LOGE("[HelicalOnlineReconstructor] not initialized");
                return;
            }

            const size_t view_elems = (size_t)param_.iPU * param_.iPV;

            // 1. 追加投影到缓冲
            size_t old_proj_size = proj_buf_.size();
            proj_buf_.resize(old_proj_size + (size_t)Kbatch * view_elems);
            std::memcpy(proj_buf_.data() + old_proj_size,
                h_proj,
                Kbatch * view_elems * sizeof(float));

            // 2. 用实测角度 + 系统参数构造每帧精确几何
            //    build_helical_vec_geometry 需要 angle_list 在 param 里，
            //    用一个临时 param 传入本批角度
            {
                SHeliCTParam tmp = param_;
                tmp.angle_list.assign(h_angles, h_angles + Kbatch);
                // start_z_mm 需要反映当前已接收帧的起始z
                // src_z = start_z_mm + pitch * angle / (2π)
                // 这里 start_z_mm 保持 param_ 原值，
                // build_helical_vec_geometry 内部会用 angle_list[i] 推算每帧z
                std::vector<SConeProjGeomVec> batch_geo;
                build_helical_vec_geometry(batch_geo, tmp);
                for (auto& g : batch_geo)
                    geo_buf_.push_back(g);
            }

            total_received_ += Kbatch;

            // 3. 检查并触发就绪 slab
            checkAndDispatch_(/*force=*/false);

            // 4. 裁剪已无用的缓冲区
            trimBuffer_();
        }

        // --------------------------------------------------------
        //  feedDone
        //  所有投影已送完，强制 flush 剩余 slab（头尾不完整也重建）
        // --------------------------------------------------------
        void feedDone()
        {
            if (!is_initialized_) return;

            checkAndDispatch_(/*force=*/true);
            trimBuffer_();

            YK_LOGI("[HelicalOnlineReconstructor] feedDone: "
                "{}/{} slabs reconstructed",
                next_slab_, (int)slabs_.size());
        }

        // --------------------------------------------------------
        //  getVolume
        //  归一化后写入 h_vol_out（调用方负责分配，大小 iVX*iVY*iVZ）
        // --------------------------------------------------------
        bool getVolume(float* h_vol_out) const
        {
            if (!is_initialized_) {
                YK_LOGE("[HelicalOnlineReconstructor] not initialized");
                return false;
            }

            const size_t vol_elems =
                (size_t)param_.iVX * param_.iVY * param_.iVZ;

            for (size_t i = 0; i < vol_elems; ++i) {
                h_vol_out[i] = (h_weight_accum_[i] > 1e-6f)
                    ? h_vol_accum_[i] / h_weight_accum_[i]
                    : 0.f;
            }
            return true;
        }

        // --------------------------------------------------------
        //  reset
        //  清空积累状态，pipeline 保持初始化，可直接开始下次扫描
        // --------------------------------------------------------
        void reset()
        {
            const size_t vol_elems =
                (size_t)param_.iVX * param_.iVY * param_.iVZ;
            std::fill(h_vol_accum_.begin(), h_vol_accum_.end(), 0.f);
            std::fill(h_weight_accum_.begin(), h_weight_accum_.end(), 0.f);

            proj_buf_.clear();
            geo_buf_.clear();
            total_received_ = 0;
            buf_start_ = 0;
            next_slab_ = 0;

            YK_LOGI("[HelicalOnlineReconstructor] reset");
        }

        // --------------------------------------------------------
        //  release
        // --------------------------------------------------------
        void release()
        {
            pw_.release();
            flt_.release();
            hpw_.release();
            gpu_ctx_.release();

            slabs_.clear();
            proj_buf_.clear();
            geo_buf_.clear();
            h_vol_accum_.clear();
            h_weight_accum_.clear();

            is_initialized_ = false;
            pipeline_initialized_ = false;
        }

        // 查询状态
        int  totalSlabs()     const { return (int)slabs_.size(); }
        int  reconstructedSlabs() const { return next_slab_; }
        int  receivedViews()  const { return total_received_; }
        bool isInitialized()  const { return is_initialized_; }

    private:

        // --------------------------------------------------------
        //  用标称几何（angle_list）提前建 slabs
        //  build_helical_vec_geometry 内部处理所有倾斜/偏移/螺旋z平移
        // --------------------------------------------------------
        std::vector<HelicalSlabConfig> buildHelicalSlabsFromParam_(
            const SHeliCTParam& param, float view_half)
        {
            if (param.angle_list.empty()) {
                YK_LOGE("[HelicalOnlineReconstructor] "
                    "param.angle_list is empty");
                return {};
            }

            std::vector<SConeProjGeomVec> nominal_geo;
            build_helical_vec_geometry(nominal_geo, param);

            return buildHelicalSlabs(param, view_half, nominal_geo);
        }

        float computeViewHalf_() const
        {
            if (param_.bShortScan) {
                const float half_fan = std::atan(
                    param_.iPU * 0.5f * param_.du_mm / param_.SDD);
                return (CUDA_PI + 2.f * half_fan) * 0.5f;
            }
            return CUDA_PI;
        }

        // --------------------------------------------------------
        //  检查 slab 就绪并触发重建
        //  force=true 时强制触发所有剩余 slab（feedDone 调用）
        // --------------------------------------------------------
        void checkAndDispatch_(bool force)
        {
            while (next_slab_ < (int)slabs_.size()) {
                const auto& slab = slabs_[next_slab_];

                // 判断就绪：该 slab 需要的最后一个视角已收到
                const int need_up_to = slab.views.indices.back();
                const bool ready = (total_received_ - 1 >= need_up_to);

                if (ready || force) {
                    YK_LOGI("[HelicalOnlineReconstructor] "
                        "slab {}/{} dispatching  "
                        "z={:.2f}mm  views={}  "
                        "angle=[{:.1f},{:.1f}]deg",
                        next_slab_ + 1, (int)slabs_.size(),
                        slab.z_center,
                        (int)slab.views.indices.size(),
                        slab.views.angles.front() * 180.f / CUDA_PI,
                        slab.views.angles.back() * 180.f / CUDA_PI);

                    reconstructSlab_(next_slab_);
                    ++next_slab_;
                }
                else {
                    break;  // 后续 slab 视角索引更大，也不会就绪
                }
            }
        }

        // --------------------------------------------------------
        //  重建单个 slab 并 merge 到累积体积
        // --------------------------------------------------------
        void reconstructSlab_(int si)
        {
            const auto& slab = slabs_[si];
            const int    K = (int)slab.views.indices.size();
            const int    iPU = param_.iPU;
            const int    iPV = param_.iPV;
            const size_t view_elems = (size_t)iPU * iPV;

            // --- 提取该 slab 的投影和几何 ---
            std::vector<float>            h_proj_slab(K * view_elems);
            std::vector<SConeProjGeomVec> h_geo(K);

            for (int i = 0; i < K; ++i) {
                const int global_idx = slab.views.indices[i];
                const int local_idx = global_idx - buf_start_;

                std::memcpy(
                    h_proj_slab.data() + (size_t)i * view_elems,
                    proj_buf_.data() + (size_t)local_idx * view_elems,
                    view_elems * sizeof(float));

                h_geo[i] = geo_buf_[local_idx];
            }

            // --- 构造每帧派生几何参数 ---
            std::vector<SFDKGeoParamPerView> h_gv(K);
            const float angle_span =
                slab.views.angles.back() - slab.views.angles.front();
            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, angle_span, h_geo, h_gv);

            // --- 上传几何到 GPU ---
            gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, 0, K, stream_);

            // --- 构造子体积几何 ---
            SVolGeom sub_geom = SVolGeom::make_centered(
                param_.iVX, param_.iVY, slab.z_count_vox,
                param_.vox_x_mm, param_.vox_z_mm);
            sub_geom.center = make_float3(
                param_.vol_offset_x_mm,
                param_.vol_offset_y_mm,
                slab.z_center);

            // --- 初始化 BpProcessor ---
            auto bp = std::make_unique<Fdk::BpProcessor>();
            {
                BpInitContext bctx{};
                bctx.vol_geom = sub_geom;
                bctx.use_precomputed = true;
                bp->setInitContext(&bctx);
                if (!bp->init()) {
                    YK_LOGE("[HelicalOnlineReconstructor] "
                        "BpProcessor init failed slab {}", si);
                    return;
                }
            }

            // --- Parker reinit（该 slab 实际 K 可能与 chunk 不同）---
            if (param_.bShortScan) {
                reinitHelicalParker_(iPU, iPV,
                    std::min(param_.Kchunk, K));
            }

            // --- 分配 slab 体积 GPU buffer ---
            const size_t slab_elems =
                (size_t)param_.iVX * param_.iVY * slab.z_count_vox;
            float* d_vol_slab = nullptr;
            YK_CUDA_CHECK(cudaMalloc(
                &d_vol_slab, slab_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMemsetAsync(
                d_vol_slab, 0, slab_elems * sizeof(float), stream_));

            float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
            float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
            float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

            bool filter_dirty = false;

            // --- chunk 循环 ---
            for (int base = 0; base < K; base += param_.Kchunk) {
                const int Kc = std::min(param_.Kchunk, K - base);

                if (Kc != param_.Kchunk) {
                    reinitChunkProcessors_(iPU, iPV, Kc, stream_);
                    filter_dirty = true;
                }

                gpu_ctx_.proj.uploadProjChunk(
                    h_proj_slab.data() + (size_t)base * view_elems,
                    Kc, stream_);
                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + base, Kc, stream_);

                // preweight
                PreweightChunkContext pctx{};
                pctx.d_geo = gpu_ctx_.geo.d_geo() + base;
                pctx.d_gv = gpu_ctx_.geo.d_gv() + base;
                pctx.K = Kc;
                pw_.setContext(&pctx);
                pw_.process(d_chunk_in, d_chunk_pw, stream_);

                // Parker
                if (param_.bShortScan) {
                    Helical::HelicalParkerChunkContext hpkctx{};
                    hpkctx.h_angles = slab.views.angles.data() + base;
                    hpkctx.K = Kc;
                    hpkctx.angle_base = slab.views.angles.front();
                    hpw_.setContext(&hpkctx);
                    hpw_.process(d_chunk_pw, stream_);
                }

                // filter
                FdkFilterContext fctx{ h_gv.data() + base, Kc };
                flt_.setContext(&fctx);
                flt_.process(d_chunk_pw, d_chunk_flt, stream_);

                // backproject
                BpChunkContext bctx{};
                bctx.d_geo = gpu_ctx_.geo.d_geo() + base;
                bctx.d_gv = gpu_ctx_.geo.d_gv() + base;
                bctx.K = Kc;
                bp->setContext(&bctx);
                bp->process(gpu_ctx_.proj.d_texObjs(),
                    d_vol_slab, stream_);
            }

            // 恢复 chunk 大小
            if (filter_dirty)
                reinitChunkProcessors_(
                    iPU, iPV, param_.Kchunk, stream_);

            // D2H
            std::vector<float> h_slab(slab_elems);
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
            YK_CUDA_CHECK(cudaMemcpy(
                h_slab.data(), d_vol_slab,
                slab_elems * sizeof(float),
                cudaMemcpyDeviceToHost));

            cudaFree(d_vol_slab);
            bp->release();

            // merge 到累积体积
            mergeSlabToVolume_(h_slab.data(), slab);
        }

        // --------------------------------------------------------
        //  裁剪缓冲区：丢弃所有未触发 slab 都不再需要的早期投影
        // --------------------------------------------------------
        void trimBuffer_()
        {
            // 找未触发 slab 中最早需要的全局视角索引
            int min_needed = total_received_;
            for (int si = next_slab_; si < (int)slabs_.size(); ++si) {
                if (!slabs_[si].views.indices.empty()) {
                    min_needed = std::min(
                        min_needed, slabs_[si].views.indices.front());
                }
            }

            const int drop = min_needed - buf_start_;
            if (drop <= 0) return;

            const size_t view_elems = (size_t)param_.iPU * param_.iPV;
            proj_buf_.erase(
                proj_buf_.begin(),
                proj_buf_.begin() + (size_t)drop * view_elems);
            geo_buf_.erase(
                geo_buf_.begin(),
                geo_buf_.begin() + drop);

            buf_start_ += drop;
        }

        // --------------------------------------------------------
        //  mergeSlabToVolume  （逻辑与 HelicalReconstructor 完全一致）
        // --------------------------------------------------------
        void mergeSlabToVolume_(const float* h_slab,
            const HelicalSlabConfig& slab)
        {
            const int iVX = param_.iVX;
            const int iVY = param_.iVY;

            const float z_slab_start =
                slab.z_center - param_.z_block_mm * 0.5f;
            const float z_slab_end =
                slab.z_center + param_.z_block_mm * 0.5f;

            for (int iz = 0; iz < slab.z_count_vox; ++iz) {
                const int vol_iz = slab.z_start_vox + iz;
                if (vol_iz < 0 || vol_iz >= param_.iVZ) continue;

                const float z_world = z_vol_start_
                    + (vol_iz + 0.5f) * param_.vox_z_mm;

                const float t = (z_world - z_slab_start)
                    / (z_slab_end - z_slab_start);
                const float cw = cosf(CUDA_PI * (t - 0.5f));
                const float w2 = cw * cw;

                const size_t slab_z_off = (size_t)iz * iVX * iVY;
                const size_t vol_z_off = (size_t)vol_iz * iVX * iVY;

                for (size_t xy = 0; xy < (size_t)iVX * iVY; ++xy) {
                    h_vol_accum_[vol_z_off + xy] +=
                        w2 * h_slab[slab_z_off + xy];
                    h_weight_accum_[vol_z_off + xy] += w2;
                }
            }
        }

        // --------------------------------------------------------
        //  Pipeline 初始化（只在 init 时调用一次）
        // --------------------------------------------------------
        bool initPipeline_()
        {
            const int iPU = param_.iPU;
            const int iPV = param_.iPV;

            int max_K = 0;
            for (auto& s : slabs_)
                max_K = std::max(max_K, (int)s.views.indices.size());

            const int chunk = std::min(param_.Kchunk, max_K);

            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_LOGE("[HelicalOnlineReconstructor] pw init failed");
                    return false;
                }
            }

            if (param_.bShortScan) {
                if (!reinitHelicalParker_(iPU, iPV, chunk))
                    return false;
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.desc = param_.desc;
                ictx.policy = {};
                ictx.stream = stream_;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    YK_LOGE("[HelicalOnlineReconstructor] flt init failed");
                    return false;
                }
            }

            {
                SProjDims dims{ iPU, iPV, max_K };
                gpu_ctx_.init(dims, max_K, stream_, device_id_);
            }

            pipeline_initialized_ = true;
            return true;
        }

        bool reinitHelicalParker_(int iPU, int iPV, int K)
        {
            Helical::HelicalParkerInitContext hpctx{};
            hpctx.dims = SProjDims{ iPU, iPV, K };
            hpctx.fDetUSize = param_.du_mm;
            hpctx.fSrcOrigin = param_.SID;
            hpctx.fDetOrigin = param_.SDD - param_.SID;
            hpw_.setInitContext(&hpctx);
            if (!hpw_.init()) {
                YK_LOGE("[HelicalOnlineReconstructor] Parker init failed");
                return false;
            }
            return true;
        }

        bool reinitChunkProcessors_(int iPU, int iPV, int K,
            cudaStream_t stream)
        {
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) return false;
            }

            if (param_.bShortScan) {
                if (!reinitHelicalParker_(iPU, iPV, K))
                    return false;
            }

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.desc = param_.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) return false;
            }

            return true;
        }

        // --------------------------------------------------------
        //  成员变量
        // --------------------------------------------------------
        bool         is_initialized_ = false;
        bool         pipeline_initialized_ = false;

        SHeliCTParam  param_{};
        cudaStream_t  stream_ = nullptr;
        int           device_id_ = 0;

        float z_vol_start_ = 0.f;
        float z_vol_end_ = 0.f;

        std::vector<HelicalSlabConfig> slabs_;

        // 投影积累缓冲（滑动窗口）
        std::vector<float>            proj_buf_;   // [local_views * iPU * iPV]
        std::vector<SConeProjGeomVec> geo_buf_;    // [local_views]
        int buf_start_ = 0;   // proj_buf_[0] 对应的全局视角索引
        int total_received_ = 0;   // 累计收到的全局视角数
        int next_slab_ = 0;   // 下一个待触发的 slab 索引

        // 累积体积
        std::vector<float> h_vol_accum_;
        std::vector<float> h_weight_accum_;

        // GPU pipeline 组件（init 时建好，reset 后复用）
        Fdk::PreweightProcessor         pw_;
        Helical::HelicalParkerProcessor hpw_;
        Fdk::FilterProcessor            flt_;
        FdkGpuContext                   gpu_ctx_;
    };

};