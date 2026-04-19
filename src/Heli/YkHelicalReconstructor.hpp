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

} // namespace YK