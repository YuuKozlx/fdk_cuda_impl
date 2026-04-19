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
#include "FP/YkFpRunnerExVec.hpp"
#include "YkHelicalGeo.hpp"
#include "YkHelicalParkerProcessor.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "util/YkCudaTimer.hpp"

namespace YK {

    class HelicalReconstructor {
    public:

        struct SHeliFpConfig {
            // scan config
            float pitch_mm = 3.f; // 仿真螺旋进给量
            float start_z_mm = 0.f;  // auto_start_z=true 时该值自动计算，被覆盖，表示扫描起始位置
            bool  auto_start_z = true;  // 自动计算扫描起始 Z

            int   views_per_rot = 360;// 每圈投影数
            int   n_rotations = 0;   // 自动计算圈数
            ETask fp_task = ETask::FP_Joseph;


            float z_block_mm = 3.f;  // 每段重建厚度 = pitch
            float z_step_mm = 1.f;   // 段间步进，50% 重叠
            int   Kchunk = 32;
        };

        bool init(const SCBCTParams& params,
            const SHeliFpConfig& cfg,
            cudaStream_t stream,
            int device_id = 0)
        {
            params_ = params;
            cfg_ = cfg;
            stream_ = stream;
            device_id_ = device_id;

            const float z_vol_half = params.iVZ * params.vox_z_mm * 0.5f;
            z_vol_start_ = params.vol_offset_z_mm - z_vol_half;
            z_vol_end_ = params.vol_offset_z_mm + z_vol_half;

            margin_ = computeMargin_();

            if (cfg_.auto_start_z)
                cfg_.start_z_mm = z_vol_start_ - margin_ - cfg_.pitch_mm;

            if (cfg_.n_rotations <= 0) {
                const float z_scan_end = z_vol_end_ + margin_ + cfg_.pitch_mm;
                const float z_total = z_scan_end - cfg_.start_z_mm;
                cfg_.n_rotations =
                    (int)std::ceil(z_total / cfg_.pitch_mm) + 1;
            }

            const int total_views = cfg_.n_rotations * cfg_.views_per_rot;
            angle_list_.resize(total_views);
            for (int i = 0; i < total_views; ++i)
                angle_list_[i] = i * 2.f * CUDA_PI / cfg_.views_per_rot;

            helical_geo_.clear();
            build_helical_vec_geometry_from_theta(
                helical_geo_, angle_list_, total_views,
                params.iPU, params.iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                cfg_.pitch_mm, cfg_.start_z_mm);

            // parker_half：π+2γ 的一半，用于投影选取
            const float half_fan = std::atan(
                params.iPU * 0.5f * params.du_mm / params.SDD);
            const float parker_half = params.bShortScan
                ? (CUDA_PI + 2.f * half_fan) * 0.5f  // π+2γ，约195°
                : CUDA_PI;

            // buildHelicalSlabs 只传 geo，信息源唯一
            slabs_ = buildHelicalSlabs(
                z_vol_start_, z_vol_end_,
                cfg_.z_step_mm, cfg_.z_block_mm,
                params.vox_z_mm,
                computeViewSelectionHalf_(),
                helical_geo_);

            if (slabs_.empty()) {
                YK_LOGE("[HelicalReconstructor] no slabs generated");
                return false;
            }

            const size_t vol_elems =
                (size_t)params.iVX * params.iVY * params.iVZ;
            h_vol_accum_.assign(vol_elems, 0.f);
            h_weight_accum_.assign(vol_elems, 0.f);

            if (!initPipeline_()) return false;

            is_initialized_ = true;
            YK_LOGI("[HelicalReconstructor] init OK: "
                "{}x{}x{} vox  {} slabs  {} views  "
                "pitch={:.1f}mm  start_z={:.1f}mm  margin={:.2f}mm  "
                "parker_half={:.2f}deg",
                params.iVX, params.iVY, params.iVZ,
                (int)slabs_.size(), total_views,
                cfg_.pitch_mm, cfg_.start_z_mm, margin_,
                parker_half * 180.f / CUDA_PI);
            return true;
        }

        void release()
        {
            pw_.release();
            flt_.release();
            hpw_.release();
            gpu_ctx_.release();

            angle_list_.clear();
            helical_geo_.clear();
            slabs_.clear();
            h_vol_accum_.clear();
            h_weight_accum_.clear();

            is_initialized_ = false;
            pipeline_initialized_ = false;
        }

        bool forwardProject(const float* d_vol,
            float* h_proj_out,
            cudaStream_t stream)
        {
            if (!is_initialized_) {
                YK_LOGE("[HelicalReconstructor] not initialized");
                return false;
            }

            const int    total_views = (int)angle_list_.size();
            const size_t view_elems = (size_t)params_.iPU * params_.iPV;
            const size_t proj_elems = view_elems * total_views;

            float* d_proj = nullptr;
            YK_CUDA_CHECK(cudaMalloc(&d_proj, proj_elems * sizeof(float)));
            YK_CUDA_CHECK(cudaMemset(d_proj, 0, proj_elems * sizeof(float)));

            SCBCTParams fp_params = params_;
            fp_params.iPAng = total_views;
            fp_params.iPAngTotal = total_views;
            fp_params.angle_list = angle_list_;

            FpReconstructorEx fp;
            if (!fp.init(fp_params, cfg_.fp_task, device_id_)) {
                cudaFree(d_proj);
                return false;
            }

            {
                Util::CudaTimer timer("helical_fp", stream);
                if (!fp.run(d_vol, fp_params, helical_geo_, d_proj, stream)) {
                    cudaFree(d_proj);
                    return false;
                }
            }

            YK_CUDA_CHECK(cudaStreamSynchronize(stream));
            YK_CUDA_CHECK(cudaMemcpy(h_proj_out, d_proj,
                proj_elems * sizeof(float), cudaMemcpyDeviceToHost));

            cudaFree(d_proj);
            return true;
        }

        bool reconstruct(const float* h_proj,
            float* h_vol_out,
            cudaStream_t stream)
        {
            if (!is_initialized_) {
                YK_LOGE("[HelicalReconstructor] not initialized");
                return false;
            }

            const int    iPU = params_.iPU;
            const int    iPV = params_.iPV;
            const int    iVX = params_.iVX;
            const int    iVY = params_.iVY;
            const size_t view_elems = (size_t)iPU * iPV;
            const size_t vol_elems =
                (size_t)iVX * params_.iVY * params_.iVZ;

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

                YK_LOGI("[helical recon] slab {}/{} z0={:.2f}mm views={}  "
                    "angle=[{:.1f},{:.1f}]deg",
                    si + 1, (int)slabs_.size(),
                    slab.z_center, K,
                    slab.views.angles.front() * 180.f / CUDA_PI,
                    slab.views.angles.back() * 180.f / CUDA_PI);

                // ---- 提取该段投影 ----
                std::vector<float> h_proj_slab(K * view_elems);
                for (int i = 0; i < K; ++i) {
                    std::memcpy(
                        h_proj_slab.data() + (size_t)i * view_elems,
                        h_proj + (size_t)slab.views.indices[i] * view_elems,
                        view_elems * sizeof(float));
                }

                // ---- 构建该段螺旋几何 ----
                std::vector<SConeProjGeomVec>    h_geo(K);
                std::vector<SFDKGeoParamPerView> h_gv(K);

                for (int i = 0; i < K; ++i)
                    h_geo[i] = helical_geo_[slab.views.indices[i]];

                GeoDerivedManagerVec{}.build_geo_params(
                    iPU, iPV,
                    slab.views.angles.back() - slab.views.angles.front(),
                    h_geo, h_gv);

                gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, 0, K, stream);

                // ---- 初始化该段 BpProcessor ----
                SVolGeom sub_geom = SVolGeom::make_centered(
                    iVX, iVY, slab.z_count_vox,
                    params_.vox_x_mm, params_.vox_z_mm);
                sub_geom.center = make_float3(
                    params_.vol_offset_x_mm,
                    params_.vol_offset_y_mm,
                    slab.z_center);

                auto bp = std::make_unique<Fdk::BpProcessor>();
                {
                    BpInitContext bctx{};
                    bctx.vol_geom = sub_geom;
                    bctx.use_precomputed = true;
                    bp->setInitContext(&bctx);
                    if (!bp->init()) {
                        YK_LOGE("[HelicalReconstructor] BpProcessor init failed slab {}", si);
                        cudaFree(d_vol_slab);
                        return false;
                    }
                }

                // ---- 重新初始化螺旋 Parker ----
                if (!reinitHelicalParker_(iPU, iPV,
                    std::min(cfg_.Kchunk, K))) {
                    cudaFree(d_vol_slab);
                    return false;
                }

                const size_t slab_elems =
                    (size_t)iVX * iVY * slab.z_count_vox;
                YK_CUDA_CHECK(cudaMemsetAsync(
                    d_vol_slab, 0, slab_elems * sizeof(float), stream));

                float* d_chunk_in = gpu_ctx_.proj.chunk_in.data();
                float* d_chunk_pw = gpu_ctx_.proj.chunk_pw.data();
                float* d_chunk_flt = gpu_ctx_.proj.chunk_flt.data();

                bool filter_dirty = false;

                for (int base = 0; base < K; base += cfg_.Kchunk) {
                    const int Kc = std::min(cfg_.Kchunk, K - base);

                    if (Kc != cfg_.Kchunk) {
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

                    // helical parker
                    if (params_.bShortScan) {
                        Helical::HelicalParkerChunkContext hpkctx{};
                        hpkctx.h_angles = slab.views.angles.data() + base;
                        hpkctx.K = Kc;
                        hpkctx.angle_base = slab.views.angles.front();
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
                    reinitChunkProcessors_(iPU, iPV, cfg_.Kchunk, stream);

                // 同步 + 回读
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

            // 归一化
            for (size_t i = 0; i < vol_elems; ++i) {
                h_vol_out[i] = (h_weight_accum_[i] > 1e-6f)
                    ? h_vol_accum_[i] / h_weight_accum_[i]
                    : 0.f;
            }

            return true;
        }

        int   totalViews() const { return (int)angle_list_.size(); }
        int   totalSlabs() const { return (int)slabs_.size(); }
        float zVolStart()  const { return z_vol_start_; }
        float zVolEnd()    const { return z_vol_end_; }

    private:

        float computeMargin_() const {
            return (params_.iPV * 0.5f * params_.dv_mm)
                * params_.SID / params_.SDD;
        }

        // 改名更清晰
        float computeViewSelectionHalf_() const
        {
            if (params_.bShortScan) {
                // Parker 模式：选取 π+2γ
                const float half_fan = std::atan(
                    params_.iPU * 0.5f * params_.du_mm / params_.SDD);
                return (CUDA_PI + 2.f * half_fan) * 0.5f;
            }
            else {
                // 整圈模式：选取 π
                return CUDA_PI;
            }
        }

        bool initPipeline_()
        {
            const int iPU = params_.iPU;
            const int iPV = params_.iPV;

            int max_K = 0;
            for (auto& s : slabs_)
                max_K = std::max(max_K, (int)s.views.indices.size());

            const int chunk = std::min(cfg_.Kchunk, max_K);

            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_LOGE("[HelicalReconstructor] PreweightProcessor init failed");
                    return false;
                }
            }

            if (!reinitHelicalParker_(iPU, iPV, chunk))
                return false;

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, chunk };
                ictx.desc = params_.desc;
                ictx.policy = {};
                ictx.stream = stream_;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    YK_LOGE("[HelicalReconstructor] FilterProcessor init failed");
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

        // z0 参数删掉，Parker 不再需要
        bool reinitHelicalParker_(int iPU, int iPV, int K)
        {


            Helical::HelicalParkerInitContext hpctx{};
            hpctx.dims = SProjDims{ iPU, iPV, K };
            hpctx.fDetUSize = params_.du_mm;
            hpctx.fSrcOrigin = params_.SID;
            hpctx.fDetOrigin = params_.SDD - params_.SID;
            hpw_.setInitContext(&hpctx);
            if (!hpw_.init()) {
                YK_LOGE("[HelicalReconstructor] HelicalParker init failed");
                return false;
            }
            return true;
        }

        // z0 参数删掉
        bool reinitChunkProcessors_(int iPU, int iPV, int K,
            cudaStream_t stream)
        {
            {
                PreweightInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.policy = {};
                pw_.setInitContext(&ictx);
                if (!pw_.init()) {
                    YK_LOGE("[HelicalReconstructor] pw reinit failed");
                    return false;
                }
            }

            if (!reinitHelicalParker_(iPU, iPV, K))
                return false;

            {
                FdkFilterInitContext ictx{};
                ictx.dims = SProjDims{ iPU, iPV, K };
                ictx.desc = params_.desc;
                ictx.policy = {};
                ictx.stream = stream;
                flt_.setInitContext(&ictx);
                if (!flt_.init()) {
                    YK_LOGE("[HelicalReconstructor] flt reinit failed");
                    return false;
                }
            }

            return true;
        }

        void mergeSlabToVolume_(const float* h_slab,
            const HelicalSlabConfig& slab)
        {
            const int iVX = params_.iVX;
            const int iVY = params_.iVY;

            const float z_slab_start = slab.z_center - cfg_.z_block_mm * 0.5f;
            const float z_slab_end = slab.z_center + cfg_.z_block_mm * 0.5f;

            for (int iz = 0; iz < slab.z_count_vox; ++iz) {
                const int vol_iz = slab.z_start_vox + iz;
                if (vol_iz < 0 || vol_iz >= params_.iVZ) continue;

                const float z_world = z_vol_start_
                    + (vol_iz + 0.5f) * params_.vox_z_mm;

                const float t = (z_world - z_slab_start)
                    / (z_slab_end - z_slab_start);
                const float cw = cosf(CUDA_PI * (t - 0.5f));
                const float w2 = cw * cw;

                const size_t slab_z_off = (size_t)iz * iVX * iVY;
                const size_t vol_z_off = (size_t)vol_iz * iVX * iVY;

                for (size_t xy = 0; xy < (size_t)iVX * iVY; ++xy) {
                    h_vol_accum_[vol_z_off + xy] += w2 * h_slab[slab_z_off + xy];
                    h_weight_accum_[vol_z_off + xy] += w2;
                }
            }
        }

        bool         is_initialized_ = false;
        bool         pipeline_initialized_ = false;

        SCBCTParams   params_{};
        SHeliFpConfig cfg_{};
        cudaStream_t  stream_ = nullptr;
        int           device_id_ = 0;

        float z_vol_start_ = 0.f;
        float z_vol_end_ = 0.f;
        float margin_ = 0.f;

        std::vector<float>             angle_list_;
        std::vector<SConeProjGeomVec>  helical_geo_;
        std::vector<HelicalSlabConfig> slabs_;

        Fdk::PreweightProcessor         pw_;
        Helical::HelicalParkerProcessor hpw_;
        Fdk::FilterProcessor            flt_;
        FdkGpuContext                   gpu_ctx_;

        std::vector<float> h_vol_accum_;
        std::vector<float> h_weight_accum_;
    };

} // namespace YK