#pragma once
#include "FP/YkFpRunner.hpp"
#include "Heli/YkHelicalGeo.hpp"

namespace YK {

    // ====================================================================
    // ConeProjectorEx
    // 在 ConeProjector 基础上扩展，支持外部传入预建几何
    // 用于螺旋正投影：几何由 build_helical_vec_geometry_from_theta 预建
    // ====================================================================
    class ConeProjectorEx {
    public:
        bool isInitialized() const { return is_initialized_; }

        void release() { is_initialized_ = false; }

        bool init(const SCBCTParams& params,
            ETask task = ETask::FP_Joseph,
            int   deviceId = 0)
        {
            if (params.iPU <= 0 || params.iPV <= 0
                || params.iVX <= 0 || params.iVY <= 0 || params.iVZ <= 0) {
                YK_LOGE("[ConeProjectorEx] invalid params");
                return false;
            }
            if (task != ETask::FP_Joseph &&
                task != ETask::FP_Siddon &&
                task != ETask::FP_CVP) {
                YK_LOGE("[ConeProjectorEx] unsupported task {}", (int)task);
                return false;
            }
            task_ = task;
            deviceId_ = deviceId;
            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // run：使用外部预建几何，不再内部调用 build_circular_geo
        // ----------------------------------------------------------------
        bool run(const float* d_vol,
            const SCBCTParams& params,
            const std::vector<SConeProjGeomVec>& h_views,  // 外部传入
            float* d_sino_out,
            cudaStream_t                          stream,
            TaskDumpCallback                      onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!is_initialized_ || !d_vol || !d_sino_out) {
                YK_LOGE("[ConeProjectorEx] not ready");
                return false;
            }
            if (params.iPAng <= 0
                || (int)h_views.size() != params.iPAng) {
                YK_LOGE("[ConeProjectorEx] h_views size mismatch");
                return false;
            }

            SVolGeom vol_geom = SVolGeom::make_centered(
                params.iVX, params.iVY, params.iVZ,
                params.vox_x_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);

            const bool needTex =
                (task_ == ETask::FP_Joseph || task_ == ETask::FP_CVP);

            Fp::FpGpuContext gpuctx;
            if (needTex)
                gpuctx.init(d_vol, vol_geom, h_views, deviceId_);
            else
                gpuctx.initNoTex(d_vol, vol_geom, h_views, deviceId_);

            const size_t sino_elems =
                (size_t)params.iPAng * params.iPV * params.iPU;
            YK_CUDA_CHECK(cudaMemsetAsync(
                d_sino_out, 0, sino_elems * sizeof(float), stream));

            const int Na = params.iPAng;
            const size_t view_elems = (size_t)params.iPU * params.iPV;

            std::vector<float4> h_src_dirs;
            if (needTex) {
                h_src_dirs.resize(Na);
                for (int i = 0; i < Na; ++i)
                    h_src_dirs[i] = gpuctx.geo.h_views_vox()[i].src;
            }

            switch (task_) {
            case ETask::FP_Joseph:
                Fp::fp_joseph_launch(
                    gpuctx.volTex.tex,
                    gpuctx.geo.h_views_vec(),
                    gpuctx.geo.d_views_vox(),
                    d_sino_out,
                    vol_geom,
                    Na, params.iPU, params.iPV,
                    false,
                    stream);
                break;

            case ETask::FP_Siddon:
                Fp::fp_siddon_launch(
                    gpuctx.d_vol_raw,
                    d_sino_out,
                    gpuctx.geo.d_views(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    false, stream);
                break;

            case ETask::FP_CVP:
                Fp::fp_cvp_launch(
                    gpuctx.volTex.tex,
                    d_sino_out,
                    gpuctx.geo.h_views(),
                    vol_geom,
                    Na, params.iPU, params.iPV,
                    stream);
                break;

            default:
                return false;
            }

            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int i = 0; i < Na; ++i) {
                    DumpPayload payload{
                        i, "sino",
                        static_cast<void*>(d_sino_out + i * view_elems),
                        view_elems, stream, userdata
                    };
                    onDump(&payload);
                }
            }

            return true;
        }

    private:
        bool  is_initialized_ = false;
        ETask task_ = ETask::FP_Joseph;
        int   deviceId_ = 0;
    };

} // namespace YK