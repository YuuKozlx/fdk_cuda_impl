#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "../FDK/YkVecGeo.hpp"
#include "../global/YkCBCTParams.h"
#include "../global/YkCudaTextureController.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

#include <cuda_runtime_api.h>
#include "YkFPGpuContext.hpp"
#include "YkFPPipelineContext.hpp"
#include "kernels/YkFPLaunch.cuh"

namespace YK {


    class FpReconstructor {
    public:
        bool isInitialized() const { return is_initialized_; }

        void release() { is_initialized_ = false; }

        bool init(const SCBCTParams& params, int deviceId = 0)
        {
            if (params.iPAng <= 0 || params.iPU <= 0 || params.iPV <= 0
                || (int)params.angle_list.size() != params.iPAng) {
                fprintf(stderr, "[FpReconstructor] invalid init params\n");
                return false;
            }
            is_initialized_ = true;
            return true;
        }

        bool run(const float* d_vol,
            const SCBCTParams& params,
            float* d_sino_out,
            cudaStream_t stream,
            int deviceId = 0,
            std::function<void(int, float*, size_t)> onDump = nullptr)
        {
            if (!is_initialized_ || !d_vol || !d_sino_out) {
                fprintf(stderr, "[FpReconstructor] run: not ready\n");
                return false;
            }

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            SVolGeom vol_geom = SVolGeom::make_centered(
                params.iVX, params.iVY, params.iVZ,
                params.vox_xy_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);

            std::vector<SConeProjGeomVec>    h_views(params.iPAng);

            build_circular_vec_geometry_from_theta(
                h_views, params.angle_list, params.iPAng,
                params.iPU, params.iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            Fp::FpGpuContext gpuctx;
            gpuctx.init(d_vol, vol_geom, h_views,
                params.iPU, params.iPV, deviceId);



            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                const size_t view_elems = (size_t)params.iPU * params.iPV;
                for (int i = 0; i < params.iPAng; ++i)
                    onDump(i, d_sino_out + i * view_elems, view_elems);
            }

            return true;
        }

    private:
        bool is_initialized_ = false;
    };

    YK_INLINE bool fp_project(
        const float* d_vol,
        float* d_sino_out,
        const SCBCTParams& params,
        cudaStream_t       stream,
        std::function<void(int, float*, size_t)> onDump = nullptr)
    {
        FpReconstructor fp;
        if (!fp.init(params)) return false;
        return fp.run(d_vol, params, d_sino_out, stream, 0, onDump);
    }

} // namespace YK