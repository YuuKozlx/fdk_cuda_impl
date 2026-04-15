#pragma once
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector>
#include <vector_types.h>
#include "../../global/YkGlobals.h"  // SConeProjGeomVec, SVolGeom
#include "../YkFPCommon.cuh"

namespace YK {
    namespace Fp {

        void fp_joseph_launch(
            cudaTextureObject_t                  volTex,
            const std::vector<SConeProjGeomVec>& h_views,   // 用于主轴判断
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss = FpStepSuperSample::x2);

        void fp_joseph_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float3>& h_src_dirs,  // 每个角度的源点，仅用于主轴判断
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream, FpStepSuperSample step = FpStepSuperSample::x2);

        void fp_joseph_ss_launch(
            cudaTextureObject_t                  volTex,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss = FpStepSuperSample::x1,
            FpDetSuperSample   det = FpDetSuperSample::x1);

        void fp_joseph_ss_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float3>& h_src_dirs,
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss = FpStepSuperSample::x1,
            FpDetSuperSample   det = FpDetSuperSample::x1);

    } // namespace Fp
} // namespace YK