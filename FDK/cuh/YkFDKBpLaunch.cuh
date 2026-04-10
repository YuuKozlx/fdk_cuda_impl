#pragma once
#include <cuda_runtime.h>

#include "../YkVecGeo.hpp"
#include "../YkFDKVecGeoDerived.hpp"   // FdkAffineCoeff, SVolGeom
#include "../../global/YkMacro.hpp"              // YK_CUDA_KERNEL_CHECK

#include "YkFDKBpPrecompute.cuh"
#include "YkFDKBpKernels.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // bp_launchBpPrecomputed
            //   反投影（预计算版本）：gC_coeffs 须已通过 cudaMemcpyToSymbol 上传。
            // ----------------------------------------------------------------
            inline void bp_launchBpPrecomputed(
                const cudaTextureObject_t* d_texObjs,
                float* d_vol,
                const SVolGeom& vol_geom,
                int                        K,
                cudaStream_t               stream)
            {
                constexpr int ZSIZE = 4;
                const dim3 block(16, 16, 1);
                const dim3 grid(
                    (vol_geom.Nx + block.x - 1) / block.x,
                    (vol_geom.Ny + block.y - 1) / block.y,
                    (vol_geom.Nz + ZSIZE - 1) / ZSIZE);

                fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
                    d_texObjs, d_vol, vol_geom, K);
                YK_CUDA_KERNEL_CHECK();
            }

            // ----------------------------------------------------------------
            // bp_launchBpDirect
            //   反投影（非预计算版本）：kernel 内实时计算投影坐标。
            // ----------------------------------------------------------------
            inline void bp_launchBpDirect(
                const cudaTextureObject_t* d_texObjs,
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                float* d_vol,
                const SVolGeom& vol_geom,
                int                        K,
                cudaStream_t               stream)
            {
                constexpr int ZSIZE = 4;
                const dim3 block(16, 16, 1);
                const dim3 grid(
                    (vol_geom.Nx + block.x - 1) / block.x,
                    (vol_geom.Ny + block.y - 1) / block.y,
                    (vol_geom.Nz + ZSIZE - 1) / ZSIZE);

                fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
                    d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                YK_CUDA_KERNEL_CHECK();
            }

        }
    }
} // namespace YK::Fdk::detail
