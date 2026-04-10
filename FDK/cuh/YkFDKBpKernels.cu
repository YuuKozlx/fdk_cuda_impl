#include "YkFDKBpLaunch.cuh"
#include "YkFDKBpHelpers.cuh"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // fdk_bp_kernel<ZSIZE>  — 预计算版本（读 gC_coeffs constant memory）
            // ----------------------------------------------------------------
            template<int ZSIZE>
            __global__ void fdk_bp_kernel(
                const cudaTextureObject_t* __restrict__ tex_views,
                float* __restrict__ vol,
                SVolGeom vg,
                int K)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                const float fX = vg.origin().x + y * vg.vox_x;
                const float fY = vg.origin().y + x * vg.vox_y;
                const float fZ = vg.origin().z + startZ * vg.vox_z;

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int i = 0; i < K; ++i) {
                    const FdkAffineCoeff& c = gC_coeffs[i];

                    float uNum = c.Cu.w + fX * c.Cu.x + fY * c.Cu.y + fZ * c.Cu.z;
                    float vNum = c.Cv.w + fX * c.Cv.x + fY * c.Cv.y + fZ * c.Cv.z;
                    float den = c.Cd.w + fX * c.Cd.x + fY * c.Cd.y + fZ * c.Cd.z;

                    const float uStep = c.Cu.z * vg.vox_z;
                    const float vStep = c.Cv.z * vg.vox_z;
                    const float dStep = c.Cd.z * vg.vox_z;

                    const float w_base = c.SID2 * c.dtheta * c.fScaleDTheta;

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz) {
                        const float fr = __fdividef(1.f, den);
                        const float u = uNum * fr;
                        const float v = vNum * fr;
                        const float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                        Z[iz] += p * (w_base * fr * fr);
                        uNum += uStep;
                        vNum += vStep;
                        den += dStep;
                    }
                }

                const int endZ = min(startZ + ZSIZE, vg.Nz);
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) {
                    if (startZ + iz < endZ) {
                        const size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                            + (size_t)y * vg.Nx + x;
                        vol[idx] += Z[iz];
                    }
                }
            }

            // ----------------------------------------------------------------
            // fdk_bp_kernel<ZSIZE>  — 非预计算版本（重载，多 d_geo/d_gv 参数）
            // ----------------------------------------------------------------
            template<int ZSIZE>
            __global__ void fdk_bp_kernel(
                const cudaTextureObject_t* __restrict__ tex_views,
                const SConeProjGeomVec* __restrict__ d_geo,
                const SFDKGeoParamPerView* __restrict__ d_gv,
                float* __restrict__ vol,
                SVolGeom vg,
                int K)
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= vg.Nx || y >= vg.Ny) return;

                const int startZ = blockIdx.z * ZSIZE;
                if (startZ >= vg.Nz) return;

                float Z[ZSIZE];
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) Z[iz] = 0.f;

                for (int i = 0; i < K; ++i) {
                    const SConeProjGeomVec& g = d_geo[i];
                    const SFDKGeoParamPerView& gv = d_gv[i];

                    const float fX = vg.origin().x + y * vg.vox_x;
                    const float fY = vg.origin().y + x * vg.vox_y;

#pragma unroll
                    for (int iz = 0; iz < ZSIZE; ++iz) {
                        const int zIdx = startZ + iz;
                        if (zIdx >= vg.Nz) continue;

                        const float3 P = make_float3(fX, fY, vg.origin().z + zIdx * vg.vox_z);

                        float u, v, denom_c;
                        if (!project_uv_and_terms_derived(g, gv, P, u, v, denom_c))
                            continue;

                        const float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                        const float w = (gv.SOD_mm * gv.SOD_mm) / (denom_c * denom_c);
                        Z[iz] += p * w * gv.dtheta * gv.fScaleDTheta;
                    }
                }

                const int endZ = min(startZ + ZSIZE, vg.Nz);
#pragma unroll
                for (int iz = 0; iz < ZSIZE; ++iz) {
                    if (startZ + iz < endZ) {
                        const size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                            + (size_t)y * vg.Nx + x;
                        vol[idx] += Z[iz];
                    }
                }
            }



            void bp_launchBpPrecomputed(
                const cudaTextureObject_t* d_texObjs,
                float* d_vol,
                const SVolGeom& vol_geom,
                int K,
                cudaStream_t stream)
            {
                const dim3 block(16, 16, 1);

                constexpr int zsize = 4;  // ⚠️ 如果你未来要外部控制，可以改成参数

                const dim3 grid(
                    (vol_geom.Nx + block.x - 1) / block.x,
                    (vol_geom.Ny + block.y - 1) / block.y,
                    (vol_geom.Nz + zsize - 1) / zsize);

                switch (zsize)
                {
                case 1:
                    fdk_bp_kernel<1> << <grid, block, 0, stream >> > (
                        d_texObjs, d_vol, vol_geom, K);
                    break;

                case 2:
                    fdk_bp_kernel<2> << <grid, block, 0, stream >> > (
                        d_texObjs, d_vol, vol_geom, K);
                    break;

                case 4:
                    fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                        d_texObjs, d_vol, vol_geom, K);
                    break;

                case 8:
                    fdk_bp_kernel<8> << <grid, block, 0, stream >> > (
                        d_texObjs, d_vol, vol_geom, K);
                    break;
                default:
                    std::printf("Unsupported zsize %d, fallback to 4\n", zsize);
                    fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                        d_texObjs, d_vol, vol_geom, K);

                }

                YK_CUDA_KERNEL_CHECK();
            }

            // ----------------------------------------------------------------
            // bp_launchBpDirect
            //   反投影（非预计算版本）：kernel 内实时计算投影坐标。
            // ----------------------------------------------------------------
            void bp_launchBpDirect(
                const cudaTextureObject_t* d_texObjs,
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                float* d_vol,
                const SVolGeom& vol_geom,
                int K,
                cudaStream_t stream)
            {
                const dim3 block(16, 16, 1);

                constexpr int zsize = 4;

                const dim3 grid(
                    (vol_geom.Nx + block.x - 1) / block.x,
                    (vol_geom.Ny + block.y - 1) / block.y,
                    (vol_geom.Nz + zsize - 1) / zsize);

                switch (zsize)
                {
                case 1:
                    fdk_bp_kernel<1> << <grid, block, 0, stream >> > (
                        d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                    break;

                case 2:
                    fdk_bp_kernel<2> << <grid, block, 0, stream >> > (
                        d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                    break;

                case 4:
                    fdk_bp_kernel<4> << <grid, block, 0, stream >> > (
                        d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                    break;

                case 8:
                    fdk_bp_kernel<8> << <grid, block, 0, stream >> > (
                        d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
                    break;

                default:
                    throw std::runtime_error("unsupported ZSIZE");
                }

                YK_CUDA_KERNEL_CHECK();
            }


        };
    };
}; // namespace YK::Fdk::detail
