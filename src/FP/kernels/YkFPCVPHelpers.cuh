#pragma once
#include <cuda_runtime.h>
#include "global/YkGlobals.h"


namespace YK
{
    namespace Fp {
        // ============================================================
        //  Per-view precomputed cache (built once on host, passed to kernels)
        //  All divisions and sqrts are done on the host so the kernel
        //  contains only multiply-add operations.
        // ============================================================
        struct SCVPViewCache {
            // Copied directly from SConeProjGeomVec
            float4 src;
            float4 srcCR;    // detector normal (unit vector, src -> detector)
            float4 detS;     // world position of pixel (0,0) origin (not pixel center)
            float4 detU;     // per-pixel U vector (carries physical spacing)
            float4 detV;     // per-pixel V vector (carries physical spacing)

            // Precomputed quantities
            float  du, dv;           // pixel physical spacing [mm] = |detU|, |detV|
            float  inv_du, inv_dv;
            float3 detU_n, detV_n;   // normalize(detU), normalize(detV)

            float  SDD;              // source-detector distance [mm]
            float3 det_center;       // geometric center of the detector in world coords

            int    M, N;             // detector size: cols (U), rows (V)

            // Volume
            float3 vol_origin;
            float  a1, a2, a3;
            float  inv_a1, inv_a2, inv_a3;
            int    Nx, Ny, Nz;

            // Projected depth of the voxel bounding box along srcCR (precomputed per view).
            // For axis-aligned voxels: depth = |a1*(X.srcCR)| + |a2*(Y.srcCR)| + |a3*(Z.srcCR)|
            // If the volume has an additional rotation matrix R, replace the world-axis
            // dot products with dot(R_col_alpha, srcCR).
            // Note: currently retained in the struct but not used in the kernel after
            // the cut-volume formula was corrected to use voxel-volume * overlap-fraction.
            float  vox_depth;  // [mm]
        };

        // ============================================================
        //  Host: build SCVPViewCache from user geometry + volume desc
        // ============================================================
        inline static SCVPViewCache make_view_cache(
            const SConeProjGeomVec& g,
            int M, int N,
            const SVolGeom& vol)
        {
            SCVPViewCache c;

            c.src = g.src;
            c.srcCR = g.srcCR;
            c.detS = g.detS;
            c.detU = g.detU;
            c.detV = g.detV;
            c.M = M; c.N = N;

            // Pixel physical spacing
            c.du = sqrtf(g.detU.x * g.detU.x + g.detU.y * g.detU.y + g.detU.z * g.detU.z);
            c.dv = sqrtf(g.detV.x * g.detV.x + g.detV.y * g.detV.y + g.detV.z * g.detV.z);
            c.inv_du = 1.f / c.du;
            c.inv_dv = 1.f / c.dv;
            c.detU_n = make_float3(g.detU.x * c.inv_du, g.detU.y * c.inv_du, g.detU.z * c.inv_du);
            c.detV_n = make_float3(g.detV.x * c.inv_dv, g.detV.y * c.inv_dv, g.detV.z * c.inv_dv);

            // Detector geometric center: detS + M/2 * detU + N/2 * detV
            // detS is the pixel (0,0) origin, not the pixel center.
            // Pixel (m,n) center = detS + (m+0.5)*detU + (n+0.5)*detV
            const float hM = (float)M * 0.5f, hN = (float)N * 0.5f;
            c.det_center = make_float3(
                g.detS.x + hM * g.detU.x + hN * g.detV.x,
                g.detS.y + hM * g.detU.y + hN * g.detV.y,
                g.detS.z + hM * g.detU.z + hN * g.detV.z);

            // SDD = dot(det_center - src, srcCR)
            const float dcx = c.det_center.x - g.src.x;
            const float dcy = c.det_center.y - g.src.y;
            const float dcz = c.det_center.z - g.src.z;
            c.SDD = dcx * g.srcCR.x + dcy * g.srcCR.y + dcz * g.srcCR.z;

            // Volume
            c.vol_origin = vol.origin();     // 或者 make_float3(vol.ox, vol.oy, vol.oz)
            c.a1 = vol.vox_x;
            c.a2 = vol.vox_y;
            c.a3 = vol.vox_z;
            c.inv_a1 = 1.f / vol.vox_x;
            c.inv_a2 = 1.f / vol.vox_y;
            c.inv_a3 = 1.f / vol.vox_z;
            c.Nx = vol.Nx; c.Ny = vol.Ny; c.Nz = vol.Nz;


            // Voxel bounding-box projection depth along srcCR.
            // srcCR is a unit vector, so dot(world_axis_alpha, srcCR) = srcCR component alpha.
            c.vox_depth = fabsf(vol.vox_x * g.srcCR.x)
                + fabsf(vol.vox_y * g.srcCR.y)
                + fabsf(vol.vox_z * g.srcCR.z);

            return c;
        }

        // ============================================================
        //  Device inline math helpers (shared by all .cu files)
        // ============================================================
        __device__ __forceinline__
            float3 d_sub(float3 a, float3 b) {
            return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
        }
        __device__ __forceinline__
            float d_dot(float3 a, float3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }
        __device__ __forceinline__
            float3 d_scale(float t, float3 a) {
            return make_float3(t * a.x, t * a.y, t * a.z);
        }
        __device__ __forceinline__
            float3 d_add(float3 a, float3 b) {
            return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
        }
        // Fused multiply-add: t*a + b (uses fmaf to reduce rounding error)
        __device__ __forceinline__
            float3 d_fma(float t, float3 a, float3 b) {
            return make_float3(fmaf(t, a.x, b.x), fmaf(t, a.y, b.y), fmaf(t, a.z, b.z));
        }



        struct CVPVolTexture {
            cudaArray_t          array = nullptr;
            cudaTextureObject_t  tex = 0;

            // 从 device 线性内存创建纹理
            // d_vol 布局：[Nz][Ny][Nx]，z-major
            void create(const float* d_vol, int Nx, int Ny, int Nz) {
                // 1. 分配 cudaArray
                cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
                cudaExtent extent = make_cudaExtent(Nx, Ny, Nz);
                YK_CUDA_CHECK(cudaMalloc3DArray(&array, &desc, extent));

                // 2. 拷贝数据
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr(
                    const_cast<float*>(d_vol),
                    Nx * sizeof(float), Nx, Ny);
                p.dstArray = array;
                p.extent = extent;
                p.kind = cudaMemcpyDeviceToDevice;
                YK_CUDA_CHECK(cudaMemcpy3D(&p));

                // 3. 创建纹理对象
                cudaResourceDesc resDesc = {};
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = array;

                cudaTextureDesc texDesc = {};
                texDesc.addressMode[0] = cudaAddressModeBorder;  // 边界外返回 0
                texDesc.addressMode[1] = cudaAddressModeBorder;
                texDesc.addressMode[2] = cudaAddressModeBorder;
                texDesc.filterMode = cudaFilterModePoint;    // 不插值
                texDesc.readMode = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;                     // 像素坐标

                YK_CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
            }

            void destroy() {
                if (tex) { cudaDestroyTextureObject(tex);  tex = 0; }
                if (array) { cudaFreeArray(array);            array = nullptr; }
            }
        };
    };


};
