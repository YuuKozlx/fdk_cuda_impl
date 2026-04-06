#pragma once
#include <cuda_runtime.h>
#include "YkFdkFilterContext.hpp"
#include "YkGlobals.h"
#include "YkVecGeo.hpp"
#include "YkVecOperation.hpp"

namespace YK {
    // ============================================================
    // project_uv_and_terms_derived
    // ============================================================
    __device__ __forceinline__ bool project_uv_and_terms_derived(
        const SConeProjectionVec& g,
        const SFDKGeoParamPerView& gv,
        float3 P,
        float& u_pix,
        float& v_pix,
        float& denom_c)
    {
        const float3 dir = f3_sub(P, g.src);

        denom_c = f3_dot(dir, gv.ray_center);

        const float denom_n = f3_dot(dir, gv.det_n);
        if (fabsf(denom_n) < 1e-8f) return false;

        const float t = __fdividef(gv.SDD_mm, denom_n);
        if (t <= 0.f) return false;

        const float DU = t * f3_dot(dir, g.detU) - gv.detS_sub_src_dot_dU;
        const float DV = t * f3_dot(dir, g.detV) - gv.detS_sub_src_dot_dV;

        u_pix = (DU * gv.VV - DV * gv.UV) * gv.invDetUV * gv.inv_du_mm;
        v_pix = (-DU * gv.UV + DV * gv.UU) * gv.invDetUV * gv.inv_dv_mm;
        return true;
    }

    // ============================================================
    // BP kernel — 预计算版本（使用 gC_coeffs constant 内存）
    // ============================================================
    template<int ZSIZE>
    __global__ void fdk_bp_kernel(
        const cudaTextureObject_t* __restrict__ tex_views,
        float* __restrict__ vol,
        SVolumeGeometry vg,
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

            const float w_base = c.SID2 * c.dtheta;

#pragma unroll
            for (int iz = 0; iz < ZSIZE; ++iz) {
                float fr = __fdividef(1.f, den);
                float u = uNum * fr;
                float v = vNum * fr;
                float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
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
                size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                    + (size_t)y * vg.Nx + x;
                vol[idx] += Z[iz];
            }
        }
    }

    // ============================================================
    // BP kernel — 非预计算版本（直接计算投影坐标）
    // ============================================================
    template<int ZSIZE>
    __global__ void fdk_bp_kernel(
        const cudaTextureObject_t* __restrict__ tex_views,
        const SConeProjectionVec* __restrict__ d_geo,
        const SFDKGeoParamPerView* __restrict__ d_gv,
        float* __restrict__ vol,
        SVolumeGeometry vg,
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
            const SConeProjectionVec& g = d_geo[i];
            const SFDKGeoParamPerView& gv = d_gv[i];

            const float fX = vg.origin().x + y * vg.vox_x;
            const float fY = vg.origin().y + x * vg.vox_y;

#pragma unroll
            for (int iz = 0; iz < ZSIZE; ++iz) {
                const int zIdx = startZ + iz;
                if (zIdx >= vg.Nz) continue;

                float3 P = make_float3(fX, fY, vg.origin().z + zIdx * vg.vox_z);

                float u, v, denom_c;
                if (!project_uv_and_terms_derived(g, gv, P, u, v, denom_c))
                    continue;

                float p = tex2D<float>(tex_views[i], u + 0.5f, v + 0.5f);
                float w = (gv.SOD_mm * gv.SOD_mm) / (denom_c * denom_c);
                Z[iz] += p * w * gv.dtheta;
            }
        }

        const int endZ = min(startZ + ZSIZE, vg.Nz);
#pragma unroll
        for (int iz = 0; iz < ZSIZE; ++iz) {
            if (startZ + iz < endZ) {
                size_t idx = (size_t)(startZ + iz) * vg.Ny * vg.Nx
                    + (size_t)y * vg.Nx + x;
                vol[idx] += Z[iz];
            }
        }
    }

    // ============================================================
    // launchBpKernel — 预计算版本
    // ============================================================
    inline void launchBpKernel(
        const cudaTextureObject_t* d_texObjs,
        float* d_vol,
        const SVolumeGeometry& vol_geom,
        int K, cudaStream_t stream)
    {

        constexpr int ZSIZE = 4;
        dim3 block(16, 16, 1);
        dim3 grid(
            (vol_geom.Nx + block.x - 1) / block.x,
            (vol_geom.Ny + block.y - 1) / block.y,
            (vol_geom.Nz + ZSIZE - 1) / ZSIZE);

        fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
            d_texObjs, d_vol, vol_geom, K);
        YK_CUDA_KERNEL_CHECK();
    }

    // ============================================================
    // launchBpKernel — 非预计算版本
    // ============================================================
    inline void launchBpKernel(
        const cudaTextureObject_t* d_texObjs,
        const SConeProjectionVec* d_geo,
        const SFDKGeoParamPerView* d_gv,
        float* d_vol,
        const SVolumeGeometry& vol_geom,
        int K, cudaStream_t stream)
    {


        constexpr int ZSIZE = 4;
        dim3 block(16, 16, 1);
        dim3 grid(
            (vol_geom.Nx + block.x - 1) / block.x,
            (vol_geom.Ny + block.y - 1) / block.y,
            (vol_geom.Nz + ZSIZE - 1) / ZSIZE);

        fdk_bp_kernel<ZSIZE> << <grid, block, 0, stream >> > (
            d_texObjs, d_geo, d_gv, d_vol, vol_geom, K);
        YK_CUDA_KERNEL_CHECK();
    }

} // namespace YK



namespace YK {

    // ----------------------------------------------------------------
    // BpProcessor : IProcessor
    // ----------------------------------------------------------------
    class BpProcessor : public IProcessor {
    public:
        BpProcessor() = default;
        ~BpProcessor() override { release(); }

        BpProcessor(const BpProcessor&) = delete;
        BpProcessor& operator=(const BpProcessor&) = delete;

        void setInitContext(const void* ctx) override
        {
            if (!ctx) {
                std::fprintf(stderr, "[YK][Bp][E] setInitContext: null.\n");
                return;
            }
            const auto* ic = static_cast<const BpInitContext*>(ctx);
            vol_geom_ = ic->vol_geom;
            use_precomputed_ = ic->use_precomputed;
            cfg_ready_ = true;
        }

        bool init() override
        {
            if (!cfg_ready_) {
                std::fprintf(stderr, "[YK][Bp][E] init: setInitContext() not called.\n");
                return false;
            }
            if (vol_geom_.Nx <= 0 || vol_geom_.Ny <= 0 || vol_geom_.Nz <= 0
                || vol_geom_.vox_x <= 0.f
                || vol_geom_.vox_y <= 0.f
                || vol_geom_.vox_z <= 0.f) {
                std::fprintf(stderr, "[YK][Bp][E] init: invalid vol_geom.\n");
                return false;
            }
            is_initialized_ = true;
            return true;
        }

        void setContext(const void* ctx) override
        {
            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Bp][E] setContext: not initialized.\n");
                return;
            }
            if (!ctx) {
                std::fprintf(stderr, "[YK][Bp][E] setContext: null.\n");
                return;
            }
            const auto* cc = static_cast<const BpChunkContext*>(ctx);

            if (!cc->d_texObjs || !cc->d_vol || cc->K <= 0) {
                std::fprintf(stderr, "[YK][Bp][E] setContext: invalid base fields.\n");
                return;
            }

            // 非预计算版本额外校验 d_geo / d_gv
            if (!use_precomputed_ && (!cc->d_geo || !cc->d_gv)) {
                std::fprintf(stderr,
                    "[YK][Bp][E] setContext: non-precomputed mode requires d_geo and d_gv.\n");
                return;
            }

            chunk_ = *cc;
        }

        void process(const float* /*d_input*/,
            float*       /*d_output*/,
            cudaStream_t stream = 0) override
        {
            if (!is_initialized_) {
                std::fprintf(stderr, "[YK][Bp][E] process: not initialized.\n");
                return;
            }
            if (!chunk_.d_texObjs || !chunk_.d_vol || chunk_.K <= 0) {
                std::fprintf(stderr, "[YK][Bp][E] process: setContext() not called.\n");
                return;
            }

            if (use_precomputed_) {
                // 预计算版本：gC_coeffs 已通过 cudaMemcpyToSymbol 上传
                launchBpKernel(
                    chunk_.d_texObjs,
                    chunk_.d_vol,
                    vol_geom_,
                    chunk_.K, stream);
            }
            else {
                // 非预计算版本：直接在 kernel 内计算投影坐标
                launchBpKernel(
                    chunk_.d_texObjs,
                    chunk_.d_geo,
                    chunk_.d_gv,
                    chunk_.d_vol,
                    vol_geom_,
                    chunk_.K, stream);
            }
        }

        void release() override
        {
            chunk_ = {};
            vol_geom_ = {};
            use_precomputed_ = true;
            is_initialized_ = false;
            cfg_ready_ = false;
        }

        bool        isInitialized() const override { return is_initialized_; }
        const char* name()          const override { return "BpProcessor"; }

    private:
        SVolumeGeometry vol_geom_ = {};
        bool            use_precomputed_ = true;
        BpChunkContext  chunk_ = {};
        bool            is_initialized_ = false;
        bool            cfg_ready_ = false;
    };

} // namespace YK