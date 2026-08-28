// YkFDKDataBus.hpp
#pragma once
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <texture_types.h>
#include <vector>
#include "global/YkCudaTextureController.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"
#include "FDK/YkFDKBackProject.cuh"
#include "FDK/kernels/YkFDKBpHelpers.cuh"
#include "FDK/kernels/YkFDKBpPrecompute.cuh"


namespace YK {
    // 在 namespace YK { 内部顶部加：
    using Mem::DeviceLinearBuffer;
    using Mem::PodDataController;
    using Mem::DeviceLinearBuffer3D;



    // ================================================================
    // FdkProjVolData
    // 投影中间缓冲：float 数据用 MemoryController
    // texture object 数组用 DeviceLinearBuffer（POD 类型）
    // ================================================================
    struct FdkProjVolData {
        Mem::DeviceLinearBuffer3D<float>                   chunk_in;
        Mem::DeviceLinearBuffer3D<float>                   chunk_pw;
        Mem::DeviceLinearBuffer3D<float>                   chunk_flt;
        Mem::DeviceLinearBuffer<cudaTextureObject_t> tex_objs;
        std::vector<cudaTextureObject_t>             h_tex_objs;

        int Nu_ = 0, Nv_ = 0;

        void init(int Nu, int Nv, int Kchunk, int deviceId = 0)
        {
            Kchunk = std::min(Kchunk, kMaxChunkAng);  // kMaxChunkAng 是硬上限

            Nu_ = Nu; Nv_ = Nv;
            const size_t view_elems = (size_t)Nu * Nv;

            Mem::MemoryController mc;
            chunk_in = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);
            chunk_pw = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);
            chunk_flt = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);

            Mem::PodDataController dc;
            tex_objs = dc.allocate<cudaTextureObject_t>(Kchunk, deviceId);
            h_tex_objs.resize(Kchunk);

            // 按最大 Kchunk 建好所有句柄，绑定 chunk_flt 地址
            buildTexObjs_(Nu, Nv, Kchunk);

            // 上传一次，之后不再需要重复上传
            dc.upload(tex_objs, h_tex_objs.data(), Kchunk);
        }

        void uploadProjChunk(const float* h_src, int K, cudaStream_t stream) const
        {
            YK_CUDA_CHECK(cudaMemcpyAsync(
                chunk_in.data(),
                h_src,
                (size_t)K * Nu_ * Nv_ * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }

        cudaTextureObject_t* d_texObjs() const { return tex_objs.data(); }

        void destroy() {
            for (auto t : h_tex_objs)
                if (t) cudaDestroyTextureObject(t);
            h_tex_objs.clear();
            tex_objs.reset();
            chunk_in.reset();
            chunk_pw.reset();
            chunk_flt.reset();
        }

        ~FdkProjVolData() { destroy(); }

    private:
        void buildTexObjs_(int Nu, int Nv, int K) {


            // float投影数据，类型自动推导
            h_tex_objs = Mem::TextureController::createTex2DLinearBatch(chunk_flt.data(), Nu, Nv, K);


        }
    };



    // ================================================================
    // FdkGeoData
    // 几何参数：自定义结构体用 PodDataController + DeviceLinearBuffer
    // ================================================================
    struct FdkGeoData {
        DeviceLinearBuffer<SConeProjGeomVec>  geo;
        DeviceLinearBuffer<SFDKGeoParamPerView> gv;
        DeviceLinearBuffer<FdkAffineCoeff>      coeffs;


        void init(int capacity = kMaxChunkAng, int deviceId = 0)
        {
            capacity = std::max(capacity, 1); // 避免零大小分配

            PodDataController dc;
            geo = dc.allocate<SConeProjGeomVec>(capacity, deviceId);
            gv = dc.allocate<SFDKGeoParamPerView>(capacity, deviceId);
            coeffs = dc.allocate<FdkAffineCoeff>(capacity, deviceId);
            // 不上传，不计算 coeffs
        }

        // 在 FdkGeoData 里加这个方法
        void uploadBatchIncremental(
            const std::vector<SConeProjGeomVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int offset, int K, cudaStream_t stream)
        {
            YK_CUDA_CHECK(cudaMemcpyAsync(
                geo.data() + offset, h_geo.data(),
                K * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice, stream));
            YK_CUDA_CHECK(cudaMemcpyAsync(
                gv.data() + offset, h_gv.data(),
                K * sizeof(SFDKGeoParamPerView), cudaMemcpyHostToDevice, stream));
            Fdk::bp_launchPrecomputeCoeffs(
                geo.data() + offset, gv.data() + offset,
                coeffs.data() + offset, K, stream);
        }

        void uploadBatchIncremental(
            const SConeProjGeomVec* h_geo,
            const SFDKGeoParamPerView* h_gv,
            int offset, int K, cudaStream_t stream)
        {
            YK_CUDA_CHECK(cudaMemcpyAsync(
                geo.data() + offset, h_geo,
                K * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice, stream));
            YK_CUDA_CHECK(cudaMemcpyAsync(
                gv.data() + offset, h_gv,
                K * sizeof(SFDKGeoParamPerView), cudaMemcpyHostToDevice, stream));
            Fdk::bp_launchPrecomputeCoeffs(
                geo.data() + offset, gv.data() + offset,
                coeffs.data() + offset, K, stream);
        }

        void uploadCoeffsChunk(const FdkAffineCoeff* d_src, int K, cudaStream_t stream) const
        {
            Fdk::bp_uploadCoeffsChunk(d_src, K, stream);
#ifdef YK_DEBUG
            verifyGCCoeffs(K);
#endif
        }

        void verifyGCCoeffs(int K) const
        {
            std::vector<FdkAffineCoeff> h(K);
            Fdk::bp_readbackCoeffs(h.data(), K);

            constexpr float kEps = FLT_EPSILON;
            bool all_zero = true;
            for (int i = 0; i < K; ++i) {
                const auto& c = h[i];
                if (fabsf(c.Cu_x) > kEps || fabsf(c.Cu_y) > kEps ||
                    fabsf(c.Cu_z) > kEps || fabsf(c.Cu_w) > kEps ||
                    fabsf(c.Cd_x) > kEps || fabsf(c.Cd_y) > kEps ||
                    fabsf(c.Cd_z) > kEps || fabsf(c.Cd_w) > kEps ||
                    fabsf(c.dtheta) > kEps || fabsf(c.source_to_axis_sq) > kEps ||
                    fabsf(c.fScaleDTheta - 1.f) > kEps) {
                    all_zero = false;
                    break;
                }
            }

            if (all_zero) {
                YK_LOGE("verifyGCCoeffs: gC_coeffs[0..{}] all near-zero "
                    "after upload — d_src may be uninitialized.", K - 1);
            }
            else {
                YK_LOGD("verifyGCCoeffs: gC_coeffs[0] Cu=({:.4f},{:.4f},{:.4f},{:.4f}) "
                    "dtheta={:.4f} SID2={:.4f}",
                    h[0].Cu_x, h[0].Cu_y, h[0].Cu_z, h[0].Cu_w,
                    h[0].dtheta, h[0].source_to_axis_sq);
                if (K > 1)
                    YK_LOGD("verifyGCCoeffs: gC_coeffs[{}] Cu=({:.4f},{:.4f},{:.4f},{:.4f}) "
                        "dtheta={:.4f} SID2={:.4f}",
                        K - 1,
                        h[K - 1].Cu_x, h[K - 1].Cu_y, h[K - 1].Cu_z, h[K - 1].Cu_w,
                        h[K - 1].dtheta, h[K - 1].source_to_axis_sq);
            }
        }

        SConeProjGeomVec* d_geo()    const { return geo.data(); }
        SFDKGeoParamPerView* d_gv()     const { return gv.data(); }
        FdkAffineCoeff* d_coeffs() const { return coeffs.data(); }
    };

    // ================================================================
    // FdkGpuContext
    // ================================================================
    struct FdkGpuContext {
        FdkProjVolData proj;
        FdkGeoData     geo;


        FdkGpuContext() = default;
        FdkGpuContext(const FdkGpuContext&) = delete;
        FdkGpuContext& operator=(const FdkGpuContext&) = delete;

        void init(const SProjDims& dims, int capacity,
            cudaStream_t stream, int deviceId = 0)
        {
            proj.init(dims.iPU, dims.iPV, dims.iPAng, deviceId); // iPAng=Kchunk
            geo.init(capacity, deviceId);  // capacity=iPAngTotal
        }


        void uploadGeoIncremental(
            const std::vector<SConeProjGeomVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int offset, int K, cudaStream_t stream)
        {
            geo.uploadBatchIncremental(h_geo, h_gv, offset, K, stream);
        }

        void uploadGeoIncremental(const SConeProjGeomVec* h_geo,
            const SFDKGeoParamPerView* h_gv, int offset, int K, cudaStream_t stream)
        {
            geo.uploadBatchIncremental(h_geo, h_gv, offset, K, stream);
        }

        void release()
        {
            proj.destroy();
            geo.geo.reset();
            geo.gv.reset();
            geo.coeffs.reset();
        }
        ~FdkGpuContext() = default;
    };

} // namespace YK
