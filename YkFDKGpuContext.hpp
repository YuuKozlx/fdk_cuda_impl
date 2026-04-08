// YkFDKDataBus.hpp
#pragma once
#include <cuda_runtime.h>
#include <vector>


#include <channel_descriptor.h>
#include "YkFDKPrecompute.hpp"
#include "YkFDKVecGeoDerived.hpp"
#include "YkGlobals.h"
#include "YkMacro.hpp"
#include "YkMem3d.hpp"
#include "YkVecGeo.hpp"

namespace YK {
    // 在 namespace YK { 内部顶部加：
    using Mem::DeviceLinearBuffer;
    using Mem::PodDataController;
    using Mem::DeviceBuffer3D;



    // ================================================================
    // FdkProjVolData
    // 投影中间缓冲：float 数据用 MemoryController
    // texture object 数组用 DeviceLinearBuffer（POD 类型）
    // ================================================================
    struct FdkProjVolData {
        Mem::DeviceBuffer3D<float>                   chunk_in;
        Mem::DeviceBuffer3D<float>                   chunk_pw;
        Mem::DeviceBuffer3D<float>                   chunk_flt;
        Mem::DeviceLinearBuffer<cudaTextureObject_t> tex_objs;
        std::vector<cudaTextureObject_t>             h_tex_objs;

        int Nu_ = 0, Nv_ = 0;

        void init(int Nu, int Nv, int Kchunk, cudaStream_t stream, int deviceId = 0)
        {
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
            dc.upload(tex_objs, h_tex_objs.data(), Kchunk, stream);
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
            for (int i = 0; i < K; ++i) {
                cudaResourceDesc res{};
                res.resType = cudaResourceTypePitch2D;
                res.res.pitch2D.devPtr = chunk_flt.data() + (size_t)i * Nu * Nv;
                res.res.pitch2D.desc = cudaCreateChannelDesc<float>();
                res.res.pitch2D.width = Nu;
                res.res.pitch2D.height = Nv;
                res.res.pitch2D.pitchInBytes = Nu * sizeof(float);

                cudaTextureDesc tex{};
                tex.addressMode[0] = cudaAddressModeClamp;
                tex.addressMode[1] = cudaAddressModeClamp;
                tex.filterMode = cudaFilterModeLinear;
                tex.readMode = cudaReadModeElementType;
                tex.normalizedCoords = 0;

                YK_CUDA_CHECK(cudaCreateTextureObject(
                    &h_tex_objs[i], &res, &tex, nullptr));
            }
        }
    };



    __global__ void printGCoeff(int K) {
        int idx = threadIdx.x;
        if (idx >= K) return;

        const FdkAffineCoeff& c = gC_coeffs[idx];
        printf("gC_coeffs[%d] = { Cu=(%f,%f,%f,%f) Cv=(%f,%f,%f,%f) Cd=(%f,%f,%f,%f) dtheta=%f SID2=%f }\n",
            idx,
            c.Cu.x, c.Cu.y, c.Cu.z, c.Cu.w,
            c.Cv.x, c.Cv.y, c.Cv.z, c.Cv.w,
            c.Cd.x, c.Cd.y, c.Cd.z, c.Cd.w,
            c.dtheta, c.SID2);
    }

    // ================================================================
    // FdkGeoData
    // 几何参数：自定义结构体用 PodDataController + DeviceLinearBuffer
    // ================================================================
    struct FdkGeoData {
        DeviceLinearBuffer<SConeProjGeomVec>  geo;
        DeviceLinearBuffer<SFDKGeoParamPerView> gv;
        DeviceLinearBuffer<FdkAffineCoeff>      coeffs;

        void init(
            int iPA, cudaStream_t stream,
            const std::vector<SConeProjGeomVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int deviceId = 0)
        {
            PodDataController dc;
            geo = dc.allocateAndUpload(h_geo, stream, deviceId);
            gv = dc.allocateAndUpload(h_gv, stream, deviceId);
            coeffs = dc.allocate<FdkAffineCoeff>(iPA, deviceId);

            launchPrecomputeCoeffs(
                geo.data(), gv.data(),
                coeffs.data(), iPA, stream);

            int threads = kMaxChunkAng;
            printGCoeff << <1, threads, 0, stream >> > (32); // 打印第一个系数作为示例
        }

        void uploadCoeffsChunk(const FdkAffineCoeff* d_src, int K, cudaStream_t stream) const
        {
            YK_CUDA_CHECK(cudaMemcpyToSymbolAsync(
                gC_coeffs,
                d_src,
                K * sizeof(FdkAffineCoeff),
                0, cudaMemcpyDeviceToDevice, stream));
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

        void init(
            const SProjDims& dims,
            const std::vector<SConeProjGeomVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int Kchunk, cudaStream_t stream,
            int deviceId = 0)
        {
            proj.init(dims.iPU, dims.iPV, Kchunk, stream, deviceId);
            geo.init(dims.iPAng, stream, h_geo, h_gv, deviceId);
        }

        ~FdkGpuContext() = default;
    };

} // namespace YK