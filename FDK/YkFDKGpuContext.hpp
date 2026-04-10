// YkFDKDataBus.hpp
#pragma once
#include <channel_descriptor.h>
#include <cuda_runtime.h>
#include <vector>
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"
#include "YkFDKBackProject.cuh"
#include "YkFDKVecGeoDerived.hpp"
#include "YkVecGeo.hpp"
#include "YkFDKBackProject.cuh"


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
            geo = dc.allocateAndUpload(h_geo, deviceId);
            gv = dc.allocateAndUpload(h_gv, deviceId);
            coeffs = dc.allocate<FdkAffineCoeff>(iPA, deviceId);

            Fdk::detail::bp_launchPrecomputeCoeffs(
                geo.data(), gv.data(),
                coeffs.data(), iPA, stream);


        }

        void uploadCoeffsChunk(const FdkAffineCoeff* d_src, int K, cudaStream_t stream) const
        {
            YK_CUDA_CHECK(cudaMemcpyToSymbolAsync(
                gC_coeffs,
                d_src,
                K * sizeof(FdkAffineCoeff),
                0, cudaMemcpyDeviceToDevice, stream));

            int threads = kMaxChunkAng;
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