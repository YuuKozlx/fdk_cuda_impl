// YkFDKDataBus.hpp
#pragma once
#include <cfloat>
#include <channel_descriptor.h>
#include <cuda_runtime.h>
#include <vector>
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"
#include "YkFDKBackProject.cuh"
#include "YkFDKBackProject.cuh"
#include "YkFDKVecGeoDerived.hpp"
#include "YkVecGeo.hpp"
#include "../global/YkCudaTextureController.hpp"


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
            Mem::TextureController texCtrl;

            // float投影数据，类型自动推导
            h_tex_objs = texCtrl.createTex2DLinearBatch(chunk_flt.data(), Nu, Nv, K);


        }
    };

    //__global__ void printGCoeff(int K) {
    //    int idx = threadIdx.x;
    //    if (idx >= K) return;

    //    const FdkAffineCoeff& c = gC_coeffs[idx];
    //    printf("gC_coeffs[%d] = { Cu=(%f,%f,%f,%f) Cv=(%f,%f,%f,%f) Cd=(%f,%f,%f,%f) dtheta=%f SID2=%f }\n",
    //        idx,
    //        c.Cu.x, c.Cu.y, c.Cu.z, c.Cu.w,
    //        c.Cv.x, c.Cv.y, c.Cv.z, c.Cv.w,
    //        c.Cd.x, c.Cd.y, c.Cd.z, c.Cd.w,
    //        c.dtheta, c.SID2);
    //}


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

            Fdk::bp_launchPrecomputeCoeffs(
                geo.data(), gv.data(),
                coeffs.data(), iPA, stream);

            //cudaStreamSynchronize(stream);
            //FdkAffineCoeff h_c;
            //cudaMemcpy(&h_c, coeffs.data(), sizeof(FdkAffineCoeff), cudaMemcpyDeviceToHost);
            //printf("[coeffs_raw][0] Cu=(%f,%f,%f,%f) den=(%f,%f,%f,%f) dtheta=%f\n",
            //    h_c.Cu_x, h_c.Cu_y, h_c.Cu_z, h_c.Cu_w,
            //    h_c.Cd_x, h_c.Cd_y, h_c.Cd_z, h_c.Cd_w,
            //    h_c.dtheta);
        }

        void uploadCoeffsChunk(const FdkAffineCoeff* d_src, int K, cudaStream_t stream) const
        {
            Fdk::bp_uploadCoeffsChunk(d_src, K, stream);
            verifyGCCoeffs(K);
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
                    fabsf(c.Cd_x) > kEps || fabsf(c.Cd_y) > kEps || fabsf(c.Cd_z) > kEps || fabsf(c.Cd_w) > kEps ||
                    fabsf(c.dtheta) > kEps || fabsf(c.SID2) > kEps || fabsf(c.fScaleDTheta - 1.f) > kEps) {
                    all_zero = false;
                    break;
                }
            }

            if (all_zero) {
                fprintf(stderr,
                    "[FdkGeoData][E] verifyGCCoeffs: gC_coeffs[0..%d] all near-zero "
                    "after upload — d_src may be uninitialized.\n", K - 1);
            }
            else {
                printf("[FdkGeoData][verify] gC_coeffs[0]  Cu=(%f,%f,%f,%f) dtheta=%f SID2=%f\n",
                    h[0].Cu_x, h[0].Cu_y, h[0].Cu_z, h[0].Cu_w,
                    h[0].dtheta, h[0].SID2);
                if (K > 1)
                    printf("[FdkGeoData][verify] gC_coeffs[%d] Cu=(%f,%f,%f,%f) dtheta=%f SID2=%f\n",
                        K - 1,
                        h[K - 1].Cu_x, h[K - 1].Cu_y, h[K - 1].Cu_z, h[K - 1].Cu_w,
                        h[K - 1].dtheta, h[K - 1].SID2);
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