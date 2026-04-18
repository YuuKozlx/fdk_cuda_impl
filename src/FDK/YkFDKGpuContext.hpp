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
#include "FDK/cuh/YkFDKBpHelpers.cuh"
#include "FDK/cuh/YkFDKBpPrecompute.cuh"


namespace YK {
    // �� namespace YK { �ڲ������ӣ�
    using Mem::DeviceLinearBuffer;
    using Mem::PodDataController;
    using Mem::DeviceLinearBuffer3D;



    // ================================================================
    // FdkProjVolData
    // ͶӰ�м仺�壺float ������ MemoryController
    // texture object ������ DeviceLinearBuffer��POD ���ͣ�
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
            Kchunk = std::min(Kchunk, kMaxChunkAng);  // kMaxChunkAng ��Ӳ����

            Nu_ = Nu; Nv_ = Nv;
            const size_t view_elems = (size_t)Nu * Nv;

            Mem::MemoryController mc;
            chunk_in = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);
            chunk_pw = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);
            chunk_flt = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);

            Mem::PodDataController dc;
            tex_objs = dc.allocate<cudaTextureObject_t>(Kchunk, deviceId);
            h_tex_objs.resize(Kchunk);

            // ����� Kchunk �������о������ chunk_flt ��ַ
            buildTexObjs_(Nu, Nv, Kchunk);

            // �ϴ�һ�Σ�֮������Ҫ�ظ��ϴ�
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


            // floatͶӰ���ݣ������Զ��Ƶ�
            h_tex_objs = Mem::TextureController::createTex2DLinearBatch(chunk_flt.data(), Nu, Nv, K);


        }
    };



    // ================================================================
    // FdkGeoData
    // ���β������Զ���ṹ���� PodDataController + DeviceLinearBuffer
    // ================================================================
    struct FdkGeoData {
        DeviceLinearBuffer<SConeProjGeomVec>  geo;
        DeviceLinearBuffer<SFDKGeoParamPerView> gv;
        DeviceLinearBuffer<FdkAffineCoeff>      coeffs;


        void init(int capacity = kMaxChunkAng, int deviceId = 0)
        {
            capacity = std::max(capacity, 1); // �������С����

            PodDataController dc;
            geo = dc.allocate<SConeProjGeomVec>(capacity, deviceId);
            gv = dc.allocate<SFDKGeoParamPerView>(capacity, deviceId);
            coeffs = dc.allocate<FdkAffineCoeff>(capacity, deviceId);
            // ���ϴ��������� coeffs
        }

        // �� FdkGeoData ����������
        void uploadBatchIncremental(
            const std::vector<SConeProjGeomVec>& h_geo,
            const std::vector<SFDKGeoParamPerView>& h_gv,
            int offset, int K, cudaStream_t stream)
        {
            YK_CUDA_CHECK(cudaMemcpy(
                geo.data() + offset, h_geo.data(),
                K * sizeof(SConeProjGeomVec), cudaMemcpyHostToDevice));
            YK_CUDA_CHECK(cudaMemcpy(
                gv.data() + offset, h_gv.data(),
                K * sizeof(SFDKGeoParamPerView), cudaMemcpyHostToDevice));
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
                    fabsf(c.dtheta) > kEps || fabsf(c.SID2) > kEps ||
                    fabsf(c.fScaleDTheta - 1.f) > kEps) {
                    all_zero = false;
                    break;
                }
            }

            if (all_zero) {
                YK_LOGE("verifyGCCoeffs: gC_coeffs[0..{}] all near-zero "
                    "after upload �� d_src may be uninitialized.", K - 1);
            }
            else {
                YK_LOGD("verifyGCCoeffs: gC_coeffs[0] Cu=({:.4f},{:.4f},{:.4f},{:.4f}) "
                    "dtheta={:.4f} SID2={:.4f}",
                    h[0].Cu_x, h[0].Cu_y, h[0].Cu_z, h[0].Cu_w,
                    h[0].dtheta, h[0].SID2);
                if (K > 1)
                    YK_LOGD("verifyGCCoeffs: gC_coeffs[{}] Cu=({:.4f},{:.4f},{:.4f},{:.4f}) "
                        "dtheta={:.4f} SID2={:.4f}",
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