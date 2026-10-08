#pragma once
// YkBpGpuContext.hpp
#pragma once
#include <vector>
#include <cuda_runtime_api.h>
#include "global/YkMem3d.hpp"
#include "global/YkCudaTextureController.hpp"
#include "global/YkMacro.hpp"
#include "common/YkVecGeo.hpp"
#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFDKBackProject.cuh"



namespace YK {
    namespace Bp
    {
        using Mem::DeviceLinearBuffer;
        using Mem::PodDataController;
        using Mem::DeviceLinearBuffer3D;

        // ================================================================
        // BpProjData
        // 只保留 d_sino + texture，去掉 chunk_in / chunk_pw
        // ================================================================
        struct BpProjData {
            Mem::DeviceLinearBuffer3D<float>             d_sino;
            Mem::DeviceLinearBuffer<cudaTextureObject_t> tex_objs;
            std::vector<cudaTextureObject_t>             h_tex_objs;

            int Nu_ = 0, Nv_ = 0;

            void init(int Nu, int Nv, int Kchunk, cudaStream_t stream,
                int deviceId = 0)
            {
                Nu_ = Nu; Nv_ = Nv;
                const size_t view_elems = (size_t)Nu * Nv;

                Mem::MemoryController mc;
                d_sino = mc.allocateDevice3D<float>(view_elems, Kchunk, 1, deviceId);

                Mem::PodDataController dc;
                tex_objs = dc.allocate<cudaTextureObject_t>(Kchunk, deviceId);
                h_tex_objs.resize(Kchunk);

                h_tex_objs = Mem::TextureController::createTex2DLinearBatch(
                    d_sino.data(), Nu, Nv, Kchunk);
                dc.upload(tex_objs, h_tex_objs.data(), Kchunk);
            }

            // 上传已滤波投影到 d_sino（D2D）
            void uploadFltChunk(const float* d_src, int K,
                cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    d_sino.data(), d_src,
                    (size_t)K * Nu_ * Nv_ * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }

            cudaTextureObject_t* d_texObjs() const { return tex_objs.data(); }

            void destroy()
            {
                for (auto t : h_tex_objs)
                    if (t) cudaDestroyTextureObject(t);
                h_tex_objs.clear();
                tex_objs.reset();
                d_sino.reset();
            }

            ~BpProjData() { destroy(); }
        };

        // ================================================================
        // BpGeoData
        // 与 FdkGeoData 相同，独立一份
        // ================================================================
        struct BpGeoData {
            DeviceLinearBuffer<SConeProjGeomVec>     geo;
            DeviceLinearBuffer<SFDKGeoParamPerView>  gv;
            DeviceLinearBuffer<FdkAffineCoeff>       coeffs;

            void allocate(int capacity, int deviceId = 0)
            {
                if (capacity <= 0) {
                    YK_LOGE("[BpGeoData] allocate: capacity={} invalid", capacity);
                    return;
                }
                PodDataController dc;
                geo = dc.allocate<SConeProjGeomVec>(capacity, deviceId);
                gv = dc.allocate<SFDKGeoParamPerView>(capacity, deviceId);
                coeffs = dc.allocate<FdkAffineCoeff>(capacity, deviceId);
            }

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

            void uploadCoeffsChunk(const FdkAffineCoeff* d_src, int K,
                cudaStream_t stream) const
            {
                Fdk::bp_uploadCoeffsChunk(d_src, K, stream);
            }

            SConeProjGeomVec* d_geo()    const { return geo.data(); }
            SFDKGeoParamPerView* d_gv()     const { return gv.data(); }
            FdkAffineCoeff* d_coeffs() const { return coeffs.data(); }
        };

        // ================================================================
        // BpGpuContext
        // ================================================================
        struct BpGpuContext {
            BpProjData proj;
            BpGeoData  geo;

            BpGpuContext() = default;
            BpGpuContext(const BpGpuContext&) = delete;
            BpGpuContext& operator=(const BpGpuContext&) = delete;

            void init(const SProjDims& dims, int geo_capacity,
                cudaStream_t stream, int deviceId = 0)
            {
                proj.init(dims.iPU, dims.iPV, dims.iPAng, stream, deviceId);
                geo.allocate(geo_capacity, deviceId);
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

            ~BpGpuContext() = default;
        };

    }


} // namespace YK