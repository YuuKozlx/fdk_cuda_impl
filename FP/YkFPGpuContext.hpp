#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "../global/YkCudaTextureController.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"
#include "YkFPPipelineContext.hpp"

namespace YK {
    namespace Fp {

        using Mem::DeviceLinearBuffer;
        using Mem::DeviceLinearBuffer3D;
        using Mem::PodDataController;
        using Mem::MemoryController;

        // ================================================================
        // FpGeoData
        // 几何数组 host → device，同时保留 host copy 供主轴分组
        // ================================================================
        struct FpGeoData {
            DeviceLinearBuffer<SConeProjGeomVec> geo;       // 原始世界坐标，FDK 等其他用途
            DeviceLinearBuffer<SConeProjGeomVec> geo_vox;   // 体素坐标系，供 Joseph kernel
            std::vector<SConeProjGeomVec>        h_geo;     // host copy，供主轴分组
            std::vector<SConeProjGeomVec>        h_geo_vox; // host copy，供调试

            void init(const std::vector<SConeProjGeomVec>& views,
                const SVolGeom& g,
                int deviceId = 0)
            {
                h_geo = views;
                PodDataController dc;
                geo = dc.allocateAndUpload(views, deviceId);

                // 归一化到体素坐标系
                h_geo_vox.resize(views.size());
                for (int i = 0; i < (int)views.size(); ++i)
                    h_geo_vox[i] = normalizeToVoxel(views[i], g);

                geo_vox = dc.allocateAndUpload(h_geo_vox, deviceId);
            }

            // 世界坐标 → 体素坐标系
            // 各分量除以对应方向体素尺寸，中心偏置通过 vol_origin 吸收
            static SConeProjGeomVec normalizeToVoxel(
                const SConeProjGeomVec& v, const SVolGeom& g)
            {
                // ASTRA 约定：体积中心为原点，体素坐标 = 世界坐标 / vox
                // vol_origin 不为零时（有中心偏置），需要把几何平移到以体积中心为原点的坐标系
                // 平移：world' = world - vol_center
                // 然后除以体素尺寸
                const float cx = g.center.x, cy = g.center.y, cz = g.center.z;
                const float ivx = 1.f / g.vox_x;
                const float ivy = 1.f / g.vox_y;
                const float ivz = 1.f / g.vox_z;

                SConeProjGeomVec r;
                // src：平移到体积中心坐标系，再除以体素尺寸
                r.src.x = (v.src.x - cx) * ivx;
                r.src.y = (v.src.y - cy) * ivy;
                r.src.z = (v.src.z - cz) * ivz;

                // detS：同上
                r.detS.x = (v.detS.x - cx) * ivx;
                r.detS.y = (v.detS.y - cy) * ivy;
                r.detS.z = (v.detS.z - cz) * ivz;

                // detU/detV：方向向量只除以体素尺寸，不平移
                r.detU.x = v.detU.x * ivx;
                r.detU.y = v.detU.y * ivy;
                r.detU.z = v.detU.z * ivz;

                r.detV.x = v.detV.x * ivx;
                r.detV.y = v.detV.y * ivy;
                r.detV.z = v.detV.z * ivz;

                // srcCR/angle 不需要归一化，主轴判断用世界坐标即可
                r.srcCR = v.srcCR;
                r.angle = v.angle;

                return r;
            }

            SConeProjGeomVec* d_views(int offset = 0) const { return geo.data() + offset; }
            SConeProjGeomVec* d_views_vox(int offset = 0) const { return geo_vox.data() + offset; }
            const SConeProjGeomVec* h_views(int offset = 0) const { return h_geo.data() + offset; }
            const SConeProjGeomVec* h_views_vox(int offset = 0) const { return h_geo_vox.data() + offset; }
            int size() const { return (int)h_geo.size(); }
        };

        // ================================================================
        // FpSinoData
        // sinogram 输出缓冲：[Na][Nv][Nu]，U 最快
        // ================================================================
        struct FpSinoData {
            DeviceLinearBuffer3D<float> sino;
            int Na_ = 0, Nv_ = 0, Nu_ = 0;

            void init(int Na, int Nv, int Nu, int deviceId = 0)
            {
                Na_ = Na; Nv_ = Nv; Nu_ = Nu;
                MemoryController mc;
                sino = mc.allocateDevice3D<float>((size_t)Nv * Nu, Na, 1, deviceId);
            }

            void zero(cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaMemsetAsync(
                    sino.data(), 0,
                    (size_t)Na_ * Nv_ * Nu_ * sizeof(float), stream));
            }

            void download(float* h_dst, cudaStream_t stream) const
            {
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    h_dst, sino.data(),
                    (size_t)Na_ * Nv_ * Nu_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream));
            }

            float* data() const { return sino.data(); }
        };

        // ================================================================
        // FpGpuContext
        // 汇总所有 GPU 资源
        // ================================================================
        struct FpGpuContext {
            Mem::TextureController::Tex3DHandle volTex;
            FpGeoData                           geo;
            FpSinoData                          sino;

            FpGpuContext() = default;
            FpGpuContext(const FpGpuContext&) = delete;
            FpGpuContext& operator=(const FpGpuContext&) = delete;

            // 从 device 体积初始化
            void init(
                const float* d_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& views,
                int Nu, int Nv,
                int deviceId = 0)
            {
                Mem::TextureController tc;
                volTex = tc.createTex3DFromDevice(d_vol, vol_geom);
                geo.init(views, vol_geom, deviceId);
                sino.init((int)views.size(), Nv, Nu, deviceId);
            }

            // 从 host 体积初始化
            void initFromHost(
                const float* h_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& views,
                int Nu, int Nv,
                int deviceId = 0)
            {
                Mem::TextureController tc;
                volTex = tc.createTex3DFromHost(h_vol, vol_geom);
                geo.init(views, vol_geom, deviceId);
                sino.init((int)views.size(), Nv, Nu, deviceId);
            }

            // 构造 FpChunkContext，供 FpProcessor::setContext() 使用
            FpChunkContext buildChunkContext(int offset, int K) const
            {
                FpChunkContext ctx;
                ctx.d_views = geo.d_views(offset);
                ctx.d_views_vox = geo.d_views_vox(offset);
                ctx.h_views = geo.h_views(offset);
                ctx.K = K;
                ctx.angleOffset = offset;
                return ctx;
            }

            ~FpGpuContext() = default;
        };

    } // namespace Fp
} // namespace YK