#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#include "../global/YkCudaTextureController.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"
#include "YkFp.cuh"
#include "kernels/YkFPHelpers.cuh"

namespace YK {
    namespace Fp {

        using Mem::DeviceLinearBuffer;
        using Mem::DeviceLinearBuffer3D;
        using Mem::PodDataController;
        using Mem::MemoryController;


        // ================================================================
        // FpGeoData
        // ================================================================
        struct FpGeoData {
        private:
            DeviceLinearBuffer<SConeProjGeomVec> d_projgeom_vox_;
            DeviceLinearBuffer<SConeProjGeomVec> d_projgeom_;     // ← 新增，世界坐标
            std::vector<SConeProjGeomVec>        h_projgeom_;
            std::vector<SConeProjGeomVec>        h_projgeom_vox_;
            SVolGeom                             h_volgeom_;

        public:
            void init(const std::vector<SConeProjGeomVec>& h_projgeom,
                const SVolGeom& vol_geom, int deviceId = 0)
            {
                h_projgeom_ = h_projgeom;
                h_projgeom_vox_ = Fp::normalizeToVoxelBatch(h_projgeom, vol_geom);

                PodDataController dc;
                d_projgeom_ = dc.allocateAndUpload(h_projgeom_, deviceId);  // ← 新增
                d_projgeom_vox_ = dc.allocateAndUpload(h_projgeom_vox_, deviceId);

                h_volgeom_ = vol_geom;
            }

            SConeProjGeomVec* d_views()     const { return d_projgeom_.data(); }  // 世界坐标，Siddon/CVP
            SConeProjGeomVec* d_views_vox() const { return d_projgeom_vox_.data(); }  // 体素坐标，Joseph
            const SConeProjGeomVec* h_views()     const { return h_projgeom_.data(); }
            const SConeProjGeomVec* h_views_vox() const { return h_projgeom_vox_.data(); }
            const SVolGeom& h_volgeom() const { return h_volgeom_; }
        };

        // ================================================================
        // FpGpuContext
        // 汇总 GPU 资源：volTex + geo
        // sinogram 输出由外部传入，不在此管理
        // ================================================================
        struct FpGpuContext {
            Mem::Tex3DHandle volTex;   // CVP 路径下保持默认（无效句柄）
            FpGeoData        geo;
            const float* d_vol_raw = nullptr;  // 无纹理路径使用

            FpGpuContext() = default;
            FpGpuContext(const FpGpuContext&) = delete;
            FpGpuContext& operator=(const FpGpuContext&) = delete;

            // ── 带纹理（Joseph / CVP）─────────────────────────────────
            void init(
                const float* d_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_projgeom,
                int deviceId = 0)
            {
                volTex = Mem::TextureController::createTex3DFromDevice(d_vol, vol_geom);
                geo.init(h_projgeom, vol_geom, deviceId);
            }

            void initFromHost(
                const float* h_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_projgeom,
                int deviceId = 0)
            {
                volTex = Mem::TextureController::createTex3DFromHost(h_vol, vol_geom);
                geo.init(h_projgeom, vol_geom, deviceId);
            }

            // ── 不建纹理（siddon / 未来迭代算法）────────────────────────────
            void initNoTex(
                const float* d_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_projgeom,
                int deviceId = 0)
            {
                d_vol_raw = d_vol;
                geo.init(h_projgeom, vol_geom, deviceId);
                // volTex 保持默认，不分配任何 GPU 纹理资源
            }

            void initFromHostNoTex(
                const float* h_vol,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_projgeom,
                int deviceId = 0)
            {
                // host 指针不能直接传给 kernel，需要先上传
                // 调用者负责先 cudaMalloc+cudaMemcpy，或使用 DeviceLinearBuffer3D
                // 这里仅作接口预留，实际项目中按需实现
                fprintf(stderr, "[FpGpuContext] initFromHostNoTex: not implemented\n");
            }

            void rebindVolume(const float* d_vol, const SVolGeom& vol_geom)
            {
                if (d_vol_raw) {
                    // 无纹理路径，直接换指针
                    d_vol_raw = d_vol;
                }
                else {
                    volTex.destroy();
                    volTex = Mem::TextureController::createTex3DFromDevice(d_vol, vol_geom);
                }
            }

            ~FpGpuContext() = default;
        };

    } // namespace Fp
} // namespace YK