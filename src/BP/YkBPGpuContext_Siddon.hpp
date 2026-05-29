// YkBpSiddonGeoData.hpp
#pragma once
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "../global/YkCudaTextureController.hpp"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"
//#include "YkBp.cuh"
#include "kernels/YkBPHelpers.cuh"

namespace YK {
    namespace Bp {

        using Mem::DeviceLinearBuffer;
        using Mem::DeviceLinearBuffer3D;
        using Mem::PodDataController;
        using Mem::MemoryController;

        // ================================================================
        // BpSiddonGeoData
        // 反投影专用几何数据，镜像 FpGeoData 结构
        // ================================================================
        struct BpSiddonGeoData {
        private:
            DeviceLinearBuffer<SConeProjGeomVec> d_views_world_;  // 世界坐标，Siddon/CVP
            DeviceLinearBuffer<SConeProjGeomVec> d_views_vox_;    // 体素坐标，Joseph
            std::vector<SConeProjGeomVec>        h_views_world_;
            std::vector<SConeProjGeomVec>        h_views_vox_;
            SVolGeom                             h_volgeom_;
            DeviceLinearBuffer<FdkAffineCoeff>      d_coeffs;
            DeviceLinearBuffer<SFDKGeoParamPerView>  d_gv;

        public:
            void init(const std::vector<SConeProjGeomVec>& h_views,
                const SVolGeom& vol_geom,
                int deviceId = 0)
            {
                h_views_world_ = h_views;
                h_views_vox_ = Bp::normalizeToVoxelBatch(h_views, vol_geom);  // 与 Fp 对称
                h_volgeom_ = vol_geom;

                PodDataController dc;
                d_views_world_ = dc.allocateAndUpload(h_views_world_, deviceId);
                d_views_vox_ = dc.allocateAndUpload(h_views_vox_, deviceId);
            }

            void init(const std::vector<SConeProjGeomVec>& h_views,
                const std::vector< SFDKGeoParamPerView>& h_gv,
                const SVolGeom& vol_geom,
                cudaStream_t stream,
                int deviceId = 0)
            {
                h_views_world_ = h_views;
                h_views_vox_ = Bp::normalizeToVoxelBatch(h_views, vol_geom);  // 与 Fp 对称
                h_volgeom_ = vol_geom;

                PodDataController dc;
                d_views_world_ = dc.allocateAndUpload(h_views_world_, deviceId);
                d_views_vox_ = dc.allocateAndUpload(h_views_vox_, deviceId);
                d_gv = dc.allocateAndUpload(h_gv, deviceId);
                d_coeffs = dc.allocate<FdkAffineCoeff>((int)h_views.size(), deviceId);

                Fdk::bp_launchPrecomputeCoeffs(
                    d_views_world_.data(), d_gv.data(),
                    d_coeffs.data(),
                    (int)h_views.size(), stream);
            }


            SConeProjGeomVec* d_views_world() const { return d_views_world_.data(); }  // 世界坐标，Siddon/CVP
            SConeProjGeomVec* d_views_vox()   const { return d_views_vox_.data(); }    // 体素坐标，Joseph
            const SConeProjGeomVec* h_views_world() const { return h_views_world_.data(); }
            const SConeProjGeomVec* h_views_vox()   const { return h_views_vox_.data(); }
            const SVolGeom& h_volgeom() const { return h_volgeom_; }
            const std::vector<SConeProjGeomVec>& h_views_world_vec() const { return h_views_world_; }
            const std::vector<SConeProjGeomVec>& h_views_vox_vec() const { return h_views_vox_; }

            const FdkAffineCoeff* d_coeffs_data() const { return d_coeffs.data(); }
            const SFDKGeoParamPerView* d_gv_data() const { return d_gv.data(); }
        };

        struct BpSiddonGpuContext {
            Mem::Tex3DHandle sinoTex;
            BpSiddonGeoData        geo;
            const float* d_sino_raw = nullptr;
            DeviceLinearBuffer3D<float> d_sino_buf; // 用于不建纹理的情况

            BpSiddonGpuContext() = default;
            BpSiddonGpuContext(const BpSiddonGpuContext&) = delete;
            BpSiddonGpuContext& operator=(const BpSiddonGpuContext&) = delete;

            // ── 带纹理（Joseph / CVP）────────────────────────────────────
            void init(
                const float* d_sino,
                const SProjDims& proj_dims,        // ← 对应 Fp 的 vol_geom
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                int deviceId = 0)
            {
                sinoTex = Mem::TextureController::createTex3DFromDevice(d_sino, proj_dims.iPU, proj_dims.iPV, proj_dims.iPAng);
                geo.init(h_views, vol_geom, deviceId);
            }


            void init(
                const float* d_sino,
                const SProjDims& proj_dims,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                const std::vector<SFDKGeoParamPerView>& h_gv,
                cudaStream_t stream,
                int deviceId = 0)
            {
                sinoTex = Mem::TextureController::createTex3DFromDevice(d_sino, proj_dims.iPU, proj_dims.iPV, proj_dims.iPAng);
                geo.init(h_views, h_gv, vol_geom, stream, deviceId);

            }

            void initFromHost(
                const float* h_sino,
                const SProjDims& proj_dims,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                int deviceId = 0)
            {
                sinoTex = Mem::TextureController::createTex3DFromHost(h_sino, proj_dims.iPU, proj_dims.iPV, proj_dims.iPAng);
                geo.init(h_views, vol_geom, deviceId);
            }

            // ── 不建纹理（Siddon / 迭代算法）────────────────────────────
            void initNoTex(
                const float* d_sino,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                int deviceId = 0)
            {
                d_sino_raw = d_sino;
                geo.init(h_views, vol_geom, deviceId);
            }

            void initNoTex(
                const float* d_sino,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                const std::vector<SFDKGeoParamPerView>& h_gv,
                cudaStream_t stream,
                int deviceId = 0)
            {
                d_sino_raw = d_sino;
                geo.init(h_views, h_gv, vol_geom, stream, deviceId);

            }

            void initFromHostNoTex(
                const float* h_sino,
                const SProjDims& proj_dims,
                const SVolGeom& vol_geom,
                const std::vector<SConeProjGeomVec>& h_views,
                int deviceId = 0)
            {
                MemoryController ctrl;
                d_sino_buf = ctrl.allocateDevice3D<float>(proj_dims.iPU, proj_dims.iPV, proj_dims.iPAng, deviceId);
                geo.init(h_views, vol_geom, deviceId);
            }

            // ── 迭代重建换绑正弦图（对应 Fp::rebindVolume）──────────────
            void rebindSinogram(const float* d_sino, const SProjDims& proj_dims)
            {
                if (d_sino_raw) {
                    d_sino_raw = d_sino;
                }
                else {
                    sinoTex.destroy();
                    sinoTex = Mem::TextureController::createTex3DFromDevice(
                        d_sino, proj_dims.iPU, proj_dims.iPV, proj_dims.iPAng);
                }
            }

            ~BpSiddonGpuContext() = default;
        };

    } // namespace Bp
} // namespace YK