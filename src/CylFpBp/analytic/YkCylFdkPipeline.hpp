#pragma once

#include <vector>

#include "CylFpBp/analytic/YkCylAnalyticGeometryCanonicalizer.hpp"
#include "FDK/YkFdkPipeline.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp {

// 柱面解析入口：内部完成柱面到平板的重采样，之后完全复用 Flat-FDK。
class CylFdkPipeline {
public:
    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const SFilterKernelDesc& filter = SFilterKernelDesc::RamLak(),
        cudaStream_t stream = nullptr, int device_id = 0)
    {
        release();
        if (!stream || geometry.size() < 2) return false;
        Analytic::GeometryCanonicalizer canonicalizer;
        Analytic::CanonicalGeometry canonical{};
        Analytic::GeometryCanonicalizationConfig config{};
        config.source_to_detector_mm = inferSdd_(geometry, channels, rows);
        if (!(config.source_to_detector_mm > 0.f) ||
            !canonicalizer.canonicalize(channels, rows, geometry, config, canonical))
            return false;

        SReconstructionParams params{};
        params.scan.Nu = channels;
        params.scan.Nv = rows;
        params.scan.NAng = static_cast<int>(geometry.size());
        params.scan.totalViews = params.scan.NAng;
        params.scan.du_mm = canonical.projection_map.target_du_mm;
        params.scan.dv_mm = canonical.projection_map.target_row_step_mm;
        params.scan.sid_mm = config.source_to_detector_mm;
        params.scan.sdd_mm = config.source_to_detector_mm;
        params.scan.range_rad = 2.f * static_cast<float>(CUDA_PI);
        params.volume.Nx = volume.Nx; params.volume.Ny = volume.Ny;
        params.volume.Nz = volume.Nz;
        params.volume.voxelX_mm = volume.vox_x;
        params.volume.voxelY_mm = volume.vox_y;
        params.volume.voxelZ_mm = volume.vox_z;
        params.volume.centerX_mm = volume.center.x;
        params.volume.centerY_mm = volume.center.y;
        params.volume.centerZ_mm = volume.center.z;
        params.reconstruction.filter = filter;
        if (!mapper_.prepare(canonical.projection_map) ||
            !fdk_.prepareWithGeometry(params, canonical.flat_geometry,
                params.scan.totalViews, stream, device_id)) {
            release();
            return false;
        }
        mapped_ = memory_.allocateDevice3D<float>(channels, rows,
            params.scan.totalViews, device_id, false);
        if (!mapped_) { release(); return false; }
        channels_ = channels; rows_ = rows; views_ = params.scan.totalViews;
        stream_ = stream; prepared_ = true;
        return true;
    }

    bool reconstruct(const float* physical_projection, float* volume,
        bool clear_output = true)
    {
        if (!prepared_ || !physical_projection || !volume) return false;
        if (!mapper_.apply(physical_projection, mapped_.data(), stream_)) return false;
        return fdk_.processBatchSync({mapped_.data(), nullptr, nullptr, views_},
            volume, clear_output);
    }

    void release()
    {
        fdk_.release(); mapper_.release(); mapped_ = {};
        channels_ = rows_ = views_ = 0; stream_ = nullptr; prepared_ = false;
    }
    bool isPrepared() const { return prepared_; }

private:
    static float inferSdd_(const std::vector<SCylConeProjGeomVec>& geometry,
        int channels, int rows)
    {
        SCylProjectionFrame frame{};
        if (!deriveCylProjectionFrame(geometry.front(), channels, rows, frame)) return 0.f;
        const float3 source = make_float3(geometry.front().source.x,
            geometry.front().source.y, geometry.front().source.z);
        const float3 d = make_float3(frame.detectorCenter.x - source.x,
            frame.detectorCenter.y - source.y, frame.detectorCenter.z - source.z);
        return d.x * frame.radialUnit.x + d.y * frame.radialUnit.y +
            d.z * frame.radialUnit.z;
    }
    Analytic::ProjectionMapper mapper_{};
    ::YK::FdkPipeline fdk_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> mapped_{};
    int channels_ = 0, rows_ = 0, views_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

using FdkPipeline = CylFdkPipeline;

} // namespace YK::CylFpBp
