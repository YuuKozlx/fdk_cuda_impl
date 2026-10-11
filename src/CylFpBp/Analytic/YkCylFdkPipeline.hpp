#pragma once

#include <vector>

#include "CylFpBp/Analytic/YkCylAnalyticReconstruction.hpp"

namespace YK::CylFpBp {

// Legacy inferred-SDD facade; all reconstruction and streaming share the
// explicit-SDD implementation used by the internal execution backend.
class CylFdkPipeline {
public:
    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const SFilterKernelDesc& filter = SFilterKernelDesc::RamLak(),
        cudaStream_t stream = nullptr, int device_id = 0,
        int max_chunk_views = kMaxChunkAng)
    {
        release();
        if (!stream || geometry.size() < 2) return false;
        Analytic::ReconstructionConfig config{};
        config.source_to_detector_mm = inferSdd_(geometry, channels, rows);
        config.max_chunk_views = max_chunk_views;
        return reconstruction_.prepare(volume, channels, rows, geometry,
            config, filter, stream, device_id);
    }

    bool reconstruct(const float* physical_projection, float* volume,
        bool clear_output = true)
    { return reconstruction_.reconstruct(physical_projection, volume, clear_output); }

    bool beginStreaming(float* d_volume, bool clear_output = true)
    { return reconstruction_.beginStreaming(d_volume, clear_output); }
    bool enqueueBatch(const Analytic::CylFdkProjectionBatch& batch)
    { return reconstruction_.enqueueBatch(batch); }
    bool completeStreaming() { return reconstruction_.completeStreaming(); }
    bool cancelStreaming() { return reconstruction_.cancelStreaming(); }
    bool streamingActive() const { return reconstruction_.streamingActive(); }
    int receivedStreamingViews() const { return reconstruction_.receivedStreamingViews(); }
    int projectionWorkspaceViews() const { return reconstruction_.projectionWorkspaceViews(); }
    void release() { reconstruction_.release(); }
    bool isPrepared() const { return reconstruction_.isPrepared(); }

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
    Analytic::Reconstruction reconstruction_{};
};

using FdkPipeline = CylFdkPipeline;

} // namespace YK::CylFpBp
