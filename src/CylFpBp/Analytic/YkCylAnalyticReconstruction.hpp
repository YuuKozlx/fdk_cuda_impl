#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFdkPipeline.hpp"
#include "CylFpBp/Analytic/YkCylAnalyticProjectionMapper.hpp"
#include "CylFpBp/Analytic/YkCylAnalyticGeometryCanonicalizer.hpp"
#include "CylFpBp/YkCylFpBpTypes.hpp"
#include "global/YkMem3d.hpp"

namespace YK::CylFpBp::Analytic {

struct ReconstructionConfig {
    // 一般圆柱面与源点存在两个交点，SCylConeProjGeomVec 本身不能唯一
    // 判定哪个交点是主射线落点，因此 SDD 必须由采集标定显式给出。
    float source_to_detector_mm = 0.f;
    // 采集坐标到解析规范坐标的刚体变换。默认保持原坐标系。
    SRigidTransform canonical_from_acquisition = SRigidTransform::identity();
    SKernelLaunchPolicy launch{};
    int max_chunk_views = kMaxChunkAng;
};

struct CylFdkProjectionBatch {
    const float* data = nullptr;
    int first_view = 0;
    int view_count = 0;
    EMemoryLocation location = EMemoryLocation::Device;
    size_t element_count = 0;
};

// 柱面解析重建的通用输入适配层。
//
// 物理探测器可使用任意曲率半径 R；每包重采样到零姿态平板，
// 再交给标准 Flat-FDK 滤波并累加，无跨帧投影依赖。
//
// 当前契约是标准同轴、逐视图一致的圆柱，V 平移由主点保留；逐视图
// 姿态、倾斜探测器和复杂轨迹应继续使用迭代算子。
class Reconstruction {
public:
    ~Reconstruction() { release(); }
    // 便捷接口：适用于调用方已经明确使用 Ram-Lak 的常见 FDK 路径。
    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& physical_geometry,
        float source_to_detector_mm, cudaStream_t stream, int device_id = 0)
    {
        ReconstructionConfig config{};
        config.source_to_detector_mm = source_to_detector_mm;
        return prepare(volume, channels, rows, physical_geometry, config,
            SFilterKernelDesc::RamLak(), stream, device_id);
    }

    bool prepare(const SVolGeom& volume, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& physical_geometry,
        const ReconstructionConfig& config, const SFilterKernelDesc& filter,
        cudaStream_t stream,
        int device_id = 0)
    {
        release();
        if (!stream || physical_geometry.size() < 2 ||
            !(config.source_to_detector_mm > 0.f) || config.max_chunk_views <= 0 ||
            physical_geometry.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            return false;

        // 先完成采集几何的刚体规范化；后续 map 和解析 FDK 只使用
        // corrected_geometry/canonical_geometry，不再在 kernel 中猜测姿态。
        GeometryCanonicalizer canonicalizer;
        CanonicalGeometry canonical{};
        GeometryCanonicalizationConfig canonical_config{};
        canonical_config.source_to_detector_mm = config.source_to_detector_mm;
        canonical_config.canonical_from_acquisition =
            config.canonical_from_acquisition;
        canonical_config.launch = config.launch;
        if (!canonicalizer.canonicalize(channels, rows, physical_geometry,
                canonical_config, canonical)) return false;

        SReconstructionParams params{};
        params.scan.Nu = channels;
        params.scan.Nv = rows;
        params.scan.totalViews = static_cast<int>(physical_geometry.size());
        params.scan.NAng = params.scan.totalViews;
        params.scan.du_mm = canonical.projection_map.target_du_mm;
        params.scan.dv_mm = canonical.projection_map.target_row_step_mm;
        params.scan.sdd_mm = config.source_to_detector_mm;
        params.scan.sid_mm = config.source_to_detector_mm;
        params.scan.range_rad = 2.f * static_cast<float>(CUDA_PI);
        params.scan.angles.reserve(canonical.flat_geometry.size());
        for (const auto& g : canonical.flat_geometry) params.scan.angles.push_back(g.angle.x);
        params.volume.Nx = volume.Nx; params.volume.Ny = volume.Ny;
        params.volume.Nz = volume.Nz;
        params.volume.voxelX_mm = volume.vox_x;
        params.volume.voxelY_mm = volume.vox_y;
        params.volume.voxelZ_mm = volume.vox_z;
        params.volume.centerX_mm = volume.center.x;
        params.volume.centerY_mm = volume.center.y;
        params.volume.centerZ_mm = volume.center.z;
        params.reconstruction.filter = filter;
        chunk_views_ = std::min({params.scan.totalViews, kMaxChunkAng,
            config.max_chunk_views});
        stream_ = stream;
        if (!mapper_.prepare(canonical.projection_map) ||
            !fdk_.prepareWithGeometry(params, canonical.flat_geometry,
                chunk_views_, stream, device_id)) {
            release();
            return false;
        }
        memory_ = Mem::MemoryController{};
        mapped_ = memory_.allocateDevice3D<float>(channels, rows,
            chunk_views_, device_id, false);
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(physical_geometry.size());
        stream_ = stream;
        device_id_ = device_id;
        volume_elements_ = static_cast<size_t>(volume.Nx) * volume.Ny * volume.Nz;
        prepared_ = static_cast<bool>(mapped_);
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* physical_projection, float* volume,
        bool clear_output = true)
    {
        if (!prepared_ || !physical_projection || !volume) return false;
        if (!beginStreaming(volume, clear_output)) return false;
        const CylFdkProjectionBatch batch{physical_projection, 0, views_,
            EMemoryLocation::Device, static_cast<size_t>(channels_) * rows_ * views_};
        if (!enqueueBatch(batch) || !completeStreaming()) {
            cancelStreaming();
            return false;
        }
        return true;
    }

    bool beginStreaming(float* d_volume, bool clear_output = true)
    {
        if (!prepared_ || !d_volume || streaming_volume_) return false;
        fdk_.reset();
        if (clear_output && cudaMemsetAsync(d_volume, 0,
                volume_elements_ * sizeof(float), stream_) != cudaSuccess) return false;
        streaming_volume_ = d_volume;
        received_views_ = 0;
        streaming_failed_ = false;
        return true;
    }

    // Synchronous packet consumption: both Host and Device inputs can be
    // reused after return. Physical and mapped projection storage is bounded.
    bool enqueueBatch(const CylFdkProjectionBatch& batch)
    {
        const size_t per_view = static_cast<size_t>(channels_) * rows_;
        if (!prepared_ || !streaming_volume_ || streaming_failed_ || !batch.data ||
            batch.view_count <= 0 || batch.first_view != received_views_ ||
            batch.view_count > views_ - received_views_ ||
            (batch.location != EMemoryLocation::Host && batch.location != EMemoryLocation::Device) ||
            static_cast<size_t>(batch.view_count) >
                (std::numeric_limits<size_t>::max() / sizeof(float)) / per_view ||
            batch.element_count < static_cast<size_t>(batch.view_count) * per_view)
            return false;
        if (batch.location == EMemoryLocation::Host && !physical_chunk_) {
            physical_chunk_ = memory_.allocateDevice3D<float>(channels_, rows_,
                chunk_views_, device_id_, false);
            if (!physical_chunk_) return failStreaming_();
        }
        for (int offset = 0; offset < batch.view_count;) {
            const int count = std::min(chunk_views_, batch.view_count - offset);
            const float* source = batch.data + static_cast<size_t>(offset) * per_view;
            if (batch.location == EMemoryLocation::Host) {
                if (cudaMemcpyAsync(physical_chunk_.data(), source,
                        static_cast<size_t>(count) * per_view * sizeof(float),
                        cudaMemcpyHostToDevice, stream_) != cudaSuccess)
                    return failStreaming_();
                source = physical_chunk_.data();
            }
            if (!mapper_.applyBatch(source, mapped_.data(), count, stream_) ||
                !fdk_.processBatchSync({mapped_.data(), nullptr, nullptr, count,
                    EMemoryLocation::Device}, streaming_volume_, false))
                return failStreaming_();
            offset += count;
        }
        received_views_ += batch.view_count;
        return true;
    }

    bool completeStreaming()
    {
        if (!prepared_ || !streaming_volume_ || streaming_failed_ ||
            received_views_ != views_ || !fdk_.complete()) return false;
        streaming_volume_ = nullptr;
        return true;
    }

    bool cancelStreaming()
    {
        if (stream_ && cudaStreamSynchronize(stream_) != cudaSuccess) return false;
        fdk_.reset();
        streaming_volume_ = nullptr;
        received_views_ = 0;
        streaming_failed_ = false;
        return true;
    }

    bool streamingActive() const { return streaming_volume_ != nullptr; }
    int receivedStreamingViews() const { return received_views_; }
    int projectionWorkspaceViews() const { return chunk_views_; }

    void release()
    {
        if (stream_) cudaStreamSynchronize(stream_);
        fdk_.release();
        mapper_.release();
        mapped_ = {};
        physical_chunk_ = {};
        streaming_volume_ = nullptr;
        received_views_ = 0;
        streaming_failed_ = false;
        chunk_views_ = device_id_ = 0;
        volume_elements_ = 0;
        channels_ = rows_ = views_ = 0;
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    bool failStreaming_() {
        streaming_failed_ = true;
        cudaStreamSynchronize(stream_);
        return false;
    }
    ProjectionMapper mapper_{};
    ::YK::FdkPipeline fdk_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> mapped_{};
    Mem::DeviceLinearBuffer3D<float> physical_chunk_{};
    float* streaming_volume_ = nullptr;
    int received_views_ = 0, chunk_views_ = 0, device_id_ = 0;
    size_t volume_elements_ = 0;
    bool streaming_failed_ = false;
    int channels_ = 0, rows_ = 0, views_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

} // namespace YK::CylFpBp::Analytic
