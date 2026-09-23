#pragma once

#include <cmath>
#include <vector>

#include "FDK/YkFdkPipeline.hpp"
#include "CylFpBp/analytic/YkCylAnalyticProjectionMapper.hpp"
#include "CylFpBp/analytic/YkCylAnalyticGeometryCanonicalizer.hpp"
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
};

// 柱面解析重建的通用输入适配层。
//
// 物理探测器可以使用任意曲率半径 R；本适配器先将投影重采样到
// 物理柱面先重采样到内部零姿态平板，再交给标准 Flat-FDK 管线。
//
// 当前契约是标准同轴、无切向 U 偏移、无轴向 V 偏移的圆柱；逐视图
// 姿态、倾斜探测器和复杂轨迹应继续使用迭代算子。
class Reconstruction {
public:
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
            !(config.source_to_detector_mm > 0.f)) return false;

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
        const auto& corrected_geometry = canonical.corrected_geometry;

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
        if (!mapper_.prepare(canonical.projection_map) ||
            !fdk_.prepareWithGeometry(params, canonical.flat_geometry,
                static_cast<int>(physical_geometry.size()), stream, device_id)) {
            release();
            return false;
        }
        memory_ = Mem::MemoryController{};
        mapped_ = memory_.allocateDevice3D<float>(channels, rows,
            static_cast<int>(physical_geometry.size()), device_id, false);
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(physical_geometry.size());
        stream_ = stream;
        prepared_ = static_cast<bool>(mapped_);
        if (!prepared_) release();
        return prepared_;
    }

    bool reconstruct(const float* physical_projection, float* volume,
        bool clear_output = true)
    {
        if (!prepared_ || !physical_projection || !volume) return false;
        if (!mapper_.apply(physical_projection, mapped_.data(), stream_))
            return false;
        return fdk_.processBatchSync({ mapped_.data(), nullptr, nullptr, views_ },
            volume, clear_output);
    }

    void release()
    {
        fdk_.release();
        mapper_.release();
        mapped_ = {};
        channels_ = rows_ = views_ = 0;
        stream_ = nullptr;
        prepared_ = false;
    }

    bool isPrepared() const { return prepared_; }

private:
    ProjectionMapper mapper_{};
    ::YK::FdkPipeline fdk_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> mapped_{};
    int channels_ = 0, rows_ = 0, views_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

} // namespace YK::CylFpBp::Analytic
