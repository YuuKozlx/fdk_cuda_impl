#pragma once

#include <cmath>
#include <vector>

#include "CylFpBp/analytic/YkCylFdkPipeline.hpp"
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
// 源中心、半径为 SDD 的等角虚拟柱面，再交给现有 FDK 管线。这样
// R=SDD 与 R!=SDD 走同一条解析重建路径，而迭代 FP/BP 仍直接使用
// 原始物理几何，不会意外引入重排或 FDK 权重。
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

        if (!mapper_.prepare(canonical.projection_map) ||
            !fdk_.prepare(volume, channels, rows, canonical.canonical_geometry,
                filter, stream, device_id)) {
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
        return fdk_.reconstruct(mapped_.data(), volume, clear_output);
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
    ::YK::CylFpBp::FdkPipeline fdk_{};
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> mapped_{};
    int channels_ = 0, rows_ = 0, views_ = 0;
    cudaStream_t stream_ = nullptr;
    bool prepared_ = false;
};

} // namespace YK::CylFpBp::Analytic
