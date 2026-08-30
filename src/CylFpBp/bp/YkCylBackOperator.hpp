#pragma once

#include <memory>
#include <utility>

#include "CylFpBp/YkCylJosephGeometry.hpp"
#include "global/YkCudaTextureController.hpp"

namespace YK::CylFpBp {

// Cyl 专用底层反投契约。Joseph FP/BP 可通过 prepare() 接收同一个几何
// 句柄，从而在纯虚接口后仍只上传一份逐视图设备几何。
class ICylBackOperator {
public:
    virtual ~ICylBackOperator() = default;

    virtual bool prepare(detail::JosephGeometryHandle geometry,
        const Config& config = {}) = 0;
    virtual bool backproject(const Mem::Tex3DHandle& texture, float* volume,
        cudaStream_t stream, bool accumulate = false) const = 0;
    virtual bool backprojectSiddon(const Mem::Tex3DHandle& texture, float* volume,
        cudaStream_t stream, bool accumulate = false) const = 0;
    virtual void release() = 0;
};

// 圆柱探测器 Joseph 反投算子。SiddonRayDriven 目前由独立 kernel 提供；
// FDK 和 Joseph V3 等其他 BP 仍由各自专用类管理，避免在一个入口中
// 混入不同权重语义。
class CylBackOperator final : public ICylBackOperator {
public:
    ~CylBackOperator() override { release(); }

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const Config& config = {}, int device_id = 0)
    {
        auto prepared = detail::prepareJosephGeometry(volume_geometry, channels,
            rows, geometry, device_id);
        return prepared && prepare(std::move(prepared), config);
    }

    bool prepare(detail::JosephGeometryHandle geometry,
        const Config& config = {}) override
    {
        release();
        if (!geometry || config.samples_per_voxel <= 0.f) return false;
        geometry_ = std::move(geometry);
        config_ = config;
        return completion_.prepare(geometry_->device());
    }

    bool backproject(const Mem::Tex3DHandle& texture, float* volume,
        cudaStream_t stream, bool accumulate = false) const override
    {
        if (!geometry_ || !volume || !stream ||
            !texture.matches(geometry_->channels(), geometry_->rows(),
                geometry_->views(), cudaFilterModePoint))
            return false;
        completion_.waitBefore(stream);
        detail::launch_main_axis_backproject_texture(texture.tex, volume,
            geometry_->data(), geometry_->views(), geometry_->rows(),
            geometry_->channels(), geometry_->volumeGeometry(), config_, stream,
            accumulate);
        completion_.record(stream);
        return true;
    }

    bool backprojectSiddon(const Mem::Tex3DHandle& texture, float* volume,
        cudaStream_t stream, bool accumulate = false) const override
    {
        if (!geometry_ || !volume || !stream ||
            !texture.matches(geometry_->channels(), geometry_->rows(),
                geometry_->views(), cudaFilterModePoint)) return false;
        completion_.waitBefore(stream);
        detail::launch_cyl_siddon_backproject(texture.tex, volume,
            geometry_->data(), geometry_->siddonChannelRays(),
            geometry_->views(), geometry_->rows(), geometry_->channels(),
            geometry_->volumeGeometry(), accumulate, stream);
        completion_.record(stream);
        return true;
    }

    void release() override
    {
        completion_.release();
        geometry_.reset();
        config_ = {};
    }

private:
    detail::JosephGeometryHandle geometry_{};
    Config config_{};
    mutable detail::OperatorCompletion completion_{};
};

inline std::unique_ptr<ICylBackOperator> makeCylBackOperator()
{
    return std::make_unique<CylBackOperator>();
}

} // namespace YK::CylFpBp
