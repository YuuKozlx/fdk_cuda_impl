#pragma once

#include <memory>
#include <utility>

#include "CylFpBp/YkCylJosephGeometry.hpp"
#include "global/YkCudaTextureController.hpp"

namespace YK::CylFpBp {

// Cyl 专用底层正投契约。接口直接接收已经准备好的纹理，以便迭代器复用
// 纹理和共享几何；线性设备内存的便捷调用由 IForwardProjection 提供。
class IForwardOperator {
public:
    virtual ~IForwardOperator() = default;

    virtual bool prepare(detail::JosephGeometryHandle geometry,
        const Config& config = {}) = 0;
    virtual bool forward(const Mem::Tex3DHandle& texture, float* projection,
        cudaStream_t stream, bool accumulate = false) const = 0;
    virtual bool forwardSiddon(const Mem::Tex3DHandle& texture,
        float* projection, cudaStream_t stream,
        bool accumulate = false) const = 0;
    virtual bool forwardMatchedReference(const Mem::Tex3DHandle& texture,
        float* projection, cudaStream_t stream,
        bool accumulate = false) const = 0;
    virtual void release() = 0;
};

// 圆柱探测器正投算子。只暴露 FP 模型，不承担反投职责。
class ForwardOperator final : public IForwardOperator {
public:
    ~ForwardOperator() override { release(); }

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

    bool forward(const Mem::Tex3DHandle& texture, float* projection,
        cudaStream_t stream, bool accumulate = false) const override
    {
        if (!geometry_ || !projection || !stream ||
            !texture.matches(geometry_->volumeGeometry().Nx,
                geometry_->volumeGeometry().Ny,
                geometry_->volumeGeometry().Nz, cudaFilterModeLinear))
            return false;
        completion_.waitBefore(stream);
        detail::launch_main_axis_forward_texture(texture.tex, projection,
            geometry_->data(), geometry_->views(), geometry_->rows(),
            geometry_->channels(), geometry_->volumeGeometry(), config_, stream,
            accumulate);
        completion_.record(stream);
        return true;
    }

    bool forwardSiddon(const Mem::Tex3DHandle& texture, float* projection,
        cudaStream_t stream, bool accumulate = false) const override
    {
        if (!geometry_ || !projection || !stream ||
            !texture.matches(geometry_->volumeGeometry().Nx,
                geometry_->volumeGeometry().Ny,
                geometry_->volumeGeometry().Nz, cudaFilterModePoint))
            return false;
        completion_.waitBefore(stream);
        detail::launch_cyl_siddon_forward(texture.tex, projection,
            geometry_->data(), geometry_->views(), geometry_->rows(),
            geometry_->channels(), geometry_->volumeGeometry(), accumulate,
            stream);
        completion_.record(stream);
        return true;
    }

    bool forwardMatchedReference(const Mem::Tex3DHandle& texture,
        float* projection, cudaStream_t stream,
        bool accumulate = false) const override
    {
        if (!geometry_ || !projection || !stream ||
            !texture.matches(geometry_->volumeGeometry().Nx,
                geometry_->volumeGeometry().Ny,
                geometry_->volumeGeometry().Nz, cudaFilterModePoint))
            return false;
        completion_.waitBefore(stream);
        detail::launch_main_axis_forward_texture_matched(texture.tex, projection,
            geometry_->data(), geometry_->views(), geometry_->rows(),
            geometry_->channels(), geometry_->volumeGeometry(), config_, stream,
            accumulate);
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

inline std::unique_ptr<IForwardOperator> makeForwardOperator()
{
    return std::make_unique<ForwardOperator>();
}

} // namespace YK::CylFpBp
