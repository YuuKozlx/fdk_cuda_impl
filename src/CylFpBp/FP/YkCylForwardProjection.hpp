#pragma once

#include <memory>
#include <vector>

#include "CylFpBp/FP/YkCylForwardOperator.hpp"
#include "common/YkExecutionContext.hpp"

namespace YK::CylFpBp {

// Cyl 专有的正投模型。它不复用 Flat 的 ETask，也不接受平板
// SConeProjGeomVec，避免两种探测器表面在同一个接口中产生含糊语义。
enum class EForwardProjection {
    Joseph,
    JosephMatchedReference,
    Siddon
};

class IForwardProjection {
public:
    virtual ~IForwardProjection() = default;

    virtual bool prepare(const SVolGeom& volume_geometry, int channels,
        int rows, const std::vector<SCylConeProjGeomVec>& geometry,
        const Config& config, ResourceContext& resources) = 0;

    // 输入体积和输出投影均为调用方管理的线性设备内存。实现内部负责建立
    // 正确过滤模式的体积纹理；accumulate 用于在线分包或多段轨迹累加。
    virtual bool apply(const float* d_volume, float* d_projection,
        bool accumulate, ResourceContext& resources) = 0;

    virtual void release() = 0;
};

class ForwardProjection final : public IForwardProjection {
public:
    explicit ForwardProjection(EForwardProjection model) : model_(model) {}
    ~ForwardProjection() override { release(); }

    ForwardProjection(const ForwardProjection&) = delete;
    ForwardProjection& operator=(const ForwardProjection&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const Config& config, ResourceContext& resources) override
    {
        release();
        if (!resources.stream()) return false;
        device_id_ = resources.device();
        stream_ = resources.stream();
        YK_CUDA_CHECK(cudaSetDevice(device_id_));

        const auto prepared = detail::prepareJosephGeometry(volume_geometry,
            channels, rows, geometry, device_id_);
        operator_ = makeCylForwardOperator();
        if (!prepared || !operator_->prepare(prepared, config)) {
            release();
            return false;
        }

        const cudaTextureFilterMode filter =
            model_ == EForwardProjection::Joseph ? cudaFilterModeLinear :
                cudaFilterModePoint;
        volume_texture_ = Mem::TextureController::createEmptyTex3D(
            volume_geometry.Nx, volume_geometry.Ny, volume_geometry.Nz,
            filter, cudaAddressModeBorder);
        if (!volume_texture_.valid()) {
            operator_->release();
            operator_.reset();
            return false;
        }
        volume_geometry_ = volume_geometry;
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_volume, float* d_projection, bool accumulate,
        ResourceContext& resources) override
    {
        if (!prepared_ || !d_volume || !d_projection ||
            resources.device() != device_id_ ||
            resources.stream() != stream_)
            return false;

        // 所有上传和 kernel 固定在同一个 stream。连续 apply 时，前一次正投
        // 必然先于本次纹理更新完成，不需要设备级同步。
        Mem::TextureController::updateTex3DFromDeviceAsync(volume_texture_,
            d_volume, volume_geometry_.Nx, volume_geometry_.Ny,
            volume_geometry_.Nz, stream_);

        switch (model_) {
        case EForwardProjection::Joseph:
            return operator_->forward(volume_texture_, d_projection, stream_,
                accumulate);
        case EForwardProjection::JosephMatchedReference:
            return operator_->forwardMatchedReference(volume_texture_,
                d_projection, stream_, accumulate);
        case EForwardProjection::Siddon:
            return operator_->forwardSiddon(volume_texture_, d_projection,
                stream_, accumulate);
        }
        return false;
    }

    void release() override
    {
        // Operator 先等待最后一次 kernel，再销毁被该 kernel 读取的纹理。
        if (operator_) operator_->release();
        operator_.reset();
        volume_texture_ = {};
        volume_geometry_ = {};
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

private:
    EForwardProjection model_;
    std::unique_ptr<ICylForwardOperator> operator_{};
    Mem::Tex3DHandle volume_texture_{};
    SVolGeom volume_geometry_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
};

inline std::unique_ptr<IForwardProjection> makeForwardProjection(
    EForwardProjection model)
{
    return std::make_unique<ForwardProjection>(model);
}

} // namespace YK::CylFpBp
