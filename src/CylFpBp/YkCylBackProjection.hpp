#pragma once

#include <memory>
#include <vector>

#include "CylFpBp/YkCylBackOperator.hpp"
#include "CylFpBp/YkCylFdkBackprojector.hpp"
#include "CylFpBp/YkCylVoxelDrivenBackprojector.hpp"
#include "common/YkExecutionContext.hpp"

namespace YK::CylFpBp {

enum class EBackProjection {
    Joseph,
    VoxelDrivenV3,
    Fdk,
    FdkMatched
};

// Cyl 专用反投影接口。所有实现接收相同的线性设备内存，内部自行维护
// 投影纹理和算法预计算，调用方无需了解某一种 BP 的上传流程。
class IBackProjection {
public:
    virtual ~IBackProjection() = default;

    virtual bool prepare(const SVolGeom& volume_geometry, int channels,
        int rows, const std::vector<SCylConeProjGeomVec>& geometry,
        const Config& config, ResourceContext& resources) = 0;
    virtual bool apply(const float* d_projection, float* d_volume,
        bool accumulate, ResourceContext& resources) = 0;
    virtual void release() = 0;
};

class JosephBackProjection final : public IBackProjection {
public:
    ~JosephBackProjection() override { release(); }

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
        operator_ = makeBackOperator();
        if (!prepared || !operator_->prepare(prepared, config)) {
            release();
            return false;
        }
        projection_texture_ = Mem::TextureController::createEmptyTex3D(
            channels, rows, static_cast<int>(geometry.size()),
            cudaFilterModePoint, cudaAddressModeBorder);
        if (!projection_texture_.valid()) {
            release();
            return false;
        }
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_projection, float* d_volume, bool accumulate,
        ResourceContext& resources) override
    {
        if (!prepared_ || !d_projection || !d_volume ||
            resources.device() != device_id_ || resources.stream() != stream_)
            return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(projection_texture_,
            d_projection, channels_, rows_, views_, stream_);
        return operator_->backproject(projection_texture_, d_volume, stream_,
            accumulate);
    }

    void release() override
    {
        // 算子先等待最后一次 kernel，再释放 kernel 读取的投影纹理。
        if (operator_) operator_->release();
        operator_.reset();
        projection_texture_ = {};
        stream_ = nullptr;
        channels_ = rows_ = views_ = 0;
        device_id_ = 0;
        prepared_ = false;
    }

private:
    std::unique_ptr<IBackOperator> operator_{};
    Mem::Tex3DHandle projection_texture_{};
    cudaStream_t stream_ = nullptr;
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool prepared_ = false;
};

class VoxelDrivenBackProjection final : public IBackProjection {
public:
    ~VoxelDrivenBackProjection() override { release(); }

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const Config&, ResourceContext& resources) override
    {
        release();
        if (!resources.stream()) return false;
        device_id_ = resources.device();
        stream_ = resources.stream();
        if (!projector_.prepare(volume_geometry, channels, rows, geometry,
                stream_, device_id_)) {
            release();
            return false;
        }
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_projection, float* d_volume, bool accumulate,
        ResourceContext& resources) override
    {
        if (!prepared_ || !d_projection || !d_volume ||
            resources.device() != device_id_ || resources.stream() != stream_)
            return false;
        return projector_.uploadProjection(d_projection) &&
            projector_.backproject(d_volume, accumulate);
    }

    void release() override
    {
        projector_.release();
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

private:
    VoxelDrivenBackprojectorV3 projector_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
};

template<bool Matched>
class FdkBackProjectionImpl final : public IBackProjection {
public:
    ~FdkBackProjectionImpl() override { release(); }

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry,
        const Config&, ResourceContext& resources) override
    {
        release();
        if (!resources.stream()) return false;
        device_id_ = resources.device();
        stream_ = resources.stream();
        if (!projector_.prepare(volume_geometry, channels, rows, geometry,
                stream_, device_id_)) {
            release();
            return false;
        }
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_projection, float* d_volume, bool accumulate,
        ResourceContext& resources) override
    {
        if (!prepared_ || !d_projection || !d_volume ||
            resources.device() != device_id_ || resources.stream() != stream_)
            return false;
        return projector_.uploadProjection(d_projection) &&
            projector_.backproject(d_volume, accumulate);
    }

    void release() override
    {
        projector_.release();
        stream_ = nullptr;
        device_id_ = 0;
        prepared_ = false;
    }

private:
    detail::CylFdkBackprojectorImpl<Matched> projector_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool prepared_ = false;
};

using FdkBackProjection = FdkBackProjectionImpl<false>;
using FdkMatchedBackProjection = FdkBackProjectionImpl<true>;

inline std::unique_ptr<IBackProjection> makeBackProjection(
    EBackProjection model)
{
    switch (model) {
    case EBackProjection::Joseph:
        return std::make_unique<JosephBackProjection>();
    case EBackProjection::VoxelDrivenV3:
        return std::make_unique<VoxelDrivenBackProjection>();
    case EBackProjection::Fdk:
        return std::make_unique<FdkBackProjection>();
    case EBackProjection::FdkMatched:
        return std::make_unique<FdkMatchedBackProjection>();
    }
    return {};
}

} // namespace YK::CylFpBp
