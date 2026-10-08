#pragma once

#include <memory>
#include <vector>

#include "CylFpBp/BP/YkCylBackOperator.hpp"
#include "CylFpBp/BP/YkCylFdkBackprojector.hpp"
#include "CylFpBp/BP/YkCylJosephV3Strategy.hpp"
#include "CylFpBp/BP/YkCylSiddonVoxelStrategy.hpp"
#include "common/YkExecutionContext.hpp"

namespace YK::CylFpBp {

enum class EBackProjection {
    Joseph,
    // 兼容默认：朴素 ray-driven，保持与 Siddon FP 的严格离散转置。
    Siddon,
    SiddonV2,
    SiddonV3,
    SiddonRayDriven,
    JosephV3,
    Fdk,
    FdkMatched
};

// Cyl 专用反投影接口。所有模型接收相同的线性设备内存，调用方只选择
// EBackProjection，不接触某个模型的纹理、预计算或上传过程。
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

// 唯一的 Cyl BP 门面。模型枚举决定内部策略，不再为每一种 BP 派生一个
// IBackProjection 包装类。detail projector 只保存各自必要的预计算和 CUDA
// 资源，不向业务层暴露。
class BackProjection final : public IBackProjection {
public:
    explicit BackProjection(EBackProjection model) : model_(model) {}
    ~BackProjection() override { release(); }
    BackProjection(const BackProjection&) = delete;
    BackProjection& operator=(const BackProjection&) = delete;

    bool prepare(const SVolGeom& volume_geometry, int channels, int rows,
        const std::vector<SCylConeProjGeomVec>& geometry, const Config& config,
        ResourceContext& resources) override
    {
        release();
        if (!resources.stream() || channels <= 0 || rows <= 0 ||
            geometry.empty()) return false;

        stream_ = resources.stream();
        device_id_ = resources.device();
        channels_ = channels;
        rows_ = rows;
        views_ = static_cast<int>(geometry.size());
        YK_CUDA_CHECK(cudaSetDevice(device_id_));

        bool ok = false;
        switch (model_) {
        case EBackProjection::Joseph:
        case EBackProjection::Siddon:
        case EBackProjection::SiddonRayDriven: {
            const auto prepared = detail::prepareJosephGeometry(volume_geometry,
                channels, rows, geometry, device_id_);
            operator_ = makeCylBackOperator();
            ok = prepared && operator_ && operator_->prepare(prepared, config);
            if (ok) {
                projection_texture_ = Mem::TextureController::createEmptyTex3D(
                    channels, rows, views_, cudaFilterModePoint,
                    cudaAddressModeBorder);
                ok = projection_texture_.valid();
            }
            break;
        }
        case EBackProjection::SiddonV2:
        case EBackProjection::SiddonV3:
            ok = siddon_voxel_.prepare(volume_geometry, channels, rows,
                geometry, model_ == EBackProjection::SiddonV2
                    ? detail::ESiddonBackprojectorVersion::V2
                    : detail::ESiddonBackprojectorVersion::V3,
                stream_, device_id_);
            break;
        case EBackProjection::JosephV3:
            ok = joseph_v3_.prepare(volume_geometry, channels, rows, geometry,
                stream_, device_id_);
            break;
        case EBackProjection::Fdk:
            ok = fdk_.prepare(volume_geometry, channels, rows, geometry,
                stream_, device_id_);
            break;
        case EBackProjection::FdkMatched:
            ok = fdk_matched_.prepare(volume_geometry, channels, rows, geometry,
                stream_, device_id_);
            break;
        }

        if (!ok) {
            release();
            return false;
        }
        prepared_ = true;
        return true;
    }

    bool apply(const float* projection, float* volume, bool accumulate,
        ResourceContext& resources) override
    {
        if (!prepared_ || !projection || !volume ||
            resources.device() != device_id_ || resources.stream() != stream_)
            return false;

        switch (model_) {
        case EBackProjection::Joseph:
            return uploadCommon_(projection) && operator_->backproject(
                projection_texture_, volume, stream_, accumulate);
        case EBackProjection::Siddon:
        case EBackProjection::SiddonRayDriven:
            return uploadCommon_(projection) && operator_->backprojectSiddon(
                projection_texture_, volume, stream_, accumulate);
        case EBackProjection::SiddonV2:
        case EBackProjection::SiddonV3:
            return siddon_voxel_.uploadProjection(projection) &&
                siddon_voxel_.backproject(volume, accumulate);
        case EBackProjection::JosephV3:
            return joseph_v3_.uploadProjection(projection) &&
                joseph_v3_.backproject(volume, accumulate);
        case EBackProjection::Fdk:
            return fdk_.uploadProjection(projection) &&
                fdk_.backproject(volume, accumulate);
        case EBackProjection::FdkMatched:
            return fdk_matched_.uploadProjection(projection) &&
                fdk_matched_.backproject(volume, accumulate);
        }
        return false;
    }

    void release() override
    {
        if (operator_) operator_->release();
        operator_.reset();
        projection_texture_ = {};
        siddon_voxel_.release();
        joseph_v3_.release();
        fdk_.release();
        fdk_matched_.release();
        stream_ = nullptr;
        channels_ = rows_ = views_ = 0;
        device_id_ = 0;
        prepared_ = false;
    }

private:
    bool uploadCommon_(const float* projection)
    {
        if (!projection_texture_.valid()) return false;
        Mem::TextureController::updateTex3DFromDeviceAsync(projection_texture_,
            projection, channels_, rows_, views_, stream_);
        return true;
    }

    EBackProjection model_{};
    std::unique_ptr<ICylBackOperator> operator_{};
    Mem::Tex3DHandle projection_texture_{};
    detail::SiddonVoxelStrategy siddon_voxel_{};
    detail::JosephV3Strategy joseph_v3_{};
    detail::CylFdkBackprojectorImpl<false> fdk_{};
    detail::CylFdkBackprojectorImpl<true> fdk_matched_{};
    cudaStream_t stream_ = nullptr;
    int channels_ = 0;
    int rows_ = 0;
    int views_ = 0;
    int device_id_ = 0;
    bool prepared_ = false;
};

inline std::unique_ptr<IBackProjection> makeBackProjection(
    EBackProjection model)
{
    return std::make_unique<BackProjection>(model);
}

} // namespace YK::CylFpBp
