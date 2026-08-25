#pragma once

#include <cmath>
#include <memory>
#include <vector>

#include "BP/YkBPGpuContext_Siddon.hpp"
#include "BP/kernels/YkBpFdkLaunch.cuh"
#include "BP/kernels/YkBpJosephLaunch.cuh"
#include "BP/kernels/YkBPSiddonLaunch.cuh"
#include "FP/YkFPGpuContext.hpp"
#include "FP/kernels/YkFPLaunch.cuh"
#include "FDK/YkFDKVecGeoDerived.hpp"
#include "common/YkExecutionContext.hpp"
#include "global/YkCudaTextureController.hpp"

namespace YK {

// FP/BP 是计算算子，不是重建 runner。调用返回后 kernel 仍可在绑定 stream 上
// 执行；算子只保留 kernel 依赖的纹理和几何设备缓冲，并用完成事件管理其生命周期。
// 输入输出设备缓冲仍由调用方持有，必要时可通过 DeviceTensorDumper 显式导出。
class IForwardOperator {
public:
    virtual ~IForwardOperator() = default;
    virtual bool prepare(const GeometryContext&, ResourceContext&) = 0;
    virtual bool apply(const float* d_volume, const SCBCTParams& batch,
        float* d_projection, ResourceContext&) = 0;
    virtual void release() = 0;
};

class IBackOperator {
public:
    virtual ~IBackOperator() = default;
    virtual bool prepare(const GeometryContext&, ResourceContext&) = 0;
    virtual bool apply(const float* d_projection, const SCBCTParams& batch,
        float* d_volume, bool clear_volume, ResourceContext&) = 0;
    virtual void release() = 0;
};

namespace detail {

inline bool isForwardTask(ETask task)
{
    return task == ETask::FP_Joseph || task == ETask::FP_Siddon;
}

inline bool isBackTask(ETask task)
{
    return task == ETask::BP_Siddon_RayDriven || task == ETask::BP_Siddon_VoxDriven ||
        task == ETask::BP_Joseph || task == ETask::BP_Joseph_v2 ||
        task == ETask::BP_Joseph_v3 || task == ETask::BP_FDK ||
        task == ETask::BP_FDK_matched;
}

inline void buildCircularViews(const SCBCTParams& p, std::vector<SConeProjGeomVec>& views)
{
    const auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };
    views.resize(p.iPAng);
    build_circular_vec_geometry_from_theta(views, p.angle_list, p.iPAng,
        p.iPU, p.iPV, p.du_mm, p.dv_mm, p.SID, p.SDD - p.SID,
        f3(p.offsetU_mm, 0.f, p.offsetV_mm),
        f3(rad2deg(p.tiltu_angle_rad), rad2deg(p.tiltn_angle_rad),
           rad2deg(p.tiltv_angle_rad)));
}

// External geometry is immutable session data.  A regular FP/BP call may use
// the whole sequence, or a contiguous angle-identifiable subset.  Iterative
// algorithms currently use circular SCBCTParams and therefore take the other
// branch; this avoids silently inventing geometry for arbitrary trajectories.
inline bool resolveViews(const std::vector<SConeProjGeomVec>& all,
    const SCBCTParams& p, std::vector<SConeProjGeomVec>& out)
{
    if (all.empty()) {
        buildCircularViews(p, out);
        return true;
    }
    if (static_cast<int>(all.size()) == p.iPAng) {
        out = all;
        return true;
    }
    if (static_cast<int>(p.angle_list.size()) != p.iPAng) return false;
    out.clear();
    size_t cursor = 0;
    for (float angle : p.angle_list) {
        while (cursor < all.size() && std::fabs(all[cursor].angle.x - angle) > 1e-6f)
            ++cursor;
        if (cursor == all.size()) return false;
        out.push_back(all[cursor++]);
    }
    return true;
}

inline GeometryContext makeSubsetGeometry(const SCBCTParams& batch,
    const std::vector<SConeProjGeomVec>& views)
{
    GeometryContext geometry;
    geometry.initialize(batch, views);
    return geometry;
}

} // namespace detail

class ForwardOperator final : public IForwardOperator {
public:
    explicit ForwardOperator(ETask kind) : kind_(kind) {}
    ~ForwardOperator() override { release(); }

    bool prepare(const GeometryContext& geometry, ResourceContext& resources) override
    {
        if (!detail::isForwardTask(kind_)) return false;
        release();
        device_id_ = resources.device();
        stream_ = resources.stream();
        YK_CUDA_CHECK(cudaSetDevice(device_id_));
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        geometry_ = geometry;
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_volume, const SCBCTParams& batch,
        float* d_projection, ResourceContext& resources) override
    {
        if (!prepared_ || !d_volume || !d_projection || batch.iPAng <= 0) return false;
        if (resources.device() != device_id_ || resources.stream() != stream_) return false;
        retireContext_();
        std::vector<SConeProjGeomVec> views;
        if (!detail::resolveViews(geometry_.allGeometry(), batch, views)) return false;

        const SVolGeom vol = geometry_.volumeGeometry();
        gpu_ = std::make_unique<Fp::FpGpuContext>();
        gpu_->init(d_volume, vol, views, resources.device());
        const size_t count = static_cast<size_t>(batch.iPAng) * batch.iPV * batch.iPU;
        YK_CUDA_CHECK(cudaMemsetAsync(d_projection, 0, count * sizeof(float), resources.stream()));
        if (kind_ == ETask::FP_Joseph) {
            Fp::fp_joseph_launch(gpu_->volTex.tex, gpu_->geo.h_views_vec(), gpu_->geo.d_views_vox(),
                d_projection, vol, batch.iPAng, batch.iPU, batch.iPV, false,
                resources.stream(), Fp::FpStepSuperSample::x1);
        } else {
            Fp::fp_siddon_launch(gpu_->volTex.tex, d_projection, gpu_->geo.d_views(), vol,
                batch.iPU, batch.iPV, batch.iPAng, false, resources.stream());
        }
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
        return true;
    }

    void release() override
    {
        if (completion_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            if (completion_recorded_) YK_CUDA_CHECK(cudaEventSynchronize(completion_));
        }
        gpu_.reset();
        if (completion_) YK_CUDA_CHECK(cudaEventDestroy(completion_));
        completion_ = nullptr;
        completion_recorded_ = false;
        stream_ = nullptr;
        prepared_ = false;
        geometry_ = {};
    }

private:
    void retireContext_()
    {
        if (!gpu_) return;
        // 只等待本算子上一次 kernel 的完成点，不会等待随后排入同一 stream 的
        // 其他工作；因此可以安全销毁旧纹理/几何缓冲而不引入全流栅栏。
        if (completion_recorded_) YK_CUDA_CHECK(cudaEventSynchronize(completion_));
        gpu_.reset();
        completion_recorded_ = false;
    }

    ETask kind_;
    GeometryContext geometry_{};
    std::unique_ptr<Fp::FpGpuContext> gpu_{};
    cudaEvent_t completion_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool completion_recorded_ = false;
    bool prepared_ = false;
};

class BackOperator final : public IBackOperator {
public:
    explicit BackOperator(ETask kind) : kind_(kind) {}
    ~BackOperator() override { release(); }

    bool prepare(const GeometryContext& geometry, ResourceContext& resources) override
    {
        if (!detail::isBackTask(kind_)) return false;
        release();
        device_id_ = resources.device();
        stream_ = resources.stream();
        YK_CUDA_CHECK(cudaSetDevice(device_id_));
        YK_CUDA_CHECK(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
        geometry_ = geometry;
        prepared_ = true;
        return true;
    }

    bool apply(const float* d_projection, const SCBCTParams& batch,
        float* d_volume, bool clear_volume, ResourceContext& resources) override
    {
        if (!prepared_ || !d_projection || !d_volume || batch.iPAng <= 0) return false;
        if (resources.device() != device_id_ || resources.stream() != stream_) return false;
        retireContext_();
        std::vector<SConeProjGeomVec> views;
        if (!detail::resolveViews(geometry_.allGeometry(), batch, views)) return false;

        std::vector<SFDKGeoParamPerView> derived(views.size());
        GeoDerivedManagerVec{}.build_geo_params(batch.iPU, batch.iPV,
            batch.scan_range_rad, views, derived);
        const SVolGeom vol = geometry_.volumeGeometry();
        gpu_ = std::make_unique<Bp::BpSiddonGpuContext>();
        gpu_->initNoTex(d_projection, vol, views, derived, resources.stream(), resources.device());
        const bool accumulate = !clear_volume;
        const int na = batch.iPAng;

        switch (kind_) {
        case ETask::BP_Siddon_RayDriven:
            if (clear_volume) YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0,
                static_cast<size_t>(batch.iVX) * batch.iVY * batch.iVZ * sizeof(float), resources.stream()));
            Bp::bp_siddon_launch(gpu_->d_sino_raw, d_volume, gpu_->geo.d_views_world(), vol,
                batch.iPU, batch.iPV, na, resources.stream());
            break;
        case ETask::BP_Siddon_VoxDriven:
            Bp::bp_siddon_voxel_launch(d_projection, d_volume, gpu_->geo.d_views_world(), vol,
                batch.iPU, batch.iPV, na, accumulate, resources.stream());
            break;
        case ETask::BP_Joseph: {
            gpu_->sinoTex = Mem::TextureController::createTex3DFromDevice(
                gpu_->d_sino_raw, batch.iPU, batch.iPV, na);
            Bp::joseph_bp_launch(gpu_->sinoTex.tex, gpu_->geo.h_views_world_vec(), gpu_->geo.d_views_vox(), d_volume,
                vol, na, batch.iPU, batch.iPV, accumulate, resources.stream());
            break;
        }
        case ETask::BP_Joseph_v2: {
            gpu_->sinoTex = Mem::TextureController::createTex3DFromDevice(
                gpu_->d_sino_raw, batch.iPU, batch.iPV, na);
            Bp::joseph_bp_v2_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), d_volume, vol,
                na, batch.iPU, batch.iPV, accumulate, resources.stream());
            break;
        }
        case ETask::BP_Joseph_v3: {
            gpu_->sinoTex = Mem::TextureController::createTex3DFromDevice(
                gpu_->d_sino_raw, batch.iPU, batch.iPV, na);
            Bp::joseph_bp_v3_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), gpu_->geo.d_coeffs_data(),
                d_volume, vol, na, accumulate, resources.stream());
            break;
        }
        case ETask::BP_FDK:
        case ETask::BP_FDK_matched: {
            gpu_->sinoTex = Mem::TextureController::createTex3DFromDevice(
                gpu_->d_sino_raw, batch.iPU, batch.iPV, na);
            if (kind_ == ETask::BP_FDK)
                Bp::fdk_bp_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), gpu_->geo.d_coeffs_data(),
                    d_volume, vol, na, accumulate, resources.stream());
            else
                Bp::fdk_matched_bp_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), gpu_->geo.d_coeffs_data(),
                    d_volume, vol, na, accumulate, resources.stream());
            break;
        }
        default: return false;
        }
        YK_CUDA_CHECK(cudaEventRecord(completion_, stream_));
        completion_recorded_ = true;
        return true;
    }

    void release() override
    {
        if (completion_) {
            YK_CUDA_CHECK(cudaSetDevice(device_id_));
            if (completion_recorded_) YK_CUDA_CHECK(cudaEventSynchronize(completion_));
        }
        gpu_.reset();
        if (completion_) YK_CUDA_CHECK(cudaEventDestroy(completion_));
        completion_ = nullptr;
        completion_recorded_ = false;
        stream_ = nullptr;
        prepared_ = false;
        geometry_ = {};
    }

private:
    void retireContext_()
    {
        if (!gpu_) return;
        // texture object、cudaArray 和预计算几何均由 gpu_ 持有；必须等上一次
        // 反投 kernel 越过完成事件后才能复用或释放。
        if (completion_recorded_) YK_CUDA_CHECK(cudaEventSynchronize(completion_));
        gpu_.reset();
        completion_recorded_ = false;
    }

    ETask kind_;
    GeometryContext geometry_{};
    std::unique_ptr<Bp::BpSiddonGpuContext> gpu_{};
    cudaEvent_t completion_ = nullptr;
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    bool completion_recorded_ = false;
    bool prepared_ = false;
};

inline std::unique_ptr<IForwardOperator> makeForwardOperator(ETask kind)
{ return std::make_unique<ForwardOperator>(kind); }
inline std::unique_ptr<IBackOperator> makeBackOperator(ETask kind)
{ return std::make_unique<BackOperator>(kind); }

// Existing iterative solvers retain their compact init/run shape while all
// actual dispatch is now performed by the common operators above.
class ForwardOperatorAdapter {
public:
    bool init(const SCBCTParams& params, ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params)) return false;
        resources_.attach(stream, device);
        op_ = makeForwardOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool init(const SCBCTParams& params, const std::vector<SConeProjGeomVec>& geometry,
        ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params, geometry)) return false;
        resources_.attach(stream, device);
        op_ = makeForwardOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool run(const float* d_volume, const SCBCTParams& batch, float* d_projection, cudaStream_t)
    { return op_ && op_->apply(d_volume, batch, d_projection, resources_); }
    void release() { if (op_) op_->release(); op_.reset(); resources_.release(); }
private:
    GeometryContext geometry_{};
    ResourceContext resources_{};
    std::unique_ptr<IForwardOperator> op_{};
};

class BackOperatorAdapter {
public:
    bool init(const SCBCTParams& params, ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params)) return false;
        resources_.attach(stream, device);
        op_ = makeBackOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool init(const SCBCTParams& params, const std::vector<SConeProjGeomVec>& geometry,
        ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params, geometry)) return false;
        resources_.attach(stream, device);
        op_ = makeBackOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool run(const float* d_projection, const SCBCTParams& batch, float* d_volume,
        cudaStream_t, bool clear_volume)
    { return op_ && op_->apply(d_projection, batch, d_volume, clear_volume, resources_); }
    void release() { if (op_) op_->release(); op_.reset(); resources_.release(); }
private:
    GeometryContext geometry_{};
    ResourceContext resources_{};
    std::unique_ptr<IBackOperator> op_{};
};

} // namespace YK
