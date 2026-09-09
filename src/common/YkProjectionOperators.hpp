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
#include "YKCBCT/geometry/YkModularGeometryBuilder.hpp"
#include "common/YkExecutionContext.hpp"
#include "global/YkCudaTextureController.hpp"

namespace YK {

// FP/BP 是计算算子，不是重建 runner。调用返回后 kernel 仍可在绑定 stream 上
// 执行；算子只保留 kernel 依赖的纹理和几何设备缓冲，并用完成事件管理其生命周期。
// 输入输出设备缓冲仍由调用方持有，必要时可通过 DeviceTensorDumper 显式导出。
class IForwardOperator {
public:
    virtual ~IForwardOperator() = default;
    virtual bool prepare(const PreparedGeometry&, ResourceContext&) = 0;
    virtual bool apply(const float* d_volume, const SReconstructionParams& batch,
        float* d_projection, ResourceContext&) = 0;
    virtual void release() = 0;
};

class IBackOperator {
public:
    virtual ~IBackOperator() = default;
    virtual bool prepare(const PreparedGeometry&, ResourceContext&) = 0;
    virtual bool apply(const float* d_projection, const SReconstructionParams& batch,
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
        task == ETask::BP_Siddon_VoxDriven_v2 ||
        task == ETask::BP_Siddon_VoxDriven_v3 ||
        task == ETask::BP_Joseph || task == ETask::BP_Joseph_v2 ||
        task == ETask::BP_Joseph_v3 || task == ETask::BP_FDK ||
        task == ETask::BP_FDK_matched;
}

inline void buildCircularViews(const SReconstructionParams& p, std::vector<SConeProjGeomVec>& views)
{
    SCircularTrajectorySpec trajectory{};
    trajectory.angles_rad = p.scan.angles;
    trajectory.sid_mm = p.scan.sid_mm;
    trajectory.sdd_mm = p.scan.sdd_mm;
    trajectory.source_offset_mm = make_float3(p.scan.sourceOffsetX_mm,
        p.scan.sourceOffsetY_mm, p.scan.sourceOffsetZ_mm);
    SFlatDetectorSpec detector{};
    detector.channels = p.scan.Nu;
    detector.rows = p.scan.Nv;
    detector.channel_size_mm = p.scan.du_mm;
    detector.row_size_mm = p.scan.dv_mm;
    detector.pose.offset_unv_mm = make_float3(p.scan.offsetU_mm, 0.f,
        p.scan.offsetV_mm);
    detector.pose.tilt_u_rad = p.scan.tiltU_rad;
    detector.pose.tilt_v_rad = p.scan.tiltV_rad;
    detector.pose.tilt_n_rad = p.scan.tiltN_rad;
    if (!buildProjectionGeometry(trajectory, detector, views)) views.clear();
}

// External geometry is immutable session data.  A regular FP/BP call may use
// the whole sequence, or a contiguous angle-identifiable subset.  Iterative
// algorithms currently use circular SReconstructionParams and therefore take the other
// branch; this avoids silently inventing geometry for arbitrary trajectories.
inline bool resolveViews(const std::vector<SConeProjGeomVec>& all,
    const SReconstructionParams& p, std::vector<SConeProjGeomVec>& out)
{
    if (all.empty()) {
        buildCircularViews(p, out);
        return true;
    }
    if (static_cast<int>(all.size()) == p.scan.NAng) {
        out = all;
        return true;
    }
    if (static_cast<int>(p.scan.angles.size()) != p.scan.NAng) return false;
    out.clear();
    size_t cursor = 0;
    for (float angle : p.scan.angles) {
        while (cursor < all.size() && std::fabs(all[cursor].angle.x - angle) > 1e-6f)
            ++cursor;
        if (cursor == all.size()) return false;
        out.push_back(all[cursor++]);
    }
    return true;
}

inline PreparedGeometry makeSubsetGeometry(const SReconstructionParams& batch,
    const std::vector<SConeProjGeomVec>& views)
{
    PreparedGeometry geometry;
    geometry.initialize(batch, views);
    return geometry;
}

} // namespace detail

class ForwardOperator final : public IForwardOperator {
public:
    explicit ForwardOperator(ETask kind) : kind_(kind) {}
    ~ForwardOperator() override { release(); }

    bool prepare(const PreparedGeometry& geometry, ResourceContext& resources) override
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

    bool apply(const float* d_volume, const SReconstructionParams& batch,
        float* d_projection, ResourceContext& resources) override
    {
        if (!prepared_ || !d_volume || !d_projection || batch.scan.NAng <= 0) return false;
        if (resources.device() != device_id_ || resources.stream() != stream_) return false;
        retireContext_();
        std::vector<SConeProjGeomVec> views;
        if (!detail::resolveViews(geometry_.allGeometry(), batch, views)) return false;

        const SVolGeom vol = geometry_.volumeGeometry();
        gpu_ = std::make_unique<Fp::FpGpuContext>();
        gpu_->init(d_volume, vol, views, resources.device(),
            kind_ == ETask::FP_Siddon ? cudaFilterModePoint :
                cudaFilterModeLinear, resources.stream());
        const size_t count = static_cast<size_t>(batch.scan.NAng) * batch.scan.Nv * batch.scan.Nu;
        YK_CUDA_CHECK(cudaMemsetAsync(d_projection, 0, count * sizeof(float), resources.stream()));
        if (kind_ == ETask::FP_Joseph) {
            Fp::fp_joseph_launch(gpu_->volTex.tex, gpu_->geo.h_views_vec(), gpu_->geo.d_views_vox(),
                d_projection, vol, batch.scan.NAng, batch.scan.Nu, batch.scan.Nv, false,
                resources.stream(), Fp::FpStepSuperSample::x1);
        } else {
            Fp::fp_siddon_launch(gpu_->volTex.tex, d_projection, gpu_->geo.d_views(), vol,
                batch.scan.Nu, batch.scan.Nv, batch.scan.NAng, false, resources.stream());
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
    PreparedGeometry geometry_{};
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

    bool prepare(const PreparedGeometry& geometry, ResourceContext& resources) override
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

    bool apply(const float* d_projection, const SReconstructionParams& batch,
        float* d_volume, bool clear_volume, ResourceContext& resources) override
    {
        if (!prepared_ || !d_projection || !d_volume || batch.scan.NAng <= 0) return false;
        if (resources.device() != device_id_ || resources.stream() != stream_) return false;
        retireContext_();
        std::vector<SConeProjGeomVec> views;
        if (!detail::resolveViews(geometry_.allGeometry(), batch, views)) return false;

        std::vector<SFDKGeoParamPerView> derived(views.size());
        GeoDerivedManagerVec{}.build_geo_params(batch.scan.Nu, batch.scan.Nv,
            batch.scan.range_rad, views, derived);
        const SVolGeom vol = geometry_.volumeGeometry();
        gpu_ = std::make_unique<Bp::BpSiddonGpuContext>();
        gpu_->initNoTex(d_projection, vol, views, derived, resources.stream(), resources.device());
        const bool accumulate = !clear_volume;
        const int na = batch.scan.NAng;
        const auto makeProjectionTexture = [&](cudaTextureFilterMode filter) {
            auto texture = Mem::TextureController::createEmptyTex3D(
                batch.scan.Nu, batch.scan.Nv, na, filter, cudaAddressModeBorder);
            // 与上游投影写入和下游 BP 位于同一 stream，避免同步拷贝依赖
            // 默认流语义，也明确表明 cudaArray 是此时刻的数据快照。
            Mem::TextureController::updateTex3DFromDeviceAsync(texture,
                gpu_->d_sino_raw, batch.scan.Nu, batch.scan.Nv, na, resources.stream());
            return texture;
        };

        switch (kind_) {
        case ETask::BP_Siddon_RayDriven: {
            if (clear_volume) YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0,
                static_cast<size_t>(batch.volume.Nx) * batch.volume.Ny * batch.volume.Nz * sizeof(float), resources.stream()));
            // RayDriven 只读取离散投影样本；Point 避免任何隐式插值或
            // 硬件插值权重量化，Border 保证越界为 0。
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModePoint);
            Bp::bp_siddon_launch(gpu_->sinoTex.tex, d_volume,
                gpu_->geo.d_views_world(), vol,
                batch.scan.Nu, batch.scan.Nv, na, resources.stream());
            break;
        }
        case ETask::BP_Siddon_VoxDriven:
            Bp::bp_siddon_voxel_launch(d_projection, d_volume, gpu_->geo.d_views_world(), vol,
                batch.scan.Nu, batch.scan.Nv, na, accumulate, resources.stream());
            break;
        case ETask::BP_Siddon_VoxDriven_v2:
            // V2 使用连续探测器坐标和双线性插值。
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModeLinear);
            Bp::bp_siddon_voxel_v2_launch(gpu_->sinoTex.tex, d_volume,
                gpu_->geo.d_views_world(), vol, batch.scan.Nu, batch.scan.Nv,
                na, accumulate, resources.stream());
            break;
        case ETask::BP_Siddon_VoxDriven_v3:
            // V3 取最近探测器像素，必须使用 Point 纹理。
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModePoint);
            Bp::bp_siddon_voxel_v3_launch(gpu_->sinoTex.tex, d_volume,
                gpu_->geo.d_views_world(), vol, batch.scan.Nu, batch.scan.Nv,
                na, accumulate, resources.stream());
            break;
        case ETask::BP_Joseph: {
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModePoint);
            Bp::joseph_bp_launch(gpu_->sinoTex.tex, gpu_->geo.h_views_world_vec(), gpu_->geo.d_views_vox(), d_volume,
                vol, na, batch.scan.Nu, batch.scan.Nv, accumulate, resources.stream());
            break;
        }
        case ETask::BP_Joseph_v2: {
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModeLinear);
            Bp::joseph_bp_v2_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), d_volume, vol,
                na, batch.scan.Nu, batch.scan.Nv, accumulate, resources.stream());
            break;
        }
        case ETask::BP_Joseph_v3: {
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModeLinear);
            Bp::joseph_bp_v3_launch(gpu_->sinoTex.tex, gpu_->geo.d_views_world(), gpu_->geo.d_coeffs_data(),
                d_volume, vol, na, accumulate, resources.stream());
            break;
        }
        case ETask::BP_FDK:
        case ETask::BP_FDK_matched: {
            gpu_->sinoTex = makeProjectionTexture(cudaFilterModeLinear);
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
    PreparedGeometry geometry_{};
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
    bool init(const SReconstructionParams& params, ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params)) return false;
        resources_.attach(stream, device);
        op_ = makeForwardOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool init(const SReconstructionParams& params, const std::vector<SConeProjGeomVec>& geometry,
        ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params, geometry)) return false;
        resources_.attach(stream, device);
        op_ = makeForwardOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool run(const float* d_volume, const SReconstructionParams& batch, float* d_projection, cudaStream_t)
    { return op_ && op_->apply(d_volume, batch, d_projection, resources_); }
    void release() { if (op_) op_->release(); op_.reset(); resources_.release(); }
private:
    PreparedGeometry geometry_{};
    ResourceContext resources_{};
    std::unique_ptr<IForwardOperator> op_{};
};

class BackOperatorAdapter {
public:
    bool init(const SReconstructionParams& params, ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params)) return false;
        resources_.attach(stream, device);
        op_ = makeBackOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool init(const SReconstructionParams& params, const std::vector<SConeProjGeomVec>& geometry,
        ETask kind, int device, cudaStream_t stream)
    {
        if (!geometry_.initialize(params, geometry)) return false;
        resources_.attach(stream, device);
        op_ = makeBackOperator(kind);
        return op_->prepare(geometry_, resources_);
    }
    bool run(const float* d_projection, const SReconstructionParams& batch, float* d_volume,
        cudaStream_t, bool clear_volume)
    { return op_ && op_->apply(d_projection, batch, d_volume, clear_volume, resources_); }
    void release() { if (op_) op_->release(); op_.reset(); resources_.release(); }
private:
    PreparedGeometry geometry_{};
    ResourceContext resources_{};
    std::unique_ptr<IBackOperator> op_{};
};

} // namespace YK
