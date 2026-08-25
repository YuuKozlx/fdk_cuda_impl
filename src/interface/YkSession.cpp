#include "YKCBCT/interface/IYkSession.hpp"

#include <algorithm>

#include <cuda_runtime.h>

#include "FDK/YkFdkPipeline.hpp"
#include "Iter/YkAlgebraicLegacyAdapters.hpp"
#include "Iter/YkAlgebraicSartSirtAdapters.hpp"
#include "Iter/YkCglsLegacyAdapters.hpp"
#include "common/YkExecutionContext.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"

namespace YK {
namespace {

SFilterKernelDesc makeFilterDesc(const SFdkAlgoParams& ap)
{
    SFilterKernelDesc d{};
    d.kind = static_cast<EFilterKernel>(ap.filter);
    d.cutoff = ap.cutoff;
    d.gain = ap.gain;
    d.order = ap.butterworth_order;
    d.beta = ap.kaiser_beta;
    d.tukey_alpha = ap.tukey_alpha;
    return d;
}

class Session final : public ISession {
public:
    ~Session() override { release(); }

    bool initialize(const SessionDesc& desc) override
    {
        release();
        if (desc.gpu.empty() || desc.scan.Nu <= 0 || desc.scan.Nv <= 0 ||
            desc.scan.NAng <= 0 || desc.volume.Nx <= 0 || desc.volume.Ny <= 0 ||
            desc.volume.Nz <= 0) {
            YK_LOGE("[Session] initialize: invalid geometry or GPU selection.");
            return false;
        }

        desc_ = desc;
        device_ = desc.gpu[0];
        if (!geometry_.initialize(desc) || !resources_.initialize(device_)) return false;
        params_ = geometry_.base();
        params_.angle_list = geometry_.allAngles();
        if (geometry_.hasExternalGeometry()) {
            // 外部 geometry 的 angle.x 是唯一角度来源。同步写入内部参数仅为
            // 复用当前 FDK/Parker 配置接口，绝不读取 SessionDesc::angles。
            params_.angle_list.resize(geometry_.allGeometry().size());
            for (size_t i = 0; i < geometry_.allGeometry().size(); ++i)
                params_.angle_list[i] = geometry_.allGeometry()[i].angle.x;
            params_.scan_start_angle_rad = params_.angle_list.front();
            if (params_.angle_list.size() >= 2)
                params_.nDirSign = params_.angle_list[1] >= params_.angle_list[0] ? 1 : -1;
        }
        params_.iPAng = static_cast<int>(params_.angle_list.size());

        bool ok = false;
        switch (desc.algorithm.pipeline) {
        case EPipeline::FDK:
            params_.desc = makeFilterDesc(desc.algorithm.fdk);
            // geometry 非空时，它是唯一的几何/角度真源。圆轨迹 angles 只在
            // geometry 为空时用于构造同一份完整 geometry。
            ok = geometry_.hasExternalGeometry()
                ? fdk_.prepareWithGeometry(params_, geometry_.allGeometry(), kMaxChunkAng,
                    resources_.stream(), device_)
                : static_cast<int>(params_.angle_list.size()) == params_.iPAngTotal
                ? fdk_.prepareWithAngles(params_, params_.angle_list, kMaxChunkAng,
                    resources_.stream(), device_)
                : fdk_.prepare(params_, kMaxChunkAng, resources_.stream(), device_);
            break;
        case EPipeline::ForwardProjection:
            forward_ = makeForwardOperator(desc.algorithm.forward_projector);
            ok = forward_->prepare(geometry_, resources_);
            break;
        case EPipeline::SIRT: {
            if (!hasCompleteAngles_()) break;
            SIRT::Config c{};
            c.n_iter = desc.algorithm.iterative.iterations;
            c.lambda = desc.algorithm.iterative.relaxation;
            c.fp_task = desc.algorithm.forward_projector;
            c.bp_task = desc.algorithm.back_projector;
            ok = sirt_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        case EPipeline::OSSART: {
            if (!hasCompleteAngles_()) break;
            OSSART::Config c{};
            c.n_iter = desc.algorithm.iterative.iterations;
            c.n_subset = std::max(1, desc.algorithm.iterative.subsets);
            c.lambda = desc.algorithm.iterative.relaxation;
            c.fp_task = desc.algorithm.forward_projector;
            c.bp_task = desc.algorithm.back_projector;
            ok = ossart_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        case EPipeline::CGLS: {
            if (!hasCompleteAngles_()) break;
            CGLS::Config c{};
            c.n_iter = desc.algorithm.iterative.iterations;
            c.fp_task = desc.algorithm.forward_projector;
            c.bp_task = desc.algorithm.back_projector;
            ok = cgls_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        }

        if (!ok) {
            release();
            return false;
        }
        initialized_ = true;
        return true;
    }

    bool execute(const ExecuteRequest& r) override
    {
        if (!initialized_) {
            YK_LOGE("[Session] execute: session is not initialized.");
            return false;
        }
        switch (desc_.algorithm.pipeline) {
        case EPipeline::FDK: return executeFdk_(r);
        case EPipeline::ForwardProjection: return executeFp_(r);
        case EPipeline::SIRT:
        case EPipeline::OSSART:
        case EPipeline::CGLS: return executeIterative_(r);
        }
        return false;
    }

    void reset() override
    {
        fdk_.reset(); sirt_.reset(); ossart_.reset();
    }

    void release() override
    {
        fdk_.release();
        if (forward_) forward_->release();
        if (backward_) backward_->release();
        forward_.reset(); backward_.reset();
        sirt_.release(); ossart_.release(); cgls_.release();
        freeScratch_();
        resources_.release();
        params_ = {};
        desc_ = {};
        initialized_ = false;
    }

    bool isInitialized() const override { return initialized_; }

private:
    bool hasCompleteAngles_() const
    {
        if (static_cast<int>(params_.angle_list.size()) != desc_.scan.NAng) {
            YK_LOGE("[Session] iterative pipelines require complete angles or per-view geometry.");
            return false;
        }
        return true;
    }

    bool executeFdk_(const ExecuteRequest& r)
    {
        if (r.K <= 0 || !r.projection.data || !r.volume.data) return false;
        if (!geometry_.hasExternalGeometry() && !r.angles) {
            YK_LOGE("[Session] FDK 圆轨迹模式要求 ExecuteRequest::angles。");
            return false;
        }
        // FDK pipeline 会按 chunk 异步完成主机到设备的投影传输。这里不接受
        // device projection，避免接口暗中引入一次同步的 device-to-host 回传。
        if (r.projection.location != EMemoryLocation::Host) {
            YK_LOGE("[Session] FDK projection input must be host memory.");
            return false;
        }
        float* d_out = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : ensureVolumeScratch_();
        if (!d_out) return false;
        // 外部 geometry 模式在 initialize 时已预计算全序列；执行时仅提交
        // 投影。圆轨迹模式的 angles 只用于 fallback geometry 构造。
        const FdkProjectionBatch batch{ r.projection.data, nullptr,
            geometry_.hasExternalGeometry() ? nullptr : r.angles, r.K };
        // 当前 DLL ExecuteRequest 没有向调用方暴露 CUDA event/fence，因此
        // execute() 必须采用同步批次语义：返回后 projection 主机缓冲即可
        // 释放或写入下一批。需要 CPU/GPU 重叠的内部调用应直接使用
        // FdkPipeline::enqueueBatch() 和双 pinned buffer。
        if (!fdk_.processBatchSync(batch, d_out, r.clear_output)) return false;
        // A host output only becomes a complete reconstruction after the
        // final FDK batch.  Copying earlier would expose a partial volume and
        // adds an unnecessary device synchronization for every batch.
        if (r.volume.location == EMemoryLocation::Host &&
            fdk_.complete()) {
            YK_CUDA_CHECK(cudaMemcpyAsync(r.volume.data, d_out, volumeBytes_(),
                cudaMemcpyDeviceToHost, resources_.stream()));
            YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        }
        return true;
    }

    bool executeFp_(const ExecuteRequest& r)
    {
        if (r.K <= 0 || !r.projection.data || !r.volume.data) return false;
        // 外部 geometry 模式下 angle.x 已是唯一角度来源。当前 ExecuteRequest
        // 没有 batch offset，因此只允许一次提交完整序列，避免用另一份 angles
        // 去猜测子集并造成几何/角度分叉。
        if (geometry_.hasExternalGeometry() && r.K != params_.iPAngTotal) {
            YK_LOGE("[Session] external-geometry FP currently requires the complete view sequence.");
            return false;
        }
        if (!geometry_.hasExternalGeometry() && !r.angles) {
            YK_LOGE("[Session] circular FP requires ExecuteRequest::angles.");
            return false;
        }
        const float* d_volume = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : uploadVolume_(r.volume.data);
        float* d_projection = r.projection.location == EMemoryLocation::Device
            ? r.projection.data : ensureProjectionScratch_(r.K);
        if (!d_volume || !d_projection) return false;
        const float* batch_angles = geometry_.hasExternalGeometry()
            ? geometry_.allAngles().data() : r.angles;
        SCBCTParams batch = geometry_.batch(batch_angles, r.K);
        batch.iPAngTotal = r.K;
        if (!forward_ || !forward_->apply(d_volume, batch, d_projection, resources_)) return false;
        if (r.projection.location == EMemoryLocation::Host) {
            const size_t bytes = static_cast<size_t>(r.K) * params_.iPU * params_.iPV * sizeof(float);
            YK_CUDA_CHECK(cudaMemcpyAsync(r.projection.data, d_projection, bytes,
                cudaMemcpyDeviceToHost, resources_.stream()));
            YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        }
        else {
            // 公共 Session 当前没有完成事件可返回。即使输入和输出都位于
            // device，execute() 也必须在正投 kernel 完成后才能返回，避免调用方
            // 立即复用 volume 或读取 projection 时与本次计算竞争。
            YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        }
        return true;
    }

    bool executeIterative_(const ExecuteRequest& r)
    {
        if (!r.projection.data || !r.volume.data || r.projection.location != EMemoryLocation::Device ||
            r.volume.location != EMemoryLocation::Device) {
            YK_LOGE("[Session] iterative pipelines currently require device projection and volume buffers.");
            return false;
        }
        bool ok = false;
        switch (desc_.algorithm.pipeline) {
        case EPipeline::SIRT:
            ok = r.iteration_count > 0
                ? sirt_.iterate(r.projection.data, r.volume.data, params_, resources_.stream(), r.iteration_count)
                : sirt_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::OSSART:
            ok = r.iteration_count > 0
                ? ossart_.iterate(r.projection.data, r.volume.data, params_, resources_.stream(), r.iteration_count)
                : ossart_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::CGLS:
            if (r.iteration_count > 0)
                YK_LOGW("[Session] CGLS iteration_count is fixed at initialize time.");
            ok = cgls_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        default: return false;
        }
        if (!ok) return false;
        // 与 FDK/FP 保持同一公共契约：execute() 返回即表示本次请求完成。
        // 底层重建器仍然使用异步 stream，算法内部不会因此逐 kernel 同步。
        YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        return true;
    }

    float* ensureVolumeScratch_()
    {
        if (!d_volume_scratch_)
            YK_CUDA_CHECK(cudaMalloc(&d_volume_scratch_, volumeBytes_()));
        return d_volume_scratch_;
    }
    float* uploadVolume_(const float* host)
    {
        float* d = ensureVolumeScratch_();
        if (!d) return nullptr;
        YK_CUDA_CHECK(cudaMemcpyAsync(d, host, volumeBytes_(), cudaMemcpyHostToDevice, resources_.stream()));
        return d;
    }
    float* ensureProjectionScratch_(int k)
    {
        const size_t needed = static_cast<size_t>(k) * params_.iPU * params_.iPV * sizeof(float);
        if (needed > projection_scratch_bytes_) {
            if (d_projection_scratch_) cudaFree(d_projection_scratch_);
            YK_CUDA_CHECK(cudaMalloc(&d_projection_scratch_, needed));
            projection_scratch_bytes_ = needed;
        }
        return d_projection_scratch_;
    }
    size_t volumeBytes_() const { return static_cast<size_t>(params_.iVX) * params_.iVY * params_.iVZ * sizeof(float); }
    void freeScratch_()
    {
        if (d_volume_scratch_) { cudaFree(d_volume_scratch_); d_volume_scratch_ = nullptr; }
        if (d_projection_scratch_) { cudaFree(d_projection_scratch_); d_projection_scratch_ = nullptr; }
        projection_scratch_bytes_ = 0;
    }

    SessionDesc desc_{};
    SCBCTParams params_{};
    int device_ = 0;
    bool initialized_ = false;
    FdkPipeline fdk_;
    GeometryContext geometry_;
    ResourceContext resources_;
    std::unique_ptr<IForwardOperator> forward_;
    std::unique_ptr<IBackOperator> backward_;
    SIRT sirt_;
    OSSART ossart_;
    CGLS cgls_;
    float* d_volume_scratch_ = nullptr;
    float* d_projection_scratch_ = nullptr;
    size_t projection_scratch_bytes_ = 0;
};

} // namespace

ISession* SessionFactory::create() { return new Session(); }
void SessionFactory::destroy(ISession* session) { delete session; }

} // namespace YK
