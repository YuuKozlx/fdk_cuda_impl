#include "YKCBCT/interface/IYkSession.hpp"

#include <algorithm>

#include <cuda_runtime.h>

#include "FDK/YkFdkReconstructor.hpp"
#include "FP/YkFPRunner.hpp"
#include "Iter/YkSART.hpp"
#include "Iter/YkOSSART.hpp"
#include "Iter/YkCGLS.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"

namespace YK {
namespace {

SCBCTParams makeParams(const SessionDesc& desc)
{
    SCBCTParams p{};
    p.iPU = desc.scan.Nu;
    p.iPV = desc.scan.Nv;
    p.iPAngTotal = desc.scan.NAng;
    p.du_mm = desc.scan.du_mm;
    p.dv_mm = desc.scan.dv_mm;
    p.offsetU_mm = desc.scan.offsetU_mm;
    p.offsetV_mm = desc.scan.offsetV_mm;
    p.tiltn_angle_rad = desc.scan.tiltN_rad;
    p.tiltu_angle_rad = desc.scan.tiltU_rad;
    p.tiltv_angle_rad = desc.scan.tiltV_rad;
    p.SID = desc.scan.SOD_mm;
    p.SDD = desc.scan.SDD_mm;
    p.scan_range_rad = desc.scan.scanRangeRad;
    p.scan_start_angle_rad = desc.scan.startAngleRad;
    p.bShortScan = desc.scan.shortScan;
    p.nDirSign = desc.scan.nDirSign;
    p.iVX = desc.volume.Nx;
    p.iVY = desc.volume.Ny;
    p.iVZ = desc.volume.Nz;
    p.vox_x_mm = desc.volume.voxX_mm;
    p.vox_y_mm = desc.volume.voxY_mm;
    p.vox_z_mm = desc.volume.voxZ_mm;
    p.vol_offset_x_mm = desc.volume.offsetX_mm;
    p.vol_offset_y_mm = desc.volume.offsetY_mm;
    p.vol_offset_z_mm = desc.volume.offsetZ_mm;
    p.angle_list = desc.angles;
    if (!p.angle_list.empty())
        p.iPAng = static_cast<int>(p.angle_list.size());
    return p;
}

SFilterKernelDesc makeFilterDesc(const SFdkAlgoParams& ap)
{
    SFilterKernelDesc d{};
    d.kind = static_cast<EFilterKernel>(ap.filter);
    d.cutoff = ap.cutoff;
    d.gain = ap.gain;
    d.force_dc_zero = ap.force_dc_zero;
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
        params_ = makeParams(desc);
        YK_CUDA_CHECK(cudaSetDevice(device_));
        YK_CUDA_CHECK(cudaStreamCreate(&stream_));

        bool ok = false;
        switch (desc.algorithm.pipeline) {
        case EPipeline::FDK:
            params_.desc = makeFilterDesc(desc.algorithm.fdk);
            ok = fdk_.init(params_, kMaxChunkAng, stream_, device_);
            break;
        case EPipeline::ForwardProjection:
            ok = fp_.init(params_, desc.algorithm.forward_projector, device_);
            break;
        case EPipeline::SIRT: {
            if (!hasCompleteAngles_()) break;
            SIRT::Config c{};
            c.n_iter = desc.algorithm.iterative.iterations;
            c.lambda = desc.algorithm.iterative.relaxation;
            c.fp_task = desc.algorithm.forward_projector;
            c.bp_task = desc.algorithm.back_projector;
            ok = sirt_.init(params_, c, stream_, device_);
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
            ok = ossart_.init(params_, c, stream_, device_);
            break;
        }
        case EPipeline::CGLS: {
            if (!hasCompleteAngles_()) break;
            CGLS::Config c{};
            c.n_iter = desc.algorithm.iterative.iterations;
            c.fp_task = desc.algorithm.forward_projector;
            c.bp_task = desc.algorithm.back_projector;
            ok = cgls_.init(params_, c, stream_, device_);
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
        fdk_.reset(); fp_.reset(); sirt_.reset(); ossart_.reset();
    }

    void release() override
    {
        fdk_.release(); fp_.release(); sirt_.release(); ossart_.release(); cgls_.release();
        freeScratch_();
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        params_ = {};
        desc_ = {};
        initialized_ = false;
    }

    bool isInitialized() const override { return initialized_; }

private:
    bool hasCompleteAngles_() const
    {
        if (static_cast<int>(desc_.angles.size()) != desc_.scan.NAng) {
            YK_LOGE("[Session] iterative pipelines require SessionDesc::angles for every view.");
            return false;
        }
        return true;
    }

    bool executeFdk_(const ExecuteRequest& r)
    {
        if (!r.angles || r.K <= 0 || !r.projection.data || !r.volume.data) return false;
        // FdkReconstructor performs asynchronous host-to-device staging per
        // chunk.  Device projections intentionally are not accepted here;
        // accepting them would hide a synchronous device->host round trip.
        if (r.projection.location != EMemoryLocation::Host) {
            YK_LOGE("[Session] FDK projection input must be host memory.");
            return false;
        }
        SCBCTParams batch = params_;
        batch.iPAng = r.K;
        batch.angle_list.assign(r.angles, r.angles + r.K);
        float* d_out = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : ensureVolumeScratch_();
        if (!d_out) return false;
        if (!fdk_.feed(r.projection.data, batch, stream_, d_out, r.clear_output)) return false;
        // A host output only becomes a complete reconstruction after the
        // final FDK batch.  Copying earlier would expose a partial volume and
        // adds an unnecessary device synchronization for every batch.
        if (r.volume.location == EMemoryLocation::Host &&
            fdk_.totalReceived() >= params_.iPAngTotal) {
            YK_CUDA_CHECK(cudaMemcpyAsync(r.volume.data, d_out, volumeBytes_(),
                cudaMemcpyDeviceToHost, stream_));
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
        return true;
    }

    bool executeFp_(const ExecuteRequest& r)
    {
        if (!r.angles || r.K <= 0 || !r.projection.data || !r.volume.data) return false;
        const float* d_volume = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : uploadVolume_(r.volume.data);
        float* d_projection = r.projection.location == EMemoryLocation::Device
            ? r.projection.data : ensureProjectionScratch_(r.K);
        if (!d_volume || !d_projection) return false;
        SCBCTParams batch = params_;
        batch.iPAng = r.K;
        batch.iPAngTotal = r.K;
        batch.angle_list.assign(r.angles, r.angles + r.K);
        if (!fp_.run(d_volume, batch, d_projection, stream_, device_)) return false;
        if (r.projection.location == EMemoryLocation::Host) {
            const size_t bytes = static_cast<size_t>(r.K) * params_.iPU * params_.iPV * sizeof(float);
            YK_CUDA_CHECK(cudaMemcpyAsync(r.projection.data, d_projection, bytes,
                cudaMemcpyDeviceToHost, stream_));
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
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
        switch (desc_.algorithm.pipeline) {
        case EPipeline::SIRT:
            return r.iteration_count > 0
                ? sirt_.iterate(r.projection.data, r.volume.data, params_, stream_, r.iteration_count)
                : sirt_.run(r.projection.data, r.volume.data, params_, stream_);
        case EPipeline::OSSART:
            return r.iteration_count > 0
                ? ossart_.iterate(r.projection.data, r.volume.data, params_, stream_, r.iteration_count)
                : ossart_.run(r.projection.data, r.volume.data, params_, stream_);
        case EPipeline::CGLS:
            if (r.iteration_count > 0)
                YK_LOGW("[Session] CGLS iteration_count is fixed at initialize time.");
            return cgls_.run(r.projection.data, r.volume.data, params_, stream_);
        default: return false;
        }
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
        YK_CUDA_CHECK(cudaMemcpyAsync(d, host, volumeBytes_(), cudaMemcpyHostToDevice, stream_));
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
    cudaStream_t stream_ = nullptr;
    int device_ = 0;
    bool initialized_ = false;
    FdkReconstructor fdk_;
    ConeProjector fp_;
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
