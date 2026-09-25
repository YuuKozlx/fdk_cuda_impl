#include "interface/YkExecutionBackend.hpp"
#include "common/YkExecutionContext.hpp"

#include <algorithm>

#include <cuda_runtime.h>

#include "FDK/YkFdkPipeline.hpp"
#include "FDK/XFDK/YkXfdkPipeline.hpp"
#include "FDK/CFDK/YkCurveFilteredFdkPipeline.hpp"
#include "CylFpBp/analytic/YkCylAnalyticReconstruction.hpp"
#include "CylFpBp/fp/YkCylForwardProjection.hpp"
#include "CylFpBp/iter/YkCylAlgebraicReconstructor.hpp"
#include "CylFpBp/iter/YkCylPwlsReconstructor.hpp"
#include "Iter/YkAlgebraicLegacyAdapters.hpp"
#include "Iter/YkAlgebraicSartSirtAdapters.hpp"
#include "Iter/YkCglsLegacyAdapters.hpp"
#include "Iter/YkPwlsReconstructor.hpp"
#include "Iter/YkTigreGradientReconstructorEx.hpp"
#include "common/YkExecutionContext.hpp"
#include "common/YkProjectionOperators.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"

#if YKCBCT_HAS_HELICAL
#include "Heli/analytic/wfbp/YkWfbpPipeline.hpp"
#endif

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

CylFpBp::EIterativeForwardModel makeCylForwardModel(ETask task)
{
    return task == ETask::FP_Siddon
        ? CylFpBp::EIterativeForwardModel::Siddon
        : CylFpBp::EIterativeForwardModel::Joseph;
}

CylFpBp::EIterativeBackprojectorModel makeCylBackModel(ETask task)
{
    using Model = CylFpBp::EIterativeBackprojectorModel;
    switch (task) {
    case ETask::BP_Siddon_RayDriven: return Model::SiddonRayDriven;
    case ETask::BP_Siddon_VoxDriven: return Model::Siddon;
    case ETask::BP_Siddon_VoxDriven_v2: return Model::SiddonV2;
    case ETask::BP_Siddon_VoxDriven_v3: return Model::SiddonV3;
    case ETask::BP_Joseph: return Model::Joseph;
    case ETask::BP_Joseph_v3: return Model::JosephV3;
    case ETask::BP_Joseph_v2: return Model::JosephV3; // 新 DLL 边界已拒绝；仅供旧内部请求过渡。
    case ETask::BP_FDK: return Model::Fdk;
    case ETask::BP_FDK_matched: return Model::FdkMatched;
    default: return Model::JosephV3;
    }
}

class Session final : public detail::IExecutionBackend {
public:
    ~Session() override { release(); }

    bool initialize(const SSystemConfig& system,
        const SReconstructionSpec& reconstruction, int device) override
    {
        release();
        PreparedGeometry prepared;
        if (!prepared.initialize(system) || device < 0) {
            YK_LOGE("[Session] initialize: invalid geometry or GPU selection.");
            return false;
        }
        reconstruction_ = reconstruction;
        device_ = device;
        geometry_ = std::move(prepared);
        if (!resources_.initialize(device_)) return false;
        params_ = geometry_.base();
        params_.scan.short_scan = !geometry_.isCylindrical() &&
            system.trajectory == ETrajectoryKind::Circular &&
            resolveParkerEnabled(system, reconstruction_);
        params_.scan.angles = geometry_.allAngles();
        if (geometry_.hasExternalGeometry()) {
            // 外部 geometry 的 angle.x 是唯一角度来源。同步写入内部参数仅为
            // 复用当前 FDK/Parker 配置接口，绝不读取执行请求中的第二份角度。
            params_.scan.angles.resize(geometry_.allGeometry().size());
            for (size_t i = 0; i < geometry_.allGeometry().size(); ++i)
                params_.scan.angles[i] = geometry_.allGeometry()[i].angle.x;
            params_.scan.start_angle_rad = params_.scan.angles.front();
            if (params_.scan.angles.size() >= 2)
                params_.scan.direction = params_.scan.angles[1] >= params_.scan.angles[0] ? 1 : -1;
        }
        params_.scan.NAng = static_cast<int>(params_.scan.angles.size());
        cylindrical_ = geometry_.isCylindrical();

        bool ok = false;
        switch (reconstruction_.pipeline) {
        case EPipeline::FDK:
            if (cylindrical_) {
                if (geometry_.cylGeometry().empty()) break;
                CylFpBp::Analytic::ReconstructionConfig config{};
                config.source_to_detector_mm = params_.scan.sdd_mm;
                ok = cyl_fdk_.prepare(geometry_.volumeGeometry(), params_.scan.Nu,
                    params_.scan.Nv, geometry_.cylGeometry(), config,
                    makeFilterDesc(reconstruction_.fdk), resources_.stream(), device_);
                break;
            }
            params_.reconstruction.filter = makeFilterDesc(reconstruction_.fdk);
            // geometry 非空时，它是唯一的几何/角度真源。圆轨迹 angles 只在
            // geometry 为空时用于构造同一份完整 geometry。
            ok = geometry_.hasExternalGeometry()
                ? fdk_.prepareWithGeometry(params_, geometry_.allGeometry(), kMaxChunkAng,
                    resources_.stream(), device_)
                : static_cast<int>(params_.scan.angles.size()) == params_.scan.totalViews
                ? fdk_.prepareWithAngles(params_, params_.scan.angles, kMaxChunkAng,
                    resources_.stream(), device_)
                : fdk_.prepare(params_, kMaxChunkAng, resources_.stream(), device_);
            break;
        case EPipeline::XFDK:
            if (!geometry_.hasExternalGeometry()) break;
            params_.reconstruction.filter = makeFilterDesc(reconstruction_.fdk);
            ok = xfdk_.prepare(params_, geometry_.allGeometry(), resources_.stream(), device_);
            break;
        case EPipeline::CFDK:
            if (!geometry_.hasExternalGeometry()) break;
            params_.reconstruction.filter = makeFilterDesc(reconstruction_.fdk);
            ok = cfdk_.prepare(params_, geometry_.allGeometry(), resources_.stream(), device_);
            break;
        case EPipeline::PWLS: {
            if (!hasCompleteAngles_()) break;
            if (cylindrical_) {
                CylFpBp::CylPwlsConfig c{};
                c.iterations = reconstruction_.iterative.iterations;
                c.relaxation = reconstruction_.iterative.relaxation;
                c.regularizer = static_cast<Iter::EPwlsRegularizer>(
                    reconstruction_.pwls.regularizer);
                c.regularization = reconstruction_.pwls.regularization;
                c.huber_delta = reconstruction_.pwls.huber_delta;
                c.epsilon = reconstruction_.pwls.epsilon;
                c.lower_bound = reconstruction_.pwls.lower_bound;
                c.upper_bound = reconstruction_.pwls.upper_bound;
                c.data_model = forwardTask_() == ETask::FP_Siddon
                    ? CylFpBp::ECylPwlsDataModel::Siddon
                    : CylFpBp::ECylPwlsDataModel::JosephMatched;
                ok = cyl_pwls_.prepare(geometry_.volumeGeometry(), params_.scan.Nu,
                    params_.scan.Nv, geometry_.cylGeometry(), c, {}, resources_.stream(),
                    device_);
                break;
            }
            if (!geometry_.hasExternalGeometry()) break;
            Iter::PwlsConfig c{};
            c.iterations = reconstruction_.iterative.iterations;
            c.subset_count = std::max(1, reconstruction_.iterative.subsets);
            c.relaxation = reconstruction_.iterative.relaxation;
            c.regularizer = static_cast<Iter::EPwlsRegularizer>(reconstruction_.pwls.regularizer);
            c.regularization = reconstruction_.pwls.regularization;
            c.huber_delta = reconstruction_.pwls.huber_delta;
            c.epsilon = reconstruction_.pwls.epsilon;
            c.lower_bound = reconstruction_.pwls.lower_bound;
            c.upper_bound = reconstruction_.pwls.upper_bound;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
            ok = pwls_.prepare(params_, geometry_.allGeometry(), c,
                resources_.stream(), device_);
            break;
        }
        case EPipeline::TigreGradient: {
            if (!hasCompleteAngles_() || !geometry_.hasExternalGeometry()) break;
            Iter::TigreGradientConfig c{};
            c.algorithm = static_cast<Iter::ETigreGradientAlgorithm>(reconstruction_.tigre.method);
            c.iterations = reconstruction_.iterative.iterations;
            // 配置层统一把 iterations 定义为完整数据轮数、subsets 定义为
            // 每轮子集数量。TIGRE 后端内部需要每块视角数，因此按算法换算。
            if (c.algorithm == Iter::ETigreGradientAlgorithm::Sart) {
                c.block_size = 1;
            }
            else if (c.algorithm == Iter::ETigreGradientAlgorithm::Sirt) {
                c.block_size = params_.scan.NAng;
            }
            else {
                const int subset_count = std::max(1, reconstruction_.iterative.subsets);
                c.block_size = std::max(1,
                    (params_.scan.NAng + subset_count - 1) / subset_count);
            }
            c.lambda = reconstruction_.iterative.relaxation;
            c.lambda_reduction = reconstruction_.tigre.lambda_reduction;
            c.relaxation_mode = reconstruction_.tigre.nesterov_relaxation
                ? Iter::ETigreRelaxationMode::Nesterov : Iter::ETigreRelaxationMode::Scalar;
            c.initialization = reconstruction_.tigre.fdk_initialization
                ? Iter::ETigreInitialization::Fdk : Iter::ETigreInitialization::Zero;
            c.non_negative = reconstruction_.tigre.non_negative;
            c.tv_iterations = reconstruction_.tigre.tv_iterations;
            c.alpha = reconstruction_.tigre.tv_alpha;
            c.alpha_reduction = reconstruction_.tigre.tv_alpha_reduction;
            c.maximum_update_ratio = reconstruction_.tigre.maximum_update_ratio;
            c.max_l2_error = reconstruction_.tigre.max_l2_error;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
            ok = tigre_.prepare(params_, geometry_.allGeometry(), c,
                resources_.stream(), device_);
            break;
        }
        case EPipeline::WFBP:
#if YKCBCT_HAS_HELICAL
        {
            if (!cylindrical_ || geometry_.cylGeometry().empty()) break;
            Helical::Wfbp::Config c{};
            c.input_detector = Helical::Wfbp::EInputDetector::CylindricalArc;
            c.redundancy_flat = reconstruction_.wfbp.redundancy_flat;
            c.filter.cutoff_c = reconstruction_.wfbp.filter_cutoff;
            c.filter.apodization_a = reconstruction_.wfbp.filter_apodization;
            ok = wfbp_.prepare(geometry_.volumeGeometry(), params_.scan.Nu,
                params_.scan.Nv, geometry_.cylGeometry(), c, resources_.stream(), device_);
            break;
        }
#else
            YK_LOGE("[Session] 当前构建未启用 YKCBCT_BUILD_HELICAL，无法使用 wFBP。");
            break;
#endif
        case EPipeline::ForwardProjection:
            if (cylindrical_) {
                const auto model = forwardTask_() == ETask::FP_Siddon
                    ? CylFpBp::EForwardProjection::Siddon
                    : CylFpBp::EForwardProjection::Joseph;
                cyl_forward_ = CylFpBp::makeForwardProjection(model);
                auto fp_geometry = geometry_.cylGeometry();
                const auto& pose = geometry_.forwardProjectionPose();
                if (pose.enabled) {
                    const auto volume = geometry_.volumeGeometry();
                    const auto local_from_world = SRigidTransform::aroundEulerPoint(
                        volume.center, pose.rotation_x_rad, pose.rotation_y_rad,
                        pose.rotation_z_rad).inverse();
                    for (auto& view : fp_geometry)
                        view = transformProjectionGeometry(view, local_from_world);
                }
                ok = cyl_forward_->prepare(geometry_.volumeGeometry(),
                    params_.scan.Nu, params_.scan.Nv, fp_geometry, {}, resources_);
            } else {
                forward_ = makeForwardOperator(forwardTask_());
                ok = forward_->prepare(geometry_, resources_);
            }
            break;
        case EPipeline::SART: {
            if (!hasCompleteAngles_()) break;
            SART::Config c{};
            c.n_iter = reconstruction_.iterative.iterations;
            c.lambda = reconstruction_.iterative.relaxation;
            c.use_min = reconstruction_.tigre.non_negative;
            c.min_constraint = 0.f;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
            ok = sart_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        case EPipeline::SIRT: {
            if (!hasCompleteAngles_()) break;
            if (cylindrical_) {
                ok = prepareCylIterative_(CylFpBp::EIterativeMethod::Sirt);
                break;
            }
            SIRT::Config c{};
            c.n_iter = reconstruction_.iterative.iterations;
            c.lambda = reconstruction_.iterative.relaxation;
            c.use_min = reconstruction_.tigre.non_negative;
            c.min_constraint = 0.f;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
            ok = sirt_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        case EPipeline::OSSART: {
            if (!hasCompleteAngles_()) break;
            if (cylindrical_) {
                ok = prepareCylIterative_(CylFpBp::EIterativeMethod::Ossart);
                break;
            }
            OSSART::Config c{};
            c.n_iter = reconstruction_.iterative.iterations;
            c.n_subset = std::max(1, reconstruction_.iterative.subsets);
            c.lambda = reconstruction_.iterative.relaxation;
            c.use_min = reconstruction_.tigre.non_negative;
            c.min_constraint = 0.f;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
            ok = ossart_.init(params_, c, geometry_.allGeometry(),
                resources_.stream(), device_);
            break;
        }
        case EPipeline::CGLS: {
            if (!hasCompleteAngles_()) break;
            if (cylindrical_) {
                ok = prepareCylIterative_(CylFpBp::EIterativeMethod::Cgls);
                break;
            }
            CGLS::Config c{};
            c.n_iter = reconstruction_.iterative.iterations;
            c.fp_task = forwardTask_();
            c.bp_task = backTask_();
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
        switch (reconstruction_.pipeline) {
        case EPipeline::FDK: return cylindrical_ ? executeCylFdk_(r) : executeFdk_(r);
        case EPipeline::XFDK: return executeXfdk_(r);
        case EPipeline::CFDK: return executeCfdk_(r);
        case EPipeline::ForwardProjection: return executeFp_(r);
        case EPipeline::SART:
        case EPipeline::SIRT:
        case EPipeline::OSSART:
        case EPipeline::CGLS: return executeIterative_(r);
        case EPipeline::PWLS:
        case EPipeline::TigreGradient: return executeIterative_(r);
        case EPipeline::WFBP: return executeWfbp_(r);
        }
        return false;
    }

    void reset() override
    {
        // reset 只清除一次执行产生的瞬态状态，保留 initialize() 已准备的
        // 几何、纹理和工作区；release() 才负责销毁完整后端资源。
        fdk_.reset();
        sart_.reset();
        sirt_.reset();
        ossart_.reset();
        cgls_.reset();
        pwls_.reset();
        tigre_.reset();
        cyl_iterative_.reset();
        cyl_pwls_.reset();
    }

    void release() override
    {
        fdk_.release();
        xfdk_.release();
        cfdk_.release();
        cyl_fdk_.release();
        if (forward_) forward_->release();
        if (backward_) backward_->release();
        forward_.reset(); backward_.reset();
        if (cyl_forward_) cyl_forward_->release();
        cyl_forward_.reset();
        cyl_iterative_.release(); cyl_pwls_.release();
#if YKCBCT_HAS_HELICAL
        wfbp_.release();
#endif
        sart_.release(); sirt_.release(); ossart_.release(); cgls_.release();
        pwls_.release(); tigre_.release();
        freeScratch_();
        resources_.release();
        params_ = {};
        reconstruction_ = {};
        initialized_ = false;
        cylindrical_ = false;
    }

    bool isInitialized() const override { return initialized_; }

private:
    ETask forwardTask_() const
    {
        return reconstruction_.projection_model == EProjectionModel::Siddon
            ? ETask::FP_Siddon : ETask::FP_Joseph;
    }
    ETask backTask_() const
    {
        return reconstruction_.projection_model == EProjectionModel::Siddon
            ? ETask::BP_Siddon_RayDriven : ETask::BP_Joseph_v3;
    }

    bool prepareCylIterative_(CylFpBp::EIterativeMethod method)
    {
        CylFpBp::IterativeConfig c{};
        c.method = method;
        c.iterations = reconstruction_.iterative.iterations;
        c.subset_count = std::max(1, reconstruction_.iterative.subsets);
        c.relaxation = reconstruction_.iterative.relaxation;
        c.nonnegative = reconstruction_.tigre.non_negative;
        c.forward_model = makeCylForwardModel(forwardTask_());
        c.backprojector_model = makeCylBackModel(backTask_());
        c.convergence.relative_residual_tolerance =
            reconstruction_.cgls.relative_residual;
        c.convergence.minimum_iterations =
            reconstruction_.cgls.minimum_iterations;
        c.convergence.check_interval =
            reconstruction_.cgls.check_interval;
        c.convergence.patience = reconstruction_.cgls.patience;
        return cyl_iterative_.prepare(geometry_.volumeGeometry(), params_.scan.Nu,
            params_.scan.Nv, geometry_.cylGeometry(), c, {}, resources_.stream(), device_);
    }

    bool hasCompleteAngles_() const
    {
        if (static_cast<int>(params_.scan.angles.size()) != params_.scan.NAng) {
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

    // xFDK/C-FDK 的重排跨多个视图取样，不能复用普通 FDK 的分批 Execute
    // 契约。公共 Session 因而只接受完整的 device 投影序列，并在返回前等待
    // 本次重建完成，避免调用者读到仍在 stream 中累加的体数据。
    bool executeXfdk_(const ExecuteRequest& r)
    {
        if (!hasFullDeviceReconstructionInput_(r, "XFDK")) return false;
        if (!xfdk_.reconstruct(r.projection.data, r.volume.data, r.clear_output)) return false;
        YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        return true;
    }

    bool executeCfdk_(const ExecuteRequest& r)
    {
        if (!hasFullDeviceReconstructionInput_(r, "CFDK")) return false;
        if (!cfdk_.reconstruct(r.projection.data, r.volume.data, r.clear_output)) return false;
        YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        return true;
    }

    bool executeCylFdk_(const ExecuteRequest& r)
    {
        if (r.K != params_.scan.NAng || !r.projection.data || !r.volume.data)
            return false;
        const size_t bytes = static_cast<size_t>(r.K) * params_.scan.Nu *
            params_.scan.Nv * sizeof(float);
        float* projection = r.projection.location == EMemoryLocation::Device
            ? r.projection.data : ensureProjectionScratch_(r.K);
        float* volume = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : ensureVolumeScratch_();
        if (!projection || !volume) return false;
        if (r.projection.location == EMemoryLocation::Host)
            YK_CUDA_CHECK(cudaMemcpyAsync(projection, r.projection.data, bytes,
                cudaMemcpyHostToDevice, resources_.stream()));
        if (r.volume.location == EMemoryLocation::Host && !r.clear_output)
            YK_CUDA_CHECK(cudaMemcpyAsync(volume, r.volume.data, volumeBytes_(),
                cudaMemcpyHostToDevice, resources_.stream()));
        if (!cyl_fdk_.reconstruct(projection, volume, r.clear_output)) {
            cudaStreamSynchronize(resources_.stream());
            return false;
        }
        if (r.volume.location == EMemoryLocation::Host)
            YK_CUDA_CHECK(cudaMemcpyAsync(r.volume.data, volume, volumeBytes_(),
                cudaMemcpyDeviceToHost, resources_.stream()));
        YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        return true;
    }

    bool executeWfbp_(const ExecuteRequest& r)
    {
#if YKCBCT_HAS_HELICAL
        if (r.K != params_.scan.NAng || !r.projection.data || !r.volume.data)
            return false;
        // Host 适配统一复用 Session 分配器；仍为全量执行，不表示支持分包。
        const size_t bytes = static_cast<size_t>(r.K) * params_.scan.Nu *
            params_.scan.Nv * sizeof(float);
        YK_LOGI("[Session] WFBP full input={} bytes, volume={} bytes; workspace additional",
            bytes, volumeBytes_());
        size_t free_bytes = 0, total_bytes = 0;
        const size_t needed = (r.projection.location == EMemoryLocation::Host &&
            projection_scratch_bytes_ < bytes ? bytes : 0) +
            (r.volume.location == EMemoryLocation::Host && !d_volume_scratch_ ? volumeBytes_() : 0);
        const auto memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (memory_status != cudaSuccess) {
            YK_LOGE("[Session] cudaMemGetInfo failed: {}", cudaGetErrorString(memory_status));
            return false;
        }
        if (needed > free_bytes) {
            // WDDM 可通过分页满足超过当前空闲物理显存的分配；查询值只用于诊断，
            // 不能作为硬性拒绝条件，否则会拒绝原本可执行的全量重建。
            YK_LOGW("[Session] WFBP staging needs {} bytes, physical free {} bytes; allocation may page or fail (workspace already allocated)",
                needed, free_bytes);
        }
        float* projection = r.projection.location == EMemoryLocation::Device
            ? r.projection.data : ensureProjectionScratch_(r.K);
        float* volume = r.volume.location == EMemoryLocation::Device
            ? r.volume.data : ensureVolumeScratch_();
        if (!projection || !volume) return false;
        if (r.projection.location == EMemoryLocation::Host)
            YK_CUDA_CHECK(cudaMemcpyAsync(projection, r.projection.data, bytes,
                cudaMemcpyHostToDevice, resources_.stream()));
        if (!wfbp_.reconstruct(projection, volume)) {
            cudaStreamSynchronize(resources_.stream());
            return false;
        }
        if (r.volume.location == EMemoryLocation::Host)
            YK_CUDA_CHECK(cudaMemcpyAsync(r.volume.data, volume, volumeBytes_(),
                cudaMemcpyDeviceToHost, resources_.stream()));
        YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
        return true;
#else
        (void)r;
        return false;
#endif
    }

    bool hasFullDeviceReconstructionInput_(const ExecuteRequest& r,
        const char* algorithm) const
    {
        if (r.K != params_.scan.NAng || !r.projection.data || !r.volume.data ||
            r.projection.location != EMemoryLocation::Device ||
            r.volume.location != EMemoryLocation::Device) {
            YK_LOGE("[Session] {} requires complete device projection and device volume buffers.", algorithm);
            return false;
        }
        return true;
    }

    bool executeFp_(const ExecuteRequest& r)
    {
        if (r.K <= 0 || !r.projection.data || !r.volume.data) return false;
        if (cylindrical_) {
            if (r.K != params_.scan.totalViews) {
                YK_LOGE("[Session] Cyl FP 当前要求一次提交完整视图序列。");
                return false;
            }
            const float* d_volume = r.volume.location == EMemoryLocation::Device
                ? r.volume.data : uploadVolume_(r.volume.data);
            float* d_projection = r.projection.location == EMemoryLocation::Device
                ? r.projection.data : ensureProjectionScratch_(r.K);
            if (!d_volume || !d_projection || !cyl_forward_ ||
                !cyl_forward_->apply(d_volume, d_projection, false, resources_))
                return false;
            if (r.projection.location == EMemoryLocation::Host) {
                const size_t bytes = static_cast<size_t>(r.K) *
                    params_.scan.Nu * params_.scan.Nv * sizeof(float);
                YK_CUDA_CHECK(cudaMemcpyAsync(r.projection.data, d_projection,
                    bytes, cudaMemcpyDeviceToHost, resources_.stream()));
            }
            YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
            return true;
        }
        // 外部 geometry 模式下 angle.x 已是唯一角度来源。当前 ExecuteRequest
        // 没有 batch offset，因此只允许一次提交完整序列，避免用另一份 angles
        // 去猜测子集并造成几何/角度分叉。
        if (geometry_.hasExternalGeometry() && r.K != params_.scan.totalViews) {
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
        SReconstructionParams batch = geometry_.batch(batch_angles, r.K);
        batch.scan.totalViews = r.K;
        if (!forward_ || !forward_->apply(d_volume, batch, d_projection, resources_)) return false;
        if (r.projection.location == EMemoryLocation::Host) {
            const size_t bytes = static_cast<size_t>(r.K) * params_.scan.Nu * params_.scan.Nv * sizeof(float);
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
        if (cylindrical_) {
            if (r.K != params_.scan.NAng) {
                YK_LOGE("[Session] Cyl iterative 当前要求完整设备投影序列。");
                return false;
            }
            const bool ok = reconstruction_.pipeline == EPipeline::PWLS
                ? cyl_pwls_.reconstruct(r.projection.data, r.volume.data)
                : cyl_iterative_.reconstruct(r.projection.data, r.volume.data);
            if (!ok) return false;
            YK_CUDA_CHECK(cudaStreamSynchronize(resources_.stream()));
            return true;
        }
        bool ok = false;
        switch (reconstruction_.pipeline) {
        case EPipeline::SART:
            ok = sart_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::SIRT:
            ok = r.iteration_count > 0
                ? sirt_.iterate(r.projection.data, r.volume.data, params_, resources_.stream(), r.iteration_count)
                : sirt_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::OSSART:
            ok = r.iteration_count > 0
                ? ossart_.iterate(r.projection.data, r.volume.data, params_, resources_.stream(),
                    static_cast<unsigned int>(r.iteration_count) *
                        static_cast<unsigned int>(std::max(1, reconstruction_.iterative.subsets)))
                : ossart_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::CGLS:
            if (r.iteration_count > 0)
                YK_LOGW("[Session] CGLS iteration_count is fixed at initialize time.");
            ok = cgls_.run(r.projection.data, r.volume.data, params_, resources_.stream());
            break;
        case EPipeline::PWLS:
            ok = pwls_.reconstruct(r.projection.data, r.volume.data);
            break;
        case EPipeline::TigreGradient:
            ok = tigre_.reconstruct(r.projection.data, r.volume.data);
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
            d_volume_scratch_ = memory_.allocateDevice3D<float>(
                params_.volume.Nx, params_.volume.Ny, params_.volume.Nz,
                device_, false);
        return d_volume_scratch_.data();
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
        const size_t needed = static_cast<size_t>(k) * params_.scan.Nu * params_.scan.Nv * sizeof(float);
        if (needed > projection_scratch_bytes_) {
            d_projection_scratch_ = memory_.allocateDevice3D<float>(
                params_.scan.Nu, params_.scan.Nv, k, device_, false);
            projection_scratch_bytes_ = needed;
        }
        return d_projection_scratch_.data();
    }
    size_t volumeBytes_() const { return static_cast<size_t>(params_.volume.Nx) * params_.volume.Ny * params_.volume.Nz * sizeof(float); }
    void freeScratch_()
    {
        d_volume_scratch_ = {};
        d_projection_scratch_ = {};
        projection_scratch_bytes_ = 0;
    }

    SReconstructionSpec reconstruction_{};
    SReconstructionParams params_{};
    int device_ = 0;
    bool initialized_ = false;
    bool cylindrical_ = false;
    FdkPipeline fdk_;
    PreparedGeometry geometry_;
    ResourceContext resources_;
    std::unique_ptr<IForwardOperator> forward_;
    std::unique_ptr<IBackOperator> backward_;
    std::unique_ptr<CylFpBp::IForwardProjection> cyl_forward_;
    CylFpBp::AlgebraicReconstructor cyl_iterative_{};
    CylFpBp::CylPwlsReconstructor cyl_pwls_{};
#if YKCBCT_HAS_HELICAL
    Helical::Wfbp::Pipeline wfbp_{};
#endif
    SART sart_;
    SIRT sirt_;
    OSSART ossart_;
    CGLS cgls_;
    Iter::PwlsReconstructor pwls_;
    Iter::TigreGradientReconstructorEx tigre_;
    Fdk::XfdkPipeline xfdk_;
    Fdk::CurveFilteredFdkPipeline cfdk_;
    CylFpBp::Analytic::Reconstruction cyl_fdk_;
    Mem::MemoryController memory_{};
    Mem::DeviceLinearBuffer3D<float> d_volume_scratch_{};
    Mem::DeviceLinearBuffer3D<float> d_projection_scratch_{};
    size_t projection_scratch_bytes_ = 0;
};

} // namespace

detail::IExecutionBackend* detail::createExecutionBackend() { return new Session(); }
void detail::destroyExecutionBackend(detail::IExecutionBackend* backend) { delete backend; }

} // namespace YK
