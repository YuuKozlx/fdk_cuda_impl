#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include "interface/YkExecutionBackend.hpp"
#include "Reconstruction/Iterative/Regularization/RegularizerPlugin.hpp"
#include "Reconstruction/Iterative/Statistical/StatisticalContracts.hpp"
#include "common/YkExecutionProgress.hpp"
#include "Reconstruction/Iterative/Common/YkProjectionInput.hpp"
#include "global/YkLog.h"

#include <cmath>
#include <memory>
#include <string>

namespace YK {
namespace {

enum class EReleaseDecision { Released, UnderTest, Invalid };

EReleaseDecision releaseDecision(const SSystemSpec& source)
{
    const auto detector = source.geometry.detector;
    const auto trajectory = source.geometry.trajectory;
    if ((detector != EDetectorKind::Flat &&
         detector != EDetectorKind::Cylindrical) ||
        (trajectory != ETrajectoryKind::Circular &&
         trajectory != ETrajectoryKind::Helical &&
         trajectory != ETrajectoryKind::PlanarSourceOrbit &&
         trajectory != ETrajectoryKind::NikonTilted))
        return EReleaseDecision::Invalid;

    const bool cyl = detector == EDetectorKind::Cylindrical;
    const bool helical = trajectory == ETrajectoryKind::Helical;
    const bool nonstandard = trajectory == ETrajectoryKind::PlanarSourceOrbit ||
        trajectory == ETrajectoryKind::NikonTilted;
    if (nonstandard) {
        if (cyl) return EReleaseDecision::Invalid;
        return source.reconstruction.pipeline == EPipeline::FDK ||
            source.reconstruction.pipeline == EPipeline::ForwardProjection
            ? EReleaseDecision::Released : EReleaseDecision::Invalid;
    }
    switch (source.reconstruction.pipeline) {
    case EPipeline::ForwardProjection:
        return EReleaseDecision::Released;
    case EPipeline::FDK:
        return !helical ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::XFDK:
        return !cyl && !helical
            ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::SART:
    case EPipeline::SIRT:
    case EPipeline::OSSART:
    case EPipeline::CGLS:
    case EPipeline::PWLS:
        return EReleaseDecision::Released;
    case EPipeline::Statistical:
        return !cyl?EReleaseDecision::Released:EReleaseDecision::UnderTest;
    case EPipeline::TigreGradient:
        return !cyl ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::WFBP:
        return helical
            ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::CFDK:
        return EReleaseDecision::UnderTest;
    }
    return EReleaseDecision::Invalid;
}

bool validProjectorSelection(const SSystemSpec& source)
{
    return source.reconstruction.projection_model == EProjectionModel::Joseph ||
        source.reconstruction.projection_model == EProjectionModel::Siddon;
}

bool requiresCanonicalAcquisitionOffsets(EPipeline pipeline)
{
    // 这些解析重建使用各自论文中的规范采集参数，当前没有从完整
    // vector geometry 严格派生任意源端/探测器平移的公式。
    switch (pipeline) {
    case EPipeline::XFDK:
    case EPipeline::CFDK:
    case EPipeline::WFBP:
        return true;
    default:
        return false;
    }
}

bool hasNonzeroAcquisitionOffset(const SSystemConfig& geometry,
    EPipeline pipeline)
{
    const float3 source = geometry.trajectory == ETrajectoryKind::Helical
        ? geometry.helical.source_offset_mm
        : geometry.circular.source_offset_mm;
    const auto& pose = geometry.detector == EDetectorKind::Cylindrical
        ? geometry.cylindrical_detector.pose
        : geometry.flat_detector.pose;
    constexpr float tolerance = 1e-7f;
    const bool source_offset = std::fabs(source.x) > tolerance ||
        std::fabs(source.y) > tolerance || std::fabs(source.z) > tolerance;
    const bool cyl_fdk = pipeline == EPipeline::FDK &&
        geometry.detector == EDetectorKind::Cylindrical;
    // Cyl-FDK preserves detector V translation in the resampled flat principal point.
    const bool detector_offset = pipeline == EPipeline::WFBP && geometry.detector == EDetectorKind::Cylindrical
        ? std::fabs(pose.offset_unv_mm.y) > tolerance
        : std::fabs(pose.offset_unv_mm.x) > tolerance ||
            std::fabs(pose.offset_unv_mm.y) > tolerance ||
            (!cyl_fdk && std::fabs(pose.offset_unv_mm.z) > tolerance);
    const bool detector_tilt = (cyl_fdk || (pipeline == EPipeline::WFBP && geometry.detector == EDetectorKind::Flat)) &&
        (!std::isfinite(pose.tilt_u_rad) || !std::isfinite(pose.tilt_v_rad) ||
         !std::isfinite(pose.tilt_n_rad) ||
         std::fabs(pose.tilt_u_rad) > tolerance ||
         std::fabs(pose.tilt_v_rad) > tolerance ||
         std::fabs(pose.tilt_n_rad) > tolerance);
    return source_offset || detector_offset || detector_tilt;
}

bool validForwardProjectionPose(const SForwardProjectionPose& pose)
{
    return std::isfinite(pose.rotation_x_rad) &&
        std::isfinite(pose.rotation_y_rad) &&
        std::isfinite(pose.rotation_z_rad);
}


class ReconstructionSession final : public IReconstructionSession {
public:
    ~ReconstructionSession() override { release(); }

    bool initialize(const SSystemSpec& requested) override
    {
        release();
        clearError_();
        const SSystemSpec& system = requested;
        if (!hasBytes(system.struct_size, offsetof(SSystemSpec, device) + sizeof(system.device)))
            return fail_(EApiErrorCode::InvalidConfig, "SSystemSpec 结构体版本过旧或被截断");
        if (system.api_version != 2)
            return fail_(EApiErrorCode::InvalidConfig, "不支持的 SSystemSpec api_version");
        const auto decision = releaseDecision(system);
        if (decision == EReleaseDecision::Invalid)
            return fail_(EApiErrorCode::InvalidConfig, "算法、探测器或轨迹枚举非法");
        if (!validProjectorSelection(system))
            return fail_(EApiErrorCode::InvalidConfig,
                "当前探测器不支持所选 FP/BP 模型，库不会隐式替换为其他版本");
        if (decision == EReleaseDecision::UnderTest)
            return fail_(EApiErrorCode::AlgorithmUnderTest,
                "该算法或几何组合仍在测试中，当前 DLL 未放行");
#if !YKCBCT_HAS_HELICAL
        if (system.reconstruction.pipeline == EPipeline::WFBP)
            return fail_(EApiErrorCode::UnsupportedCombination,
                "当前 DLL 构建未启用 YKCBCT_BUILD_HELICAL，缺少 wFBP 后端");
#endif
        const bool nonstandard =
            system.geometry.trajectory == ETrajectoryKind::PlanarSourceOrbit ||
            system.geometry.trajectory == ETrajectoryKind::NikonTilted;
        if (nonstandard &&
            !hasBytes(system.struct_size,
                offsetof(SSystemSpec, nonstandard) + sizeof(system.nonstandard)))
            return fail_(EApiErrorCode::InvalidConfig,
                "Non-standard trajectories require the SSystemSpec nonstandard tail");
        SSystemConfig backend_geometry = system.geometry;
        if (!validForwardProjectionPose(backend_geometry.forward_projection_pose))
            return fail_(EApiErrorCode::InvalidGeometry,
                "Forward-projection volume rotation must contain finite radians");
        if (system.reconstruction.pipeline != EPipeline::ForwardProjection)
            backend_geometry.forward_projection_pose = {};
        if ((requiresCanonicalAcquisitionOffsets(system.reconstruction.pipeline) ||
             (system.reconstruction.pipeline == EPipeline::FDK &&
              system.geometry.detector == EDetectorKind::Cylindrical)) &&
            hasNonzeroAcquisitionOffset(backend_geometry,
                system.reconstruction.pipeline)) {
            return fail_(EApiErrorCode::UnsupportedCombination,
                "Algorithm does not support the requested acquisition offsets or detector tilt; geometry was not changed");
        }
        if (system.struct_size > offsetof(SSystemSpec, regularization) &&
            !hasBytes(system.struct_size, offsetof(SSystemSpec,regularization)+sizeof(system.regularization)))
            return fail_(EApiErrorCode::InvalidConfig,"SSystemSpec regularization tail is truncated");
        const auto* regularization = hasBytes(system.struct_size,
            offsetof(SSystemSpec, regularization) + sizeof(system.regularization))
            ? &system.regularization : nullptr;
        if (regularization && regularization->plugin_id &&
            regularization->plugin_id[0] && std::string(regularization->plugin_id) != "none" &&
            (system.geometry.detector != EDetectorKind::Flat ||
             (system.reconstruction.pipeline != EPipeline::SART &&
              system.reconstruction.pipeline != EPipeline::SIRT &&
              system.reconstruction.pipeline != EPipeline::OSSART)))
            return fail_(EApiErrorCode::UnsupportedCombination,
                "TV plugins currently support flat SART/SIRT/OSSART only");
        if (regularization && regularization->plugin_id && regularization->plugin_id[0] &&
            std::string(regularization->plugin_id)!="none") {
            if (!Iter::createRegularizerPlugin(regularization->plugin_id))
                return fail_(EApiErrorCode::InvalidConfig,"Unknown regularization plugin ID");
            const auto& c=*regularization;
            if (!std::isfinite(c.strength)||c.strength<0||c.inner_iterations<=0||
                !std::isfinite(c.strength_reduction)||c.strength_reduction<=0||
                (c.dimensions!=2&&c.dimensions!=3))
                return fail_(EApiErrorCode::InvalidConfig,"Invalid TV strength, dimensions or inner iterations");
            const std::string id=c.plugin_id;
            if (id=="huber_tv_prox") {
                if (!std::isfinite(c.huber_delta)||c.huber_delta<=0||!std::isfinite(c.step_size)||c.step_size<0)
                    return fail_(EApiErrorCode::InvalidConfig,"Huber delta must be positive; Prox step must be nonnegative");
                const auto& v=system.geometry.volume;
                if (v.voxel_x_mm>0&&v.voxel_y_mm>0&&v.voxel_z_mm>0) {
                    const double hx=v.voxel_x_mm,hy=v.voxel_y_mm,hz=v.voxel_z_mm;
                    const double bound=1+double(c.strength)*4*(1/(hx*hx)+1/(hy*hy)+(c.dimensions==3?1/(hz*hz):0))/c.huber_delta;
                    if (c.step_size>0&&c.step_size>1/bound*(1+1e-6))
                        return fail_(EApiErrorCode::InvalidConfig,"Prox step exceeds safe bound; use step_size=0 for automatic step");
                }
            } else if (id=="smoothed_tv"&&(!std::isfinite(c.epsilon)||c.epsilon<=0))
                return fail_(EApiErrorCode::InvalidConfig,"Smoothed TV epsilon must be positive");
        }
        const SStatisticalReconstructionSpec* statistical=nullptr;
        if(system.struct_size>offsetof(SSystemSpec,statistical)&&
            !hasBytes(system.struct_size,offsetof(SSystemSpec,statistical)+sizeof(system.statistical)))
            return fail_(EApiErrorCode::InvalidConfig,"SSystemSpec statistical tail is truncated");
        if(system.reconstruction.pipeline==EPipeline::Statistical){
            if(!hasBytes(system.struct_size,offsetof(SSystemSpec,statistical)+sizeof(system.statistical)))
                return fail_(EApiErrorCode::InvalidConfig,"Statistical pipeline requires the statistical tail");
            statistical=&system.statistical;const auto& c=*statistical;
            auto optimizer=Iter::Statistical::createOptimizer(c.optimizer.kind);
            if(!optimizer||!optimizer->configureComposition(Iter::Statistical::defaultComposition(c)))
                return fail_(EApiErrorCode::UnsupportedCombination,"Invalid regularization composition for optimizer");
            const auto required=optimizer->requirements();
            if(c.optimizer.kind!=EStatisticalOptimizer::GradientDescent&&c.optimizer.kind!=EStatisticalOptimizer::RegularizedAlgebraic)
                return fail_(EApiErrorCode::UnsupportedCombination,"Requested statistical optimizer is not implemented; FISTA is reserved");
            if(c.data_fidelity.kind!=EDataFidelityKind::WeightedLeastSquares||c.data_fidelity.require_verified_adjoint)
                return fail_(EApiErrorCode::UnsupportedCombination,"This data fidelity/operator combination is unavailable or not adjoint-verified");
            const std::string id=c.regularization.plugin_id?c.regularization.plugin_id:"none";
            if(!std::isfinite(c.data_fidelity.scale)||c.data_fidelity.scale<=0||
                !std::isfinite(c.regularization.strength)||c.regularization.strength<0||
                (c.regularization.dimensions!=2&&c.regularization.dimensions!=3)||
                c.optimizer.iterations<=0||!std::isfinite(c.optimizer.step_size)||c.optimizer.step_size<0||
                c.optimizer.max_backtracking<0||!std::isfinite(c.optimizer.step_shrink)||c.optimizer.step_shrink<=0||c.optimizer.step_shrink>=1||
                !std::isfinite(c.optimizer.relative_tolerance)||c.optimizer.relative_tolerance<0||c.optimizer.minimum_iterations<=0||
                !std::isfinite(c.constraints.lower_bound)||!std::isfinite(c.constraints.upper_bound)||c.constraints.lower_bound>c.constraints.upper_bound)
                return fail_(EApiErrorCode::InvalidConfig,"Invalid statistical data/regularization/optimizer/constraint parameters");
            if(id!="none"&&!id.empty()){
                auto plugin=Iter::createRegularizerPlugin(id);
                if(!plugin)return fail_(EApiErrorCode::InvalidConfig,"Unknown statistical regularizer");
                if((required.regularization_gradient&&!plugin->capabilities().gradient)||(required.regularization_prox&&!plugin->capabilities().prox)||!plugin->capabilities().value)
                    return fail_(EApiErrorCode::UnsupportedCombination,"Regularizer does not support selected strategy; standard TV requires Proximal");
                if(id!="tikhonov"&&id!="tv"&&(!std::isfinite(c.regularization.shape)||c.regularization.shape<=0))
                    return fail_(EApiErrorCode::InvalidConfig,"Statistical regularization shape must be positive");
            }else if(c.regularization.strength>0)return fail_(EApiErrorCode::InvalidConfig,"Positive strength requires a regularizer");
        }
        session_ = detail::createExecutionBackend();
        if (!session_ || !session_->initialize(backend_geometry,
                system.reconstruction,
                nonstandard ? &system.nonstandard : nullptr, system.device, regularization, statistical)) {
            const std::string message=session_&&session_->lastErrorMessage()?session_->lastErrorMessage():"几何或算法初始化失败";
            release();
            return fail_(EApiErrorCode::InvalidGeometry,message.c_str());
        }
        if (system.geometry.trajectory == ETrajectoryKind::Helical)
            view_count_ = system.geometry.helical.total_views;
        else if (system.geometry.trajectory == ETrajectoryKind::PlanarSourceOrbit)
            view_count_ = system.nonstandard.planar_source_orbit.total_views;
        else if (system.geometry.trajectory == ETrajectoryKind::NikonTilted)
            view_count_ = system.nonstandard.nikon_tilted.total_views;
        else
            view_count_ = system.geometry.circular.total_views;
        const int channels = system.geometry.detector == EDetectorKind::Cylindrical
            ? system.geometry.cylindrical_detector.channels
            : system.geometry.flat_detector.channels;
        const int rows = system.geometry.detector == EDetectorKind::Cylindrical
            ? system.geometry.cylindrical_detector.rows
            : system.geometry.flat_detector.rows;
        projection_view_elements_ = static_cast<size_t>(channels) * rows;
        volume_elements_ = static_cast<size_t>(system.geometry.volume.nx) *
            system.geometry.volume.ny * system.geometry.volume.nz;
        streaming_fdk_ = system.reconstruction.pipeline == EPipeline::FDK &&
            system.geometry.detector == EDetectorKind::Flat && !nonstandard;
        batched_iterative_=system.geometry.detector==EDetectorKind::Flat &&
            (system.reconstruction.pipeline==EPipeline::Statistical || system.reconstruction.pipeline==EPipeline::SART || system.reconstruction.pipeline==EPipeline::SIRT || system.reconstruction.pipeline==EPipeline::OSSART ||
             (system.reconstruction.pipeline==EPipeline::TigreGradient &&
              (system.reconstruction.tigre.method==ETigreGradientMethodSpec::Sart || system.reconstruction.tigre.method==ETigreGradientMethodSpec::OsSart || system.reconstruction.tigre.method==ETigreGradientMethodSpec::Sirt || system.reconstruction.tigre.method==ETigreGradientMethodSpec::AsdPocs || system.reconstruction.tigre.method==ETigreGradientMethodSpec::OsAsdPocs) &&
              !system.reconstruction.tigre.nesterov_relaxation && !system.reconstruction.tigre.fdk_initialization));
        statistical_=system.reconstruction.pipeline==EPipeline::Statistical;
        statistical_weights_=statistical_&&system.statistical.data_fidelity.use_projection_weights;
        received_views_ = 0;
        initialized_ = true;
        return true;
    }

    bool execute(const SExecutionRequest& request) override
    {
        clearError_();
        if (!initialized_)
            return fail_(EApiErrorCode::StreamingStateError, "Session 尚未初始化");
        if (!hasBytes(request.struct_size, offsetof(SExecutionRequest, iteration_count) + sizeof(request.iteration_count)))
            return fail_(EApiErrorCode::InvalidConfig, "SExecutionRequest 结构体版本过旧或被截断");
        if(request.struct_size>offsetof(SExecutionRequest,progress_callback) &&
            request.struct_size<offsetof(SExecutionRequest,progress_user_data)+sizeof(request.progress_user_data))
            return fail_(EApiErrorCode::InvalidConfig,"SExecutionRequest progress tail is truncated");
        if (request.api_version != 2)
            return fail_(EApiErrorCode::InvalidConfig, "不支持的 SExecutionRequest api_version");
        if(request.struct_size>offsetof(SExecutionRequest,projection_read) &&
            request.struct_size<offsetof(SExecutionRequest,projection_batch_mib)+sizeof(request.projection_batch_mib))
            return fail_(EApiErrorCode::InvalidConfig,"SExecutionRequest projection source tail is truncated");
        const bool weights_tail=hasBytes(request.struct_size,offsetof(SExecutionRequest,projection_weights_read_user_data)+sizeof(request.projection_weights_read_user_data));
        if(request.struct_size>offsetof(SExecutionRequest,projection_weights)&&!weights_tail)
            return fail_(EApiErrorCode::InvalidConfig,"SExecutionRequest statistical weights tail is truncated");
        const bool supplied_weights=weights_tail&&(request.projection_weights.data||request.projection_weights_read);
        if(supplied_weights&&!statistical_weights_)
            return fail_(EApiErrorCode::UnsupportedCombination,"Projection W is only accepted when statistical diagonal weights are enabled");
        if(statistical_weights_&&(!supplied_weights||
            !validLocation_(request.projection_weights.location)||
            (!request.projection_weights_read&&request.projection_weights.element_count<projection_view_elements_*view_count_)))
            return fail_(EApiErrorCode::InvalidBuffer,"Statistical diagonal W requires a complete weight buffer or reader");
        const bool source_tail=hasBytes(request.struct_size,offsetof(SExecutionRequest,projection_batch_mib)+sizeof(request.projection_batch_mib));
        const auto reader=source_tail?request.projection_read:nullptr;
        const bool batch=reader || (batched_iterative_&&request.projection.location==EMemoryLocation::Host);
        if(reader&&!batched_iterative_)return fail_(EApiErrorCode::UnsupportedCombination,"Projection reader requires supported flat SART/SIRT/OSSART or TIGRE SART/OS-SART/SIRT/TV without Nesterov/FDK initialization");
        if((batch||statistical_weights_) && source_tail && (request.projection_batch_views<=0 || request.projection_batch_mib<=0 || request.projection_batch_mib>512))
            return fail_(EApiErrorCode::InvalidConfig,"Invalid projection batch views/MiB budget");
        if ((!reader&&!request.projection.data) || !request.volume.data ||
            !validLocation_(request.projection.location) || !validLocation_(request.volume.location))
            return fail_(EApiErrorCode::InvalidBuffer, "projection/volume 缓冲区为空或位置枚举非法");
        const int offset = request.view_offset;
        const int count = request.view_count > 0 ? request.view_count : view_count_;
        if (offset < 0 || count <= 0 || offset > view_count_ - count)
            return fail_(EApiErrorCode::InvalidConfig, "view_offset/view_count 超出已准备的角度范围");
        const bool streaming_fdk = streaming_fdk_;
        if (streaming_fdk) {
            if (offset != received_views_)
                return fail_(EApiErrorCode::StreamingStateError,
                    "FDK 分包必须按 view_offset 连续提交，不能跳帧或重复提交");
            if (received_views_ > 0 && request.clear_output)
                return fail_(EApiErrorCode::StreamingStateError,
                    "FDK 仅首包允许 clear_output=true");
        }
        else if (offset != 0 || count != view_count_) {
            return fail_(EApiErrorCode::StreamingStateError,
                "当前算法不支持分包，必须从 view_offset=0 提交完整投影序列");
        }
        const size_t required_projection = projection_view_elements_ * count;
        if ((!reader&&request.projection.element_count < required_projection) ||
            request.volume.element_count < volume_elements_)
            return fail_(EApiErrorCode::InvalidBuffer,
                "projection/volume 的 element_count 小于本次执行所需容量");
        ExecuteRequest backend_request{};
        // 后端使用初始化时生成的逐视图 geometry；执行请求不再携带第二份角度。
        backend_request.angles = nullptr;
        backend_request.K = count;
        backend_request.projection = request.projection;
        backend_request.volume = request.volume;
        backend_request.clear_output = request.clear_output;
        backend_request.iteration_count = request.iteration_count;
        if(weights_tail){backend_request.projection_weights=request.projection_weights;
            backend_request.projection_weights_read=request.projection_weights_read;
            backend_request.projection_weights_read_user_data=request.projection_weights_read_user_data;}
        if(source_tail){backend_request.projection_batch_views=request.projection_batch_views;backend_request.projection_batch_mib=request.projection_batch_mib;}
        const bool has_progress=hasBytes(request.struct_size,offsetof(SExecutionRequest,progress_user_data)+sizeof(request.progress_user_data));
        detail::ScopedExecutionProgress progress(has_progress?request.progress_callback:nullptr,has_progress?request.progress_user_data:nullptr);
        std::unique_ptr<detail::ProjectionInput> source;
        if(batch)source=std::make_unique<detail::ProjectionInput>(reader,source_tail?request.projection_read_user_data:nullptr,
            request.projection.data,projection_view_elements_,view_count_,source_tail?request.projection_batch_views:64,
            source_tail?request.projection_batch_mib:256,volume_elements_*sizeof(float));
        detail::ScopedProjectionInput projection_source(source.get());
        if (!session_->execute(backend_request)) {
            if(source&&!source->error().empty())return fail_(EApiErrorCode::InvalidBuffer,source->error().c_str());
            const char* backend_error=session_->lastErrorMessage();
            return fail_(EApiErrorCode::CudaError,backend_error&&backend_error[0]?backend_error:"后端执行失败，请检查 CUDA 状态和输入尺寸");
        }
        if (streaming_fdk) received_views_ += count;
        return true;
    }

    bool isInitialized() const override { return initialized_; }
    EApiErrorCode lastError() const override { return error_code_; }
    const char* lastErrorMessage() const override { return error_message_.c_str(); }
    void reset() override
    {
        if (session_) session_->reset();
        received_views_ = 0;
    }
    void release() override
    {
        if (session_) detail::destroyExecutionBackend(session_);
        session_ = nullptr; view_count_ = received_views_ = 0;
        projection_view_elements_ = volume_elements_ = 0;
        initialized_ = false;batched_iterative_=false;statistical_=statistical_weights_=false;
    }

private:
    static bool hasBytes(uint32_t actual, size_t required)
    { return actual >= required; }

    static bool validLocation_(EMemoryLocation location)
    { return location == EMemoryLocation::Host || location == EMemoryLocation::Device; }

    bool fail_(EApiErrorCode code, const char* message)
    { error_code_ = code; error_message_ = message; return false; }

    void clearError_()
    { error_code_ = EApiErrorCode::None; error_message_.clear(); }

    detail::IExecutionBackend* session_ = nullptr;
    int view_count_ = 0;
    int received_views_ = 0;
    size_t projection_view_elements_ = 0;
    size_t volume_elements_ = 0;
    bool streaming_fdk_ = false;
    bool batched_iterative_=false;
    bool statistical_=false,statistical_weights_=false;
    bool initialized_ = false;
    EApiErrorCode error_code_ = EApiErrorCode::None;
    std::string error_message_;
};

} // namespace

bool estimateStatisticalWorkspace(const SSystemSpec& s,SStatisticalWorkspaceEstimate& out){
    out={};
    if(s.struct_size<offsetof(SSystemSpec,statistical)+sizeof(s.statistical)||s.geometry.detector!=EDetectorKind::Flat||
       (s.geometry.trajectory!=ETrajectoryKind::Circular&&s.geometry.trajectory!=ETrajectoryKind::Helical))return false;
    const auto& g=s.geometry;SReconstructionParams p{};
    p.scan.Nu=g.flat_detector.channels;p.scan.Nv=g.flat_detector.rows;
    p.scan.NAng=g.trajectory==ETrajectoryKind::Helical?g.helical.total_views:g.circular.total_views;
    p.volume.Nx=g.volume.nx;p.volume.Ny=g.volume.ny;p.volume.Nz=g.volume.nz;
    p.volume.voxelX_mm=g.volume.voxel_x_mm;p.volume.voxelY_mm=g.volume.voxel_y_mm;p.volume.voxelZ_mm=g.volume.voxel_z_mm;
    if(p.scan.Nu<=0||p.scan.Nv<=0||p.scan.NAng<=0||p.volume.Nx<=0||p.volume.Ny<=0||p.volume.Nz<=0)return false;
    const size_t n=size_t(p.volume.Nx)*p.volume.Ny*p.volume.Nz,pixels=size_t(p.scan.Nu)*p.scan.Nv;
    auto data=Iter::Statistical::createDataFidelity(s.statistical.data_fidelity.kind);
    auto optimizer=Iter::Statistical::createOptimizer(s.statistical.optimizer.kind);
    if(!data||!optimizer)return false;
    if(!optimizer->configureComposition(Iter::Statistical::defaultComposition(s.statistical)))return false;
    const auto required=optimizer->requirements();
    out.optimizer_bytes=optimizer->workspaceBytes(n);out.data_fidelity_bytes=data->workspaceBytes(p)+(optimizer->requirements().data_algebraic_update?data->algebraicWorkspaceBytes(p):0);
    const auto& reg=s.statistical.regularization;const std::string id=reg.plugin_id?reg.plugin_id:"none";
    if(!id.empty()&&id!="none"){
        auto plugin=Iter::createRegularizerPlugin(id);if(!plugin)return false;
        if((required.regularization_gradient&&!plugin->capabilities().gradient)||(required.regularization_prox&&!plugin->capabilities().prox)||!plugin->capabilities().value)return false;
        Iter::AlgebraicRegularizationConfig cfg;cfg.gradient_only=!required.regularization_prox;cfg.require_prox=required.regularization_prox;cfg.strength=reg.strength;cfg.huber_delta=cfg.epsilon=reg.shape;
        cfg.tv_dimensionality=static_cast<Iter::ETvDimensionality>(reg.dimensions);
        out.regularization_bytes=plugin->workspaceBytes(p,cfg);
    }
    out.caller_volume_bytes=n*sizeof(float);
    out.projection_staging_bytes=size_t(detail::iterativeBatchViews(pixels,p.scan.NAng))*pixels*sizeof(float);
    out.weight_staging_bytes=s.statistical.data_fidelity.use_projection_weights?out.projection_staging_bytes:0;
    return true;
}

IReconstructionSession* ReconstructionSessionFactory::create()
{ return new ReconstructionSession(); }

void ReconstructionSessionFactory::destroy(IReconstructionSession* session)
{ delete session; }

} // namespace YK
