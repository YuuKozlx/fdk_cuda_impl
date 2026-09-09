#include "YKCBCT/interface/YkReconstructionApi.hpp"
#include "interface/YkExecutionBackend.hpp"

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
         trajectory != ETrajectoryKind::Helical))
        return EReleaseDecision::Invalid;

    const bool cyl = detector == EDetectorKind::Cylindrical;
    const bool helical = trajectory == ETrajectoryKind::Helical;
    switch (source.reconstruction.pipeline) {
    case EPipeline::ForwardProjection:
        return EReleaseDecision::Released;
    case EPipeline::FDK:
    case EPipeline::XFDK:
        return !cyl && !helical
            ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::SIRT:
    case EPipeline::OSSART:
    case EPipeline::CGLS:
    case EPipeline::PWLS:
        return EReleaseDecision::Released;
    case EPipeline::TigreGradient:
        return !cyl ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::WFBP:
        return cyl && helical
            ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::CylAnalyticFDK:
        return cyl && !helical
            ? EReleaseDecision::Released : EReleaseDecision::UnderTest;
    case EPipeline::CFDK:
        return EReleaseDecision::UnderTest;
    }
    return EReleaseDecision::Invalid;
}

bool validProjectorSelection(const SSystemSpec& source)
{
    const auto pipeline = source.reconstruction.pipeline;
    if (pipeline == EPipeline::ForwardProjection &&
        source.reconstruction.forward_projector != ETask::FP_Joseph &&
        source.reconstruction.forward_projector != ETask::FP_Siddon)
        return false;
    if (source.geometry.detector == EDetectorKind::Cylindrical &&
        (pipeline == EPipeline::SIRT || pipeline == EPipeline::OSSART ||
         pipeline == EPipeline::CGLS || pipeline == EPipeline::PWLS) &&
        source.reconstruction.back_projector == ETask::BP_Joseph_v2)
        return false;
    return true;
}

class ReconstructionSession final : public IReconstructionSession {
public:
    ~ReconstructionSession() override { release(); }

    bool initialize(const SSystemSpec& system) override
    {
        release();
        clearError_();
        if (!hasBytes(system.struct_size, offsetof(SSystemSpec, device) + sizeof(system.device)))
            return fail_(EApiErrorCode::InvalidConfig, "SSystemSpec 结构体版本过旧或被截断");
        if (system.api_version != 1)
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
        session_ = detail::createExecutionBackend();
        if (!session_ || !session_->initialize(system.geometry,
                system.reconstruction, system.device)) {
            release();
            return fail_(EApiErrorCode::InvalidGeometry, "几何或算法初始化失败");
        }
        view_count_ = system.geometry.trajectory == ETrajectoryKind::Helical
            ? system.geometry.helical.total_views
            : system.geometry.circular.total_views;
        const int channels = system.geometry.detector == EDetectorKind::Cylindrical
            ? system.geometry.cylindrical_detector.channels
            : system.geometry.flat_detector.channels;
        const int rows = system.geometry.detector == EDetectorKind::Cylindrical
            ? system.geometry.cylindrical_detector.rows
            : system.geometry.flat_detector.rows;
        projection_view_elements_ = static_cast<size_t>(channels) * rows;
        volume_elements_ = static_cast<size_t>(system.geometry.volume.nx) *
            system.geometry.volume.ny * system.geometry.volume.nz;
        pipeline_ = system.reconstruction.pipeline;
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
        if (request.api_version != 2)
            return fail_(EApiErrorCode::InvalidConfig, "不支持的 SExecutionRequest api_version");
        if (!request.projection.data || !request.volume.data ||
            !validLocation_(request.projection.location) || !validLocation_(request.volume.location))
            return fail_(EApiErrorCode::InvalidBuffer, "projection/volume 缓冲区为空或位置枚举非法");
        const int offset = request.view_offset;
        const int count = request.view_count > 0 ? request.view_count : view_count_;
        if (offset < 0 || count <= 0 || offset > view_count_ - count)
            return fail_(EApiErrorCode::InvalidConfig, "view_offset/view_count 超出已准备的角度范围");
        const bool streaming_fdk = pipeline_ == EPipeline::FDK;
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
        if (request.projection.element_count < required_projection ||
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
        if (!session_->execute(backend_request))
            return fail_(EApiErrorCode::CudaError, "后端执行失败，请检查 CUDA 状态和输入尺寸");
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
        initialized_ = false;
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
    EPipeline pipeline_ = EPipeline::FDK;
    bool initialized_ = false;
    EApiErrorCode error_code_ = EApiErrorCode::None;
    std::string error_message_;
};

} // namespace

IReconstructionSession* ReconstructionSessionFactory::create()
{ return new ReconstructionSession(); }

void ReconstructionSessionFactory::destroy(IReconstructionSession* session)
{ delete session; }

} // namespace YK
