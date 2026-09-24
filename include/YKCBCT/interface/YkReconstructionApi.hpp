#pragma once

#include <cstdint>
#include <cstddef>

#include "YKCBCT/global/YkExport.hpp"
#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"
#include "YKCBCT/interface/YkReconstructionTypes.hpp"

namespace YK {

// 新接口的公开系统描述。用户只选择一种宏观系统几何；逐视图 geometry
// 由 builder 在初始化阶段生成或校准，不要求调用方拼接底层 kernel 参数。
struct SSystemSpec {
    uint32_t struct_size = sizeof(SSystemSpec);
    uint32_t api_version = 1;
    SSystemConfig geometry{};
    SReconstructionSpec reconstruction{};
    int device = 0;
};

struct SExecutionRequest {
    uint32_t struct_size = sizeof(SExecutionRequest);
    uint32_t api_version = 2;
    Buffer projection{};
    Buffer volume{};
    int view_offset = 0;
    int view_count = 0;
    bool clear_output = true;
    int iteration_count = 0;
};

// 公共接口的失败原因。算法后端仍可继续扩展，但调用方不必再从一个
// 无上下文的 bool 值猜测是配置、缓冲区还是流式状态错误。
enum class EApiErrorCode : int32_t {
    None = 0,
    InvalidConfig,
    InvalidGeometry,
    UnsupportedCombination,
    AlgorithmUnderTest,
    InvalidBuffer,
    StreamingStateError,
    CudaError,
};

class YK_API IReconstructionSession {
public:
    virtual ~IReconstructionSession() = default;
    virtual bool initialize(const SSystemSpec& system) = 0;
    virtual bool execute(const SExecutionRequest& request) = 0;
    virtual bool isInitialized() const = 0;
    virtual EApiErrorCode lastError() const = 0;
    virtual const char* lastErrorMessage() const = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
};

struct YK_API ReconstructionSessionFactory {
    static IReconstructionSession* create();
    static void destroy(IReconstructionSession* session);
};

} // namespace YK
