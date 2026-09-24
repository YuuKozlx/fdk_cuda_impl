#pragma once

#include "YKCBCT/geometry/YkSystemGeometry.hpp"
#include "YKCBCT/interface/YkSystemReconstruction.hpp"
#include "common/YkOperatorTypes.hpp"

namespace YK::detail {

// DLL 外观之后的唯一执行后端。它直接接收统一系统配置，不要求接口层先
// 展开为旧 SessionDesc，也不会在同一次初始化中重复构造逐视图 geometry。
class IExecutionBackend {
public:
    virtual ~IExecutionBackend() = default;
    virtual bool initialize(const SSystemConfig& system,
        const SReconstructionSpec& reconstruction, int device) = 0;
    virtual bool execute(const ExecuteRequest& request) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;
};

IExecutionBackend* createExecutionBackend();
void destroyExecutionBackend(IExecutionBackend* backend);

} // namespace YK::detail
