#pragma once
#include <driver_types.h>
#include "tl/expected.hpp"

namespace YK {
    // ============================================================
    // 反卷积/滤波接口
    // ============================================================
    class IProcessor {
    public:
        virtual ~IProcessor() = default;

        // 初始化接口：继承的类可根据需要添加参数；init() 只负责初始化资源/plan，不生成权重（lazy）
        virtual bool init() = 0;

        // 处理接口：输入输出均为 device pointer，in-place 亦可
        virtual void process(const void* d_input, void* d_output, cudaStream_t stream = 0) = 0;

        // 资源释放接口：析构时自动调用；也可手动调用 release() 以提前释放资源
        virtual void release() = 0;

        // 状态查询接口：根据需要添加，如是否已初始化、当前配置等
        virtual bool isInitialized() const = 0;

        // 名称接口：返回处理器名称，便于日志/调试
        virtual const char* name() const = 0;

        // 运行期上下文注入，默认空实现
        // 需要 per-chunk 状态的子类覆盖此接口
        // 调用方在 process() 前调用
        virtual void setContext(const void* /*ctx*/) {}

        // 初始化配置注入（init() 之前调用）
        virtual void setInitContext(const void* /*ctx*/) {}

    };
} // namespace YK