#pragma once
namespace YK {

    // ----------------------------------------------------------------
    // IReconstructor
    //
    //   所有重建器的顶层抽象，只定义生命周期契约。
    //   不包含任何算法相关参数，对重建方式无假设。
    // ----------------------------------------------------------------
    class IReconstructor {
    public:
        virtual ~IReconstructor() = default;

        // 开始新一轮扫描前调用，清空累积状态
        virtual void reset() = 0;

        // 释放所有 GPU 资源（析构时自动调用，也可手动提前释放）
        virtual void release() = 0;

        virtual bool isInitialized() const = 0;
        virtual int  totalReceived()  const = 0;
    };

} // namespace YK

