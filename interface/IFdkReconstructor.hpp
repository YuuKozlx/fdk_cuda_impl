#pragma once
#include "IReconstructor.hpp"
#include "YkFdkTypes.hpp"

namespace YK {

    // ----------------------------------------------------------------
    // IFdkReconstructor
    //
    //   FDK 锥束重建专属接口，继承通用重建器契约。
    //   若未来新增其他 FDK 变体（如 short-scan only 版本），
    //   均从本接口派生。
    // ----------------------------------------------------------------
    class IFdkReconstructor : public IReconstructor {
    public:
        virtual ~IFdkReconstructor() = default;

        // 分配 GPU 资源，建立 FFT plan，只调一次
        // 返回 false 表示参数非法或 GPU 初始化失败
        virtual bool init(const FdkInitParams& params) = 0;

        // 送入一批投影（K 个视角），累加到 d_vol
        // 库内部自动按 kMaxChunkAng 分批，调用方无需关心 chunk 细节
        // dump_cb / userdata 可为 nullptr（不 dump）
        virtual bool recon(
            const FdkBatchParams& fp,
            FdkDumpCallback      dump_cb = nullptr,
            void* userdata = nullptr) = 0;
    };

} // namespace YK
