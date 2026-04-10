#pragma once
#include "../interface/IFdkReconstructor.hpp"

#define IN_TEST

#ifdef IN_TEST
#  define YKFDK_API
#else
#  ifdef YKFDK_EXPORTS
#    define YKFDK_API __declspec(dllexport)
#  else
#    define YKFDK_API __declspec(dllimport)
#  endif
#endif

namespace YK {

    // ----------------------------------------------------------------
    // FdkFactory
    //
    //   跨 DLL 的唯一创建 / 销毁点。
    //   new / delete 在同一 CRT 侧执行，避免堆损坏。
    //
    //   推荐用法：
    //     auto* r = YK::FdkFactory::create();
    //     // ... 使用 ...
    //     YK::FdkFactory::destroy(r);
    //
    //   或配合自定义 deleter：
    //     auto r = std::unique_ptr<YK::IReconstructor,
    //                  decltype(&YK::FdkFactory::destroy)>(
    //                      YK::FdkFactory::create(),
    //                      YK::FdkFactory::destroy);
    // ----------------------------------------------------------------
    struct YKFDK_API FdkFactory {
        static IReconstructor* create();
        static void            destroy(IReconstructor* p);
    };

} // namespace YK