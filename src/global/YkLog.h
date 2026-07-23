#pragma once
// ============================================================
//  yk_log.h  —  .cpp 编译单元日志
//  .cu 文件请用 YkMacro.hpp 中的 YK_LOGI 等宏
// ============================================================

#include <cstdio>
#include <mutex>
#include <functional>

namespace YK {

    enum class LogLevel {
        Trace = 0, Debug, Info, Warn, Error, Critical, Off
    };

    // sink 类型：接收 level、tag、格式化好的完整日志字符串
    using LogSink = std::function<void(LogLevel, const char* tag, const char* msg)>;


    class Logger {
    public:
        static Logger& instance();
        void     set_level(LogLevel lv);
        LogLevel level() const;
        void     log(LogLevel lv, const char* tag, const char* msg);

        // 添加/清除 sink
        void add_sink(LogSink sink);
        void clear_sinks();

        void add_file_sink(const std::string& path);
    private:
        Logger();
        LogLevel min_level_{ LogLevel::Trace };
        std::mutex mutex_;  // 新增
        std::vector<LogSink> sinks_;
    };

} // namespace YK


#ifndef __CUDACC__
// ============================================================
//  以下内容仅 .cpp 编译单元可见
//  .cu 文件中 __CUDACC__ 已定义，跳过，避免与 YkMacro.hpp 冲突
// ============================================================

#ifndef FMT_UNICODE
#  define FMT_UNICODE 0
#endif
#define FMT_HEADER_ONLY
#include "util/fmt/format.h"

#ifndef YK_LOG_TAG
#  define YK_LOG_TAG "YK"
#endif


//// 普通函数名
//__FUNCTION__        // "myFunction"
//
//// 完整签名（MSVC）
//__FUNCSIG__         // "void __cdecl MyClass::myFunction(int)"
//
//// 完整签名（GCC/Clang）
//__PRETTY_FUNCTION__ // "void MyClass::myFunction(int)"
// 
// 
// 只取文件名
#define _YK_FILENAME (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : \
                     (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__))

// 普通版：不带位置
#define _YK_LOG_DEV(lv_, fmt_, ...)                                              \
    do {                                                                         \
        if ((lv_) < YK::Logger::instance().level()) break;                      \
        try {                                                                    \
            auto _yk_s = fmt::format("[{}] " fmt_, __FUNCTION__, ##__VA_ARGS__);\
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_s.c_str());        \
        } catch (const fmt::format_error&) {                                    \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, "[fmt error] " fmt_);  \
        }                                                                        \
    } while(0)

// LOC 版：带文件名+行号+函数名，定位问题时用
#define _YK_LOG_DEV_LOC(lv_, fmt_, ...)                                                                        \
    do {                                                                                                       \
        if ((lv_) < YK::Logger::instance().level()) break;                                                    \
        try {                                                                                                  \
            auto _yk_s = fmt::format("[{}|{}:{}] " fmt_, __FUNCTION__, _YK_FILENAME, __LINE__, ##__VA_ARGS__);\
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_s.c_str());                                      \
        } catch (const fmt::format_error&) {                                                                  \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, "[fmt error] " fmt_);                                \
        }                                                                                                      \
    } while(0)

// 普通版
#define YK_LOGT(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

// LOC 版：需要定位问题时用
#define YK_LOGT_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#endif // __CUDACC__