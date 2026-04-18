#pragma once
// ============================================================
//  yk_log.h  —  .cpp 编译单元日志
//  .cu 文件请用 YkMacro.hpp 中的 YK_LOGI 等宏
// ============================================================

#include <cstdio>

namespace YK {

    enum class LogLevel {
        Trace = 0, Debug, Info, Warn, Error, Critical, Off
    };

    class Logger {
    public:
        static Logger& instance();
        void     set_level(LogLevel lv);
        LogLevel level() const;
        void     log(LogLevel lv, const char* tag, const char* msg);
    private:
        Logger() = default;
        LogLevel min_level_{ LogLevel::Trace };
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
#include <spdlog/fmt/fmt.h>

#ifndef YK_LOG_TAG
#  define YK_LOG_TAG "YK"
#endif

#define _YK_LOG_DEV(lv_, fmt_, ...)                                              \
    do {                                                                         \
        if ((lv_) < YK::Logger::instance().level()) break;                      \
        try {                                                                    \
            auto _yk_s = fmt::format(fmt_, ##__VA_ARGS__);                      \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_s.c_str());        \
        } catch (const fmt::format_error&) {                                    \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, "[fmt error] " fmt_);  \
        }                                                                        \
    } while(0)

#define _YK_LOG_DEV_LOC(lv_, fmt_, ...)                                          \
    _YK_LOG_DEV(lv_, "[{}:{}] " fmt_, __FILE__, __LINE__, ##__VA_ARGS__)

#define YK_LOGT(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#define YK_LOGE_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#endif // __CUDACC__