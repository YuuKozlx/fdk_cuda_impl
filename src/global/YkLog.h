#pragma once
// ============================================================
// YKCBCT logging facade. C++ host code uses spdlog with its bundled fmt.
// CUDA device code continues to use YkMacro.hpp and device printf.
// ============================================================

#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <string>

namespace YK {

enum class LogLevel {
    Trace = 0, Debug, Info, Warn, Error, Critical, Off
};

using LogSink = std::function<void(LogLevel, const char* tag, const char* msg)>;

class Logger {
public:
    static Logger& instance();
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void set_level(LogLevel lv);
    LogLevel level() const;
    void log(LogLevel lv, const char* tag, const char* msg);

    void add_sink(LogSink sink);
    void clear_sinks();
    void add_file_sink(const std::string& path);

private:
    Logger();

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<LogLevel> min_level_{LogLevel::Trace};
};

} // namespace YK

#ifndef __CUDACC__

#ifndef FMT_UNICODE
#  define FMT_UNICODE 0
#endif
#include <spdlog/fmt/fmt.h>

#ifndef YK_LOG_TAG
#  define YK_LOG_TAG "YK"
#endif

#define _YK_FILENAME (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : \
                     (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__))

#define _YK_LOG_DEV(lv_, fmt_, ...)                                              \
    do {                                                                         \
        if ((lv_) < YK::Logger::instance().level()) break;                       \
        try {                                                                    \
            auto _yk_s = fmt::format("[{}] " fmt_, __FUNCTION__, ##__VA_ARGS__); \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_s.c_str());          \
        } catch (const fmt::format_error&) {                                     \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, "[fmt error] " fmt_);    \
        }                                                                        \
    } while (0)

#define _YK_LOG_DEV_LOC(lv_, fmt_, ...)                                          \
    do {                                                                         \
        if ((lv_) < YK::Logger::instance().level()) break;                       \
        try {                                                                    \
            auto _yk_s = fmt::format("[{}|{}:{}] " fmt_, __FUNCTION__,          \
                _YK_FILENAME, __LINE__, ##__VA_ARGS__);                          \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_s.c_str());          \
        } catch (const fmt::format_error&) {                                     \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, "[fmt error] " fmt_);    \
        }                                                                        \
    } while (0)

#define YK_LOGT(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC(fmt, ...) _YK_LOG_DEV(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#define YK_LOGT_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC_LOC(fmt, ...) _YK_LOG_DEV_LOC(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#endif // __CUDACC__
