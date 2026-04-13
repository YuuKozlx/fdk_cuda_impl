// yk_log.h
#pragma once
#include <cstdio>
#include <vector>

namespace YK {

    enum class LogLevel {
        Trace = 0, Debug, Info, Warn, Error, Critical, Off
    };

    class Logger {
    public:
        static Logger& instance();
        void set_level(LogLevel lv);
        LogLevel level() const;
        void log(LogLevel lv, const char* tag, const char* msg);

    private:
        Logger() = default;
        LogLevel min_level_{ LogLevel::Trace };
    };

} // namespace YK

// ---- ºê ----
#ifndef YK_LOG_TAG
#  define YK_LOG_TAG "YK"
#endif

#define YK_LOG_IMPL(lv_, fmt_, ...)                                          \
    do {                                                                     \
        if ((lv_) < YK::Logger::instance().level()) break;                  \
        char _yk_stk[512];                                                   \
        int _yk_n = std::snprintf(_yk_stk, sizeof(_yk_stk),                 \
                                  fmt_, ##__VA_ARGS__);                      \
        if (_yk_n < 0) break;                                                \
        if (_yk_n < (int)sizeof(_yk_stk)) {                                  \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_stk);           \
        } else {                                                             \
            std::vector<char> _yk_heap(_yk_n + 1);                          \
            std::snprintf(_yk_heap.data(), _yk_heap.size(),                  \
                          fmt_, ##__VA_ARGS__);                              \
            YK::Logger::instance().log(lv_, YK_LOG_TAG, _yk_heap.data());   \
        }                                                                    \
    } while(0)

#define YK_LOGT(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Trace,    fmt, ##__VA_ARGS__)
#define YK_LOGD(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Debug,    fmt, ##__VA_ARGS__)
#define YK_LOGI(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Info,     fmt, ##__VA_ARGS__)
#define YK_LOGW(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Warn,     fmt, ##__VA_ARGS__)
#define YK_LOGE(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Error,    fmt, ##__VA_ARGS__)
#define YK_LOGC(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Critical, fmt, ##__VA_ARGS__)

#define YK_LOGE_LOC(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Error,    \
    "[%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)
#define YK_LOGC_LOC(fmt, ...) YK_LOG_IMPL(YK::LogLevel::Critical, \
    "[%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)