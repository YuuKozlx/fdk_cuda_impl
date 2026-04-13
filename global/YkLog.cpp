// yk_log.cpp  ← spdlog 只在这里，只被 MSVC 编译
#include "YkLog.h"

#ifndef YK_EXPORT_BUILD
#  define FMT_UNICODE 0
#  include <spdlog/spdlog.h>
#  include <spdlog/sinks/stdout_color_sinks.h>
#endif

namespace YK {

#ifndef YK_EXPORT_BUILD
    static spdlog::logger& dev_logger() {
        static const auto l = []() {
            auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            auto logger = std::make_shared<spdlog::logger>("YK", sink);
            logger->set_level(spdlog::level::trace);
            logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
            return logger;
            }();
        return *l;
    }

    static void dev_log(LogLevel lv, const char* tag, const char* msg) {
        auto& l = dev_logger();
        switch (lv) {
        case LogLevel::Trace:    l.trace("[{}] {}", tag, msg); break;
        case LogLevel::Debug:    l.debug("[{}] {}", tag, msg); break;
        case LogLevel::Info:     l.info("[{}] {}", tag, msg); break;
        case LogLevel::Warn:     l.warn("[{}] {}", tag, msg); break;
        case LogLevel::Error:    l.error("[{}] {}", tag, msg); break;
        case LogLevel::Critical: l.critical("[{}] {}", tag, msg); break;
        default: break;
        }
    }
#else
    static void fallback_log(LogLevel lv, const char* tag, const char* msg) {
        static const char* lv_str[] = { "T","D","I","W","E","C" };
        FILE* out = (lv >= LogLevel::Warn) ? stderr : stdout;
        std::fprintf(out, "[YK][%s][%s] %s\n",
            lv_str[static_cast<int>(lv)], tag, msg);
    }
#endif

    Logger& Logger::instance() {
        static Logger s;
        return s;
    }

    void Logger::set_level(LogLevel lv) { min_level_ = lv; }
    LogLevel Logger::level() const { return min_level_; }

    void Logger::log(LogLevel lv, const char* tag, const char* msg) {
        if (lv < min_level_ || lv == LogLevel::Off) return;
#ifndef YK_EXPORT_BUILD
        dev_log(lv, tag, msg);
#else
        fallback_log(lv, tag, msg);
#endif
    }

} // namespace YK