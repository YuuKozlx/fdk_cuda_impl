// yk_log.cpp
#include "YkLog.h"

#ifndef FMT_UNICODE
#  define FMT_UNICODE 0
#endif
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <spdlog/common.h>

namespace YK {

    // --------------------------------------------------------
    //  spdlog 实例，只在这个编译单元可见
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    //  LogLevel <-> spdlog::level 映射
    // --------------------------------------------------------
    static spdlog::level::level_enum to_spdlog_level(LogLevel lv) {
        switch (lv) {
        case LogLevel::Trace:    return spdlog::level::trace;
        case LogLevel::Debug:    return spdlog::level::debug;
        case LogLevel::Info:     return spdlog::level::info;
        case LogLevel::Warn:     return spdlog::level::warn;
        case LogLevel::Error:    return spdlog::level::err;
        case LogLevel::Critical: return spdlog::level::critical;
        default:                 return spdlog::level::off;
        }
    }

    // --------------------------------------------------------
    //  Logger 实现
    // --------------------------------------------------------
    Logger& Logger::instance() {
        static Logger s;
        return s;
    }

    void Logger::set_level(LogLevel lv) {
        min_level_ = lv;
        dev_logger().set_level(to_spdlog_level(lv));  // 两边同步
    }

    LogLevel Logger::level() const {
        return min_level_;
    }

    void Logger::log(LogLevel lv, const char* tag, const char* msg) {
        if (lv < min_level_ || lv == LogLevel::Off) return;
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

} // namespace YK