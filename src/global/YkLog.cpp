#include "YkLog.h"

#include <cstdio>
#include <mutex>
#include <vector>

#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace YK {
namespace {

spdlog::level::level_enum toSpdlogLevel(LogLevel level)
{
    switch (level) {
    case LogLevel::Trace:    return spdlog::level::trace;
    case LogLevel::Debug:    return spdlog::level::debug;
    case LogLevel::Info:     return spdlog::level::info;
    case LogLevel::Warn:     return spdlog::level::warn;
    case LogLevel::Error:    return spdlog::level::err;
    case LogLevel::Critical: return spdlog::level::critical;
    case LogLevel::Off:      return spdlog::level::off;
    }
    return spdlog::level::off;
}

constexpr const char* kPattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v";

} // namespace

struct Logger::Impl {
    std::mutex mutex;
    std::vector<spdlog::sink_ptr> spdlog_sinks;
    std::vector<LogSink> callbacks;
    std::shared_ptr<spdlog::logger> backend;

    void rebuildBackend(LogLevel level)
    {
        backend = std::make_shared<spdlog::logger>(
            "YK", spdlog_sinks.begin(), spdlog_sinks.end());
        backend->set_pattern(kPattern);
        backend->set_level(toSpdlogLevel(level));
        backend->flush_on(spdlog::level::err);
    }
};

Logger::Logger() : impl_(std::make_unique<Impl>())
{
    impl_->spdlog_sinks.push_back(
        std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    impl_->rebuildBackend(min_level_.load(std::memory_order_relaxed));
}

Logger::~Logger() = default;

Logger& Logger::instance()
{
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level)
{
    min_level_.store(level, std::memory_order_relaxed);
    std::shared_ptr<spdlog::logger> backend;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        backend = impl_->backend;
    }
    backend->set_level(toSpdlogLevel(level));
}

LogLevel Logger::level() const
{
    return min_level_.load(std::memory_order_relaxed);
}

void Logger::add_sink(LogSink sink)
{
    if (!sink) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->callbacks.push_back(std::move(sink));
}

void Logger::clear_sinks()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->spdlog_sinks.clear();
    impl_->callbacks.clear();
    impl_->rebuildBackend(level());
}

void Logger::add_file_sink(const std::string& path)
{
    try {
        auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, false);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->spdlog_sinks.push_back(std::move(sink));
        impl_->rebuildBackend(level());
    } catch (const spdlog::spdlog_ex& error) {
        std::fprintf(stderr, "[YK] failed to open log file '%s': %s\n",
            path.c_str(), error.what());
    }
}

void Logger::log(LogLevel level_value, const char* tag, const char* message)
{
    if (level_value < level() || level_value == LogLevel::Off) return;

    std::shared_ptr<spdlog::logger> backend;
    std::vector<LogSink> callbacks;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        backend = impl_->backend;
        callbacks = impl_->callbacks;
    }

    const char* safe_tag = tag ? tag : "YK";
    const char* safe_message = message ? message : "";
    try {
        backend->log(toSpdlogLevel(level_value), "[{}] {}", safe_tag, safe_message);
    } catch (const spdlog::spdlog_ex&) {
        // Logging failures must not terminate reconstruction work.
    }

    for (auto& callback : callbacks) {
        try {
            callback(level_value, safe_tag, safe_message);
        } catch (...) {
            // User callbacks are isolated from the logging caller.
        }
    }
}

} // namespace YK
