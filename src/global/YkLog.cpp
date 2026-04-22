#include "YkLog.h"

#define FMT_UNICODE 0
#define FMT_HEADER_ONLY
#include "util/fmt/format.h"
#include "util/fmt/color.h"
#include "util/fmt/ostream.h"

#include <chrono>
#include <ctime>
#include <mutex>
#include <fstream>

namespace YK {

    // ---- 必须在 console_sink 之前定义 ----
    static const char* level_str(LogLevel lv) {
        switch (lv) {
        case LogLevel::Trace:    return "TRACE";
        case LogLevel::Debug:    return "DEBUG";
        case LogLevel::Info:     return "INFO ";
        case LogLevel::Warn:     return "WARN ";
        case LogLevel::Error:    return "ERROR";
        case LogLevel::Critical: return "CRIT ";
        default:                 return "OFF  ";
        }
    }

    static fmt::color level_color(LogLevel lv) {
        switch (lv) {
        case LogLevel::Trace:    return fmt::color::gray;
        case LogLevel::Debug:    return fmt::color::cyan;
        case LogLevel::Info:     return fmt::color::green;
        case LogLevel::Warn:     return fmt::color::yellow;
        case LogLevel::Error:    return fmt::color::red;
        case LogLevel::Critical: return fmt::color::magenta;
        default:                 return fmt::color::white;
        }
    }

    static void console_sink(LogLevel lv, const char* tag, const char* msg) {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        auto ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count() % 1000
            );
        std::tm tm{};
        localtime_s(&tm, &t);

        int year = tm.tm_year + 1900;
        int mon = tm.tm_mon + 1;
        int mday = tm.tm_mday;
        int hour = tm.tm_hour;
        int min = tm.tm_min;
        int sec = tm.tm_sec;

        fmt::print(stderr,
            "[{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:03d}] [{}] [{}] {}\n",
            year, mon, mday, hour, min, sec, ms,
            fmt::styled(level_str(lv), fmt::fg(level_color(lv))),
            tag, msg
        );
    }

    // 在 console_sink 下方加
    static LogSink make_file_sink(const std::string& path) {
        auto file = std::make_shared<std::ofstream>(path, std::ios::app);
        if (!file->is_open()) {
            fmt::print(stderr, "[YK] failed to open log file: {}\n", path);
            return nullptr;
        }
        return [file](LogLevel lv, const char* tag, const char* msg) {
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            auto ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch()).count() % 1000
                );
            std::tm tm{};
            localtime_s(&tm, &t);

            int year = tm.tm_year + 1900;
            int mon = tm.tm_mon + 1;
            int mday = tm.tm_mday;
            int hour = tm.tm_hour;
            int min = tm.tm_min;
            int sec = tm.tm_sec;

            fmt::print(*file,                    // ← 写到文件，不是 stderr
                "[{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:03d}] [{}] [{}] {}\n",
                year, mon, mday, hour, min, sec, ms,
                level_str(lv),                   // ← 文件不需要彩色，去掉 styled
                tag, msg
            );
            file->flush();
            };
    }


    Logger::Logger() {
        sinks_.push_back(console_sink);
    }

    Logger& Logger::instance() {
        static Logger s;
        return s;
    }

    void Logger::add_sink(LogSink sink) {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.push_back(std::move(sink));
    }

    void Logger::clear_sinks() {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.clear();
    }


    void Logger::add_file_sink(const std::string& path) {
        auto sink = make_file_sink(path);
        if (sink) add_sink(std::move(sink));
    }


    void Logger::set_level(LogLevel lv) {
        min_level_ = lv;
    }

    LogLevel Logger::level() const {
        return min_level_;
    }

    void Logger::log(LogLevel lv, const char* tag, const char* msg) {
        if (lv < min_level_ || lv == LogLevel::Off) return;
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& sink : sinks_) {
            sink(lv, tag, msg);
        }
    }

} // namespace YK