#include "processing/support/SimLogger.hpp"

#define SPDLOG_HEADER_ONLY
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace yk::spectral {

SimLogger::SimLogger()
{
    logger_ = spdlog::stdout_color_mt("multispectrum_sim");
    logger_->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [multispectrum_sim] %v");
    logger_->set_level(spdlog::level::info);
}

SimLogger& SimLogger::instance()
{
    static SimLogger logger;
    return logger;
}

void SimLogger::info(const std::string& message)
{
    logger_->info(message);
    std::lock_guard<std::mutex> lock(mutex_);
    recent_.emplace_back(++sequence_,message.substr(0,4096));
    while(recent_.size()>1024)recent_.pop_front();
}
std::vector<SimLogger::Entry> SimLogger::readSince(std::uint64_t sequence) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Entry> result;
    for(const auto& entry:recent_)if(entry.first>sequence)result.push_back(entry);
    return result;
}

}
