#include "SimLogger.hpp"

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
}

}
