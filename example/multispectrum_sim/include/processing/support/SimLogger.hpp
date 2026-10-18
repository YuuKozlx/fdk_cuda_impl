#pragma once

#include <memory>
#include <string>

namespace spdlog { class logger; }

namespace yk::spectral {

class SimLogger {
public:
    static SimLogger& instance();
    void info(const std::string& message);

private:
    SimLogger();
    std::shared_ptr<spdlog::logger> logger_;
};

}
