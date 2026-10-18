#pragma once

#include <memory>
#include <string>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>
#include <utility>

namespace spdlog { class logger; }

namespace yk::spectral {

class SimLogger {
public:
    static SimLogger& instance();
    void info(const std::string& message);
    using Entry=std::pair<std::uint64_t,std::string>;
    std::vector<Entry> readSince(std::uint64_t sequence) const;

private:
    SimLogger();
    mutable std::mutex mutex_;
    std::deque<Entry> recent_;
    std::uint64_t sequence_=0;
    std::shared_ptr<spdlog::logger> logger_;
};

}
