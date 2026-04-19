#pragma once
#include <chrono>
#include <cstdio>

namespace YK {
    namespace Util {

        struct CpuTimer {
            const char* tag;
            std::chrono::high_resolution_clock::time_point start;

            CpuTimer(const char* tag) : tag(tag)
            {
                start = std::chrono::high_resolution_clock::now();
            }
            ~CpuTimer()
            {
                auto end = std::chrono::high_resolution_clock::now();
                const float ms =
                    std::chrono::duration<float, std::milli>(end - start).count();
                printf("[%s] %.3f ms (CPU)\n", tag, ms);
            }
            CpuTimer(const CpuTimer&) = delete;
            CpuTimer& operator=(const CpuTimer&) = delete;
        };

        struct ScopeLogger {
            const char* tag;
            ScopeLogger(const char* tag) : tag(tag) {
                printf("[%s] enter\n", tag);
            }
            ~ScopeLogger() {
                printf("[%s] exit\n", tag);
            }
            ScopeLogger(const ScopeLogger&) = delete;
            ScopeLogger& operator=(const ScopeLogger&) = delete;
        };

    } // namespace Util
} // namespace YK