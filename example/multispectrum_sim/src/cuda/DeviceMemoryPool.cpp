#include "common/YkTimingWatchdog.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include <cuda_runtime.h>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <stdexcept>
namespace yk::spectral::cuda_resources {
namespace {
constexpr size_t cache_limit = 256ull * 1024 * 1024;
struct Block {
    size_t bytes;
    int device;
    bool active;
    cudaEvent_t ready;
};
struct Pool {
    std::mutex mutex;
    std::unordered_map<void *, Block> blocks;
    std::unordered_map<int, PoolStatistics> statistics;
    ~Pool() {
        for (auto &pair : blocks) {
            cudaSetDevice(pair.second.device);
            cudaEventDestroy(pair.second.ready);
            cudaFree(pair.first);
        }
    }
};
Pool &pool() {
    static Pool value;
    return value;
}
void check(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}
struct DeviceScope {
    int previous = -1;
    cudaError_t status = cudaSuccess;
    explicit DeviceScope(int target) {
        status = cudaGetDevice(&previous);
        if (status == cudaSuccess && previous != target)
            status = cudaSetDevice(target);
    }
    ~DeviceScope() {
        if (previous >= 0)
            cudaSetDevice(previous);
    }
};
void trim(Pool &p, int device) {
    auto &stats = p.statistics[device];
    for (auto it = p.blocks.begin(); it != p.blocks.end();) {
        if (it->second.device == device && !it->second.active) {
            // cudaFree waits for outstanding use, including the recorded event.
            check(cudaEventSynchronize(it->second.ready));
            check(cudaFree(it->first));
            cudaEventDestroy(it->second.ready);
            stats.cached_bytes -= it->second.bytes;
            it = p.blocks.erase(it);
        } else
            ++it;
    }
}
} // namespace
cudaError_t poolAllocate(void **pointer, size_t bytes) noexcept {
    if (!pointer || !bytes)
        return cudaErrorInvalidValue;
    *pointer = nullptr;
    try {
        YK::diagnostics::TimingSpan watch("gpu.pool.allocate","bytes="+std::to_string(bytes));
        int device;
        auto code = cudaGetDevice(&device);
        if (code != cudaSuccess)
            return code;
        auto &p = pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        auto &stats = p.statistics[device];
        auto chosen = p.blocks.end();
        size_t best = std::numeric_limits<size_t>::max();
        for (auto it = p.blocks.begin(); it != p.blocks.end(); ++it) {
            const auto &b = it->second;
            if (b.device == device && !b.active && b.bytes >= bytes && b.bytes < best) {
                chosen = it;
                best = b.bytes;
            }
        }
        if (chosen != p.blocks.end()) {
            code = cudaStreamWaitEvent(nullptr, chosen->second.ready, 0);
            if (code != cudaSuccess)
                return code;
            chosen->second.active = true;
            stats.cached_bytes -= best;
            stats.active_bytes += best;
            ++stats.reuses;
            *pointer = chosen->first;
            return cudaSuccess;
        }
        size_t free_bytes = 0, total_bytes = 0;
        code = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (code != cudaSuccess)
            return code;
        // Do not preallocate the card. Leave room for VTK textures/volume rendering
        // and CUDA workspaces outside this pool, even when a job grows in size.
        const size_t reserve =
            std::min(total_bytes / 2, std::max(size_t(512ull * 1024 * 1024), total_bytes / 5));
        const size_t budget = total_bytes / 5 * 3;
        auto fits = [&]() {
            return bytes <= free_bytes && free_bytes - bytes >= reserve &&
                   stats.active_bytes + stats.cached_bytes <= budget &&
                   bytes <= budget - stats.active_bytes - stats.cached_bytes;
        };
        if (!fits()) {
            trim(p, device);
            code = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (code != cudaSuccess)
                return code;
        }
        if (!fits())
            return cudaErrorMemoryAllocation;
        void *storage = nullptr;
        code = cudaMalloc(&storage, bytes);
        if (code == cudaErrorMemoryAllocation) {
            trim(p, device);
            code = cudaMemGetInfo(&free_bytes,&total_bytes);
            if(code!=cudaSuccess)return code;
            if(!fits())return cudaErrorMemoryAllocation;
            code = cudaMalloc(&storage, bytes);
        }
        if (code != cudaSuccess)
            return code;
        cudaEvent_t ready = nullptr;
        code = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (code != cudaSuccess) {
            cudaFree(storage);
            return code;
        }
        try {
            p.blocks.emplace(storage, Block{bytes, device, true, ready});
        } catch (...) {
            cudaEventDestroy(ready);
            cudaFree(storage);
            throw;
        }
        ++stats.allocations;
        stats.active_bytes += bytes;
        *pointer = storage;
        return cudaSuccess;
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }
}
cudaError_t poolRelease(void *pointer) noexcept {
    if (!pointer)
        return cudaSuccess;
    try {
        auto &p = pool();
        std::lock_guard<std::mutex> lock(p.mutex);
        auto it = p.blocks.find(pointer);
        if (it == p.blocks.end() || !it->second.active)
            return cudaErrorInvalidValue;
        auto &b = it->second;
        DeviceScope device(b.device);
        if (device.status != cudaSuccess)
            return device.status;
        auto &stats = p.statistics[b.device];
        auto code = cudaEventRecord(b.ready, nullptr);
        if (code != cudaSuccess || b.bytes > cache_limit ||
            stats.cached_bytes > cache_limit - b.bytes) {
            auto freed = cudaFree(pointer);
            if (freed != cudaSuccess)
                return freed;
            cudaEventDestroy(b.ready);
            stats.active_bytes -= b.bytes;
            p.blocks.erase(it);
            return code;
        }
        b.active = false;
        stats.active_bytes -= b.bytes;
        stats.cached_bytes += b.bytes;
        return cudaSuccess;
    } catch (...) {
        return cudaErrorUnknown;
    }
}
void initializeDevicePool(const std::vector<size_t> &sizes) {
    std::vector<void *> leases;
    try {
        for (size_t size : sizes) {
            void *pointer = nullptr;
            check(poolAllocate(&pointer, size));
            try {
                leases.push_back(pointer);
            } catch (...) {
                poolRelease(pointer);
                throw;
            }
        }
    } catch (...) {
        for (auto pointer : leases)
            poolRelease(pointer);
        throw;
    }
    for (auto pointer : leases)
        check(poolRelease(pointer));
}
PoolStatistics devicePoolStatistics() {
    int device;
    check(cudaGetDevice(&device));
    auto &p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    return p.statistics[device];
}
void trimDevicePool() {
    YK::diagnostics::TimingSpan watch("gpu.pool.trim");
    int device;
    check(cudaGetDevice(&device));
    auto &p = pool();
    std::lock_guard<std::mutex> lock(p.mutex);
    trim(p, device);
}
} // namespace yk::spectral::cuda_resources
