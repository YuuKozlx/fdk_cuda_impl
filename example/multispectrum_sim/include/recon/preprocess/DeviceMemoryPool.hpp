#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <vector>
namespace yk::spectral::preprocess {
struct PoolStatistics {
    size_t allocations = 0, reuses = 0, active_bytes = 0, cached_bytes = 0;
};
// Leaves max(512 MiB, 20% VRAM) free (capped at 50% on small GPUs),
// limits pooled storage to 60% VRAM, and returned cache to 256 MiB.
// Process-wide pool, separated by CUDA device, synchronized for multiple workers.
// Buffers use the current device and default stream. release records completion;
// the next borrower waits on that event before using the storage.
cudaError_t poolAllocate(void **pointer, size_t bytes) noexcept;
template <class T> cudaError_t poolAllocate(T **pointer, size_t bytes) noexcept {
    return poolAllocate(reinterpret_cast<void **>(pointer), bytes);
}
cudaError_t poolRelease(void *pointer) noexcept;
// Reserve all listed blocks simultaneously; repeat calls reuse cached storage.
void initializeDevicePool(const std::vector<size_t> &bytes);
PoolStatistics devicePoolStatistics();
// Free only returned buffers on the current device. Active leases remain valid.
void trimDevicePool();
} // namespace yk::spectral::preprocess
