#pragma once
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>
namespace yk::spectral::reconstruction_resources {
inline bool fitsReconstructionBudget(size_t required, const char* stage)
{
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
        return false;
    const size_t reserve = std::max(size_t(512) * 1024 * 1024, total / 5);
    if (free >= reserve && required <= free - reserve)
        return true;
    std::cerr << "[reconstruction memory] " << stage << " requires_bytes=" << required
              << " free_bytes=" << free << " render_reserve_bytes=" << reserve
              << "; reduce projection/subset/grid sizes" << std::endl;
    return false;
}
inline int iterativeBatchViews(size_t pixels, int views)
{
    return int(std::max(size_t(1), std::min(size_t(views), std::min(size_t(64), size_t(256) * 1024 * 1024 / (pixels * sizeof(float))))));
}
}
