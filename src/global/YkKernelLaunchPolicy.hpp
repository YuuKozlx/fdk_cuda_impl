#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

#include "global/YkMacro.hpp"

namespace YK {

// WarpStrideCtx kernel 必须使用一维且 warp 对齐的 block。
inline void validateWarpLaunch(dim3 block, dim3 grid)
{
    assert(block.x >= 32 && block.x <= 1024 && (block.x % 32) == 0);
    assert(block.y == 1 && block.z == 1);
    assert(grid.x > 0 && grid.y > 0 && grid.z > 0);
}

struct SKernelLaunchPolicy {
    int block_threads = 256;
    bool bounds_check = true;
    int blocks_per_sm_1d = 4;
    int blocks_per_sm_rowwarp = 2;

    struct SDeviceLimits {
        int sm_count = 1;
        int max_grid_x = 1;
        int max_grid_y = 1;
        int max_grid_z = 1;
    };

    struct SLaunch1D {
        dim3 block = dim3(1, 1, 1);
        dim3 grid = dim3(1, 1, 1);
    };

    int normalizedBlockThreads() const
    {
        int threads = std::clamp(block_threads, 32, 1024);
        threads = ((threads + 31) / 32) * 32;
        return std::min(threads, 1024);
    }

    SDeviceLimits deviceLimits(int device = -1) const
    {
        if (device < 0) YK_CUDA_CHECK(cudaGetDevice(&device));
        SDeviceLimits limits;
        YK_CUDA_CHECK(cudaDeviceGetAttribute(&limits.sm_count,
            cudaDevAttrMultiProcessorCount, device));
        YK_CUDA_CHECK(cudaDeviceGetAttribute(&limits.max_grid_x,
            cudaDevAttrMaxGridDimX, device));
        YK_CUDA_CHECK(cudaDeviceGetAttribute(&limits.max_grid_y,
            cudaDevAttrMaxGridDimY, device));
        YK_CUDA_CHECK(cudaDeviceGetAttribute(&limits.max_grid_z,
            cudaDevAttrMaxGridDimZ, device));
        limits.sm_count = std::max(limits.sm_count, 1);
        limits.max_grid_x = std::max(limits.max_grid_x, 1);
        limits.max_grid_y = std::max(limits.max_grid_y, 1);
        limits.max_grid_z = std::max(limits.max_grid_z, 1);
        return limits;
    }

    // 对应 kernel 必须使用 grid-stride 循环覆盖 n。
    SLaunch1D make1D(std::size_t n, int device = -1) const
    {
        if (n == 0) return {};
        const auto limits = deviceLimits(device);
        const int threads = normalizedBlockThreads();
        const std::size_t needed = (n + threads - 1) / threads;
        const std::size_t cap = std::min(
            static_cast<std::size_t>(limits.max_grid_x),
            static_cast<std::size_t>(limits.sm_count) *
                static_cast<std::size_t>(std::max(blocks_per_sm_1d, 1)));
        return {dim3(static_cast<unsigned>(threads), 1, 1),
            dim3(static_cast<unsigned>(std::min(needed, cap)), 1, 1)};
    }

    // 对应 kernel 必须按全局 warp 编号执行 row-stride 循环。
    SLaunch1D makeRowWarp(std::size_t rows, int device = -1) const
    {
        if (rows == 0) return {};
        const auto limits = deviceLimits(device);
        const int threads = normalizedBlockThreads();
        const int warps_per_block = threads / 32;
        const std::size_t needed = (rows + warps_per_block - 1) /
            warps_per_block;
        const std::size_t cap = std::min(
            static_cast<std::size_t>(limits.max_grid_x),
            static_cast<std::size_t>(limits.sm_count) *
                static_cast<std::size_t>(std::max(blocks_per_sm_rowwarp, 1)));
        return {dim3(static_cast<unsigned>(threads), 1, 1),
            dim3(static_cast<unsigned>(std::min(needed, cap)), 1, 1)};
    }

    int maxAngleChunk(int device = -1) const
    {
        return deviceLimits(device).max_grid_z;
    }

    template<typename KernelFunc>
    static SKernelLaunchPolicy fromKernel(KernelFunc* kernel,
        std::size_t dynamic_smem = 0, bool bounds_check = true)
    {
        int block_size = 0;
        int min_grid_size = 0;
        YK_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size,
            &block_size, kernel, dynamic_smem, 0));
        block_size = std::max((block_size / 32) * 32, 32);
        SKernelLaunchPolicy policy;
        policy.block_threads = block_size;
        policy.bounds_check = bounds_check;
        return policy;
    }
};

} // namespace YK
