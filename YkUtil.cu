#include "YkUtil.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace YK {

    namespace Util {
        // ============================================================
// internal helper kernel
// ============================================================

        __global__ void kernel_crop_u_2d(
            const float* __restrict__ src,
            float* __restrict__ dst,
            int Nu,
            int Nv,
            int paddedN,
            int start_u)
        {
            int u = blockIdx.x * blockDim.x + threadIdx.x;
            int v = blockIdx.y * blockDim.y + threadIdx.y;
            if (u >= Nu || v >= Nv) return;

            int su = start_u + u;
            float val = 0.0f;
            if ((unsigned)su < (unsigned)paddedN) {
                val = src[v * paddedN + su];
            }
            dst[v * Nu + u] = val;
        }

        // ============================================================
        // exported host wrapper
        // ============================================================

        void crop_u_2d(
            const float* src,
            float* dst,
            int Nu,
            int Nv,
            int paddedN,
            int start_u,
            cudaStream_t stream)
        {
            constexpr int BX = 32;
            constexpr int BY = 8;

            dim3 block(BX, BY);
            dim3 grid((Nu + BX - 1) / BX,
                (Nv + BY - 1) / BY);

            kernel_crop_u_2d << <grid, block, 0, stream >> > (
                src, dst, Nu, Nv, paddedN, start_u);
        }
    };
};