// YkAlignPadCropVec.hpp
#pragma once
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector_types.h>
#include "YkGlobals.h"

namespace YK {

    // ============================================================
    // pad: [Nv*Nu] -> [Nv*paddedN] using start_u
    // ============================================================
    __global__ void _kernel_pad_startu(
        const float* __restrict__ src,  // [Nv*Nu]
        float* __restrict__ dst,        // [Nv*paddedN]
        int Nu, int Nv, int paddedN,
        int start_u)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int v = blockIdx.y;
        if (u >= paddedN || v >= Nv) return;

        int su = u - start_u;
        float val = 0.0f;
        if ((unsigned)su < (unsigned)Nu) val = src[v * Nu + su];
        dst[v * paddedN + u] = val;
    }

    // ============================================================
    // crop: [Nv*paddedN] -> [Nv*Nu] using start_u
    // ============================================================
    __global__ void _kernel_crop_startu(
        const float* __restrict__ src,  // [Nv*paddedN]
        float* __restrict__ dst,        // [Nv*Nu]
        int Nu, int Nv, int paddedN,
        int start_u)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int v = blockIdx.y;
        if (u >= Nu || v >= Nv) return;

        int su = start_u + u;
        float val = 0.0f;
        if ((unsigned)su < (unsigned)paddedN) val = src[v * paddedN + su];
        dst[v * Nu + u] = val;
    }

    // ============================================================
    // AlignPadCropManagerVec
    //  - owns paddedN policy (nextPow2(2*Nu))
    //  - computes start_u from offsetU_pix
    //  - pad -> (filter in between) -> crop (using same start_u)
    // ============================================================
    class AlignPadCropManagerVec {
    public:
        bool init(int Nu, int Nv, cudaStream_t stream = 0)
        {
            Nu_ = Nu;
            Nv_ = Nv;
            stream_ = stream;

            paddedN_ = computePaddedN_(Nu_);
            inited_ = (Nu_ > 0 && Nv_ > 0 && paddedN_ >= Nu_);

            last_startu_ = 0;
            last_offsetU_ = 0.0f;
            last_offsetV_ = 0.0f;
            return inited_;
        }

        void setStream(cudaStream_t s) { stream_ = s; }

        // ---- query ----
        int Nu() const { return Nu_; }
        int Nv() const { return Nv_; }
        int paddedN() const { return paddedN_; }

        int lastStartU() const { return last_startu_; }
        float lastOffsetU() const { return last_offsetU_; }
        float lastOffsetV() const { return last_offsetV_; }

        // ------------------------------------------------------------
        // (1): offsetU_pix -> start_u -> pad
        // offsetU_pix: central-ray hit offset in pixel (u-axis)
        // ------------------------------------------------------------
        bool pad(const float* d_in_view,   // [Nv*Nu]
                 float* d_padded,          // [Nv*paddedN]
                 float offsetU_pix)
        {
            if (!inited_) return false;
            last_offsetU_ = offsetU_pix;

            // align padded center to detector axis index
            // axis_idx = (Nu-1)/2 + offsetU_pix
            const float axis_idx = (Nu_ - 1) * 0.5f + offsetU_pix;
            last_startu_ = (int)lrintf(paddedN_ * 0.5f - axis_idx);

            dim3 block(256, 1);
            dim3 grid((paddedN_ + block.x - 1) / block.x, Nv_);
            _kernel_pad_startu << <grid, block, 0, stream_ >> > (
                d_in_view, d_padded, Nu_, Nv_, paddedN_, last_startu_);
            YK_CUDA_KERNEL_CHECK();
            return true;
        }



        // ------------------------------------------------------------
        // (2): crop back using same start_u
        // NOTE: call after padFromOffset* for current view
        // ------------------------------------------------------------
        void crop(
            const float* d_padded, // [Nv*paddedN]
            float* d_out_view      // [Nv*Nu]
        ) const
        {
            dim3 block(256, 1);
            dim3 grid((Nu_ + block.x - 1) / block.x, Nv_);
            _kernel_crop_startu << <grid, block, 0, stream_ >> > (
                d_padded, d_out_view, Nu_, Nv_, paddedN_, last_startu_);
            YK_CUDA_KERNEL_CHECK();
        }

    private:
        static int computePaddedN_(int Nu)
        {
            // policy: nextPow2(2*Nu)
            int need = 2 * Nu;
            int n = 1;
            while (n < need) n <<= 1;
            return n;
        }

    private:
        int Nu_ = 0, Nv_ = 0, paddedN_ = 0;
        cudaStream_t stream_ = 0;
        bool inited_ = false;

        float last_offsetU_ = 0.0f, last_offsetV_ = 0.0f;
        int last_startu_ = 0;
    };

} // namespace YK
