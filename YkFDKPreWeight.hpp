#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "YkGlobals.h"   // 你的 YK_CUDA_CHECK / YK_KERNEL_CHECK

namespace YK {

    // power=1 -> cos；power=2 -> cos^2（可选）
    __global__ void _kernel_preweight_cosine(
        const float* __restrict__ src,
        float* __restrict__ dst,
        int Nu, int Nv, int batch,
        float du, float dv,
        float offsetU, float offsetV,   // 单位：pixel（像素偏移）
        float DSD,                      // 源到探测器距离（mm）
        int power                        // 1 or 2
    ) {
        int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int v_idx = blockIdx.y * blockDim.y + threadIdx.y;
        int b = blockIdx.z;

        if (u_idx >= Nu || v_idx >= Nv || b >= batch) return;

        // 探测器中心（带 offset）
        float u0 = (Nu - 1) * 0.5f + offsetU;
        float v0 = (Nv - 1) * 0.5f + offsetV;

        float u = (u_idx - u0) * du;  // mm
        float v = (v_idx - v0) * dv;  // mm

        float L2 = DSD * DSD + u * u + v * v;
        float invL = rsqrtf(L2);
        float w = DSD * invL;         // cos = DSD / sqrt(DSD^2 + u^2 + v^2)

        if (power == 2) w = w * w;

        // layout: [b][v][u]
        int idx = (b * Nv + v_idx) * Nu + u_idx;
        dst[idx] = src[idx] * w;
    }

    class PreweightManager {
    public:
        bool init(int Nu, int Nv, int batch,
            float du, float dv,
            float DSD,
            float offsetU = 0.0f, float offsetV = 0.0f,
            int power = 1,
            cudaStream_t stream = 0)
        {
            Nu_ = Nu; Nv_ = Nv; batch_ = batch;
            du_ = du; dv_ = dv;
            DSD_ = DSD;
            offsetU_ = offsetU; offsetV_ = offsetV;
            power_ = power;
            stream_ = stream;
            inited_ = true;
            return true;
        }

        void setStream(cudaStream_t s) { stream_ = s; }

        // src/dst 可以相同（就地写）
        void apply(const float* d_src, float* d_dst) const {
            if (!inited_) return;

            dim3 block(16, 16, 1);
            dim3 grid((Nu_ + block.x - 1) / block.x,
                (Nv_ + block.y - 1) / block.y,
                batch_);

            _kernel_preweight_cosine << <grid, block, 0, stream_ >> > (
                d_src, d_dst,
                Nu_, Nv_, batch_,
                du_, dv_,
                offsetU_, offsetV_,
                DSD_, power_);

            YK_CUDA_KERNEL_CHECK();
        }

    private:
        int Nu_ = 0, Nv_ = 0, batch_ = 0;
        float du_ = 0, dv_ = 0;
        float DSD_ = 0;
        float offsetU_ = 0, offsetV_ = 0;
        int power_ = 1;
        cudaStream_t stream_ = 0;
        bool inited_ = false;
    };

    // 统一风格 helper
    inline void executeFdkPreweight(PreweightManager& pw,
        const float* d_in, float* d_out,
        cudaStream_t stream = 0)
    {
        pw.setStream(stream);
        pw.apply(d_in, d_out);
    }

} // namespace YK
#pragma once
