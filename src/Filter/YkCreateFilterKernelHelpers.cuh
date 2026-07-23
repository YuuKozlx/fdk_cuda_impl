#pragma once
#include <cuda_runtime.h>
#include <math.h>

#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"

namespace YK {
    namespace Filter {
        namespace detail {

            static __device__ __forceinline__ float sinc_pi(float x)
            {
                const float pi = CUDA_PI;
                float t = pi * x;
                if (fabsf(t) < 1e-8f) return 1.0f;
                return sinf(t) / t;
            }

            static __device__ __forceinline__ float bessel_i0(float x)
            {
                float ax = fabsf(x);
                if (ax < 3.75f) {
                    float y = x / 3.75f;
                    y *= y;
                    return 1.0f + y * (3.5156229f + y * (3.0899424f +
                        y * (1.2067492f + y * (0.2659732f +
                        y * (0.0360768f + y * 0.0045813f)))));
                }
                float y = 3.75f / ax;
                return (expf(ax) / sqrtf(ax)) *
                    (0.39894228f + y * (0.01328592f + y * (0.00225319f +
                    y * (-0.00157565f + y * (0.00916281f + y * (-0.02057706f +
                    y * (0.02635537f + y * (-0.01647633f + y * 0.00392377f))))))));
            }

            static __device__ __forceinline__ float window_shape(float x, EFilterKernel kind)
            {
                // x in [0, 1]
                const float pi = CUDA_PI;
                x = fminf(fmaxf(x, 0.0f), 1.0f);

                switch (kind) {
                    // None 在外部被处理
                case EFilterKernel::RamLak:
                    return 1.0f;

                case EFilterKernel::SheppLogan:
                    return sinc_pi(0.5f * x);

                case EFilterKernel::Cosine:
                    return cosf(0.5f * pi * x);

                case EFilterKernel::Hann:
                    return 0.5f * (1.0f + cosf(pi * x));

                case EFilterKernel::Hamming:
                    return 0.54f + 0.46f * cosf(pi * x);

                case EFilterKernel::Blackman: {
                    const float a0 = 0.42f;
                    const float a1 = 0.5f;
                    const float a2 = 0.08f;
                    return a0 + a1 * cosf(pi * x) + a2 * cosf(2.0f * pi * x);
                }

                case EFilterKernel::Butterworth: // 参数化版本在 window_shape_desc 中计算
                    return 1.0f;
                case EFilterKernel::Kaiser:
                case EFilterKernel::Tukey:
                    return 1.0f;

                default:
                    return 1.0f;
                }
            }

            static __device__ __forceinline__ float window_shape_desc(
                float x, const SFilterKernelDesc& desc)
            {
                const float pi = CUDA_PI;
                x = fminf(fmaxf(x, 0.0f), 1.0f);
                if (desc.kind == EFilterKernel::Butterworth) {
                    const float order = fmaxf(desc.order, 0.01f);
                    return 1.0f / (1.0f + powf(x, 2.0f * order));
                }
                if (desc.kind == EFilterKernel::Kaiser) {
                    const float beta = fmaxf(desc.beta, 0.0f);
                    const float t = 2.0f * x - 1.0f;
                    return bessel_i0(beta * sqrtf(fmaxf(0.0f, 1.0f - t * t))) /
                        bessel_i0(beta);
                }
                if (desc.kind == EFilterKernel::Tukey) {
                    const float alpha = fminf(fmaxf(desc.tukey_alpha, 0.0f), 1.0f);
                    if (alpha <= 0.0f) return 1.0f;
                    if (x < alpha * 0.5f)
                        return 0.5f * (1.0f + cosf(pi * (2.0f * x / alpha - 1.0f)));
                    if (x <= 1.0f - alpha * 0.5f) return 1.0f;
                    return 0.5f * (1.0f + cosf(pi * (2.0f * x / alpha - 2.0f / alpha + 1.0f)));
                }
                return window_shape(x, desc.kind);
            }

        } // namespace detail
    } // namespace Filter
} // namespace YK
