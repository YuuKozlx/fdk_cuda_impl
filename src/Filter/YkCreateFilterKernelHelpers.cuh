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

                default:
                    return 1.0f;
                }
            }

        } // namespace detail
    } // namespace Filter
} // namespace YK
