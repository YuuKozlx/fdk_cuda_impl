#pragma once
#include <cuda_runtime.h>

#include "../global/YkGlobals.h"              // kMaxChunkAng, CUDA_PI

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // Constant memory — 每 chunk 的相对扫描角（已归一化到 [0, 2π)）
            // ----------------------------------------------------------------
            __constant__ float gC_parker_angle[kMaxChunkAng];

        }
    }
} // namespace YK::Fdk::detail
