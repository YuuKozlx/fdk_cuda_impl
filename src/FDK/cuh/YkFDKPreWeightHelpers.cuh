#pragma once
#include <algorithm>

#include "../YkFdkPipelineContext.hpp"   // SKernelLaunchPolicy

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // shared memory 槽位布局
            // ----------------------------------------------------------------
            constexpr int kValidFloats = 13;
            constexpr int kPreweightSlot = (kValidFloats + 15) & ~15;  // = 16

            // ----------------------------------------------------------------
            // Launch-policy normalization
            // ----------------------------------------------------------------
            YK_INLINE SKernelLaunchPolicy normalizePreweightPolicy(SKernelLaunchPolicy p)
            {
                p.block_threads = p.normalizedBlockThreads();
                return p;
            }

        }
    }
} // namespace YK::Fdk::detail
