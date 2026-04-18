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
                if (p.block_threads < 32) p.block_threads = 32;
                p.block_threads = (p.block_threads + 31) & ~31;
                p.block_threads = std::min(p.block_threads, 1024);
                return p;
            }

        }
    }
} // namespace YK::Fdk::detail
