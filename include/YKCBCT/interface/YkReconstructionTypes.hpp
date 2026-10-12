#pragma once
#include <cstddef>
#include <cstdint>
#include "YKCBCT/algorithms/projection/YkProjectionParams.hpp"
#include "YKCBCT/algorithms/analytic/YkFdkParams.hpp"
#include "YKCBCT/algorithms/analytic/YkWfbpParams.hpp"
#include "YKCBCT/algorithms/iterative/YkIterationParams.hpp"
#include "YKCBCT/algorithms/iterative/YkCglsParams.hpp"
#include "YKCBCT/algorithms/iterative/YkPwlsParams.hpp"
#include "YKCBCT/algorithms/iterative/YkTigreParams.hpp"

namespace YK {
    // Fill Host destination in the requested view-index order, [view][v][u].
    // Borrowed only during execute; false means an input read failure.
    using ProjectionReadCallback=bool(*)(void*,const int*,int,float*,size_t);

    enum class EPipeline : int32_t {
        FDK,
        ForwardProjection,
        SIRT,
        OSSART,
        CGLS,
        PWLS,
        XFDK,
        CFDK,
        WFBP,
        TigreGradient,
        SART,
        Statistical,
    };

    enum class EMemoryLocation : int32_t {
        Host,
        Device,
    };

// Contiguous float arrays: projection [view][v][u], volume [z][y][x].
    struct Buffer {
        float* data = nullptr;
        EMemoryLocation location = EMemoryLocation::Device;
        // 连续 float 元素容量。新 DLL API 要求调用方填写，用于在进入 CUDA
        // 后端前拒绝尺寸不足的投影或体缓冲；旧内部调用可暂时保留 0。
        size_t element_count = 0;
    };

} // namespace YK
