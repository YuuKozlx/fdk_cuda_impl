#pragma once
#include <cuda_runtime.h>
#include "../global/YkGlobals.h"  // SVolGeom, SConeProjGeomVec

namespace YK {
    namespace Fp {

        // ----------------------------------------------------------------
        // init 阶段注入（init() 之前调用一次）
        // ----------------------------------------------------------------
        struct FpInitContext {
            SVolGeom vol_geom;
            int      Nu = 0;
            int      Nv = 0;
            bool     accumulate = false;  // true: +=   false: =
        };

        // ----------------------------------------------------------------
        // per-chunk 运行期注入（每次 process() 之前调用）
        // ----------------------------------------------------------------
        struct FpChunkContext {
            const SConeProjGeomVec* d_views = nullptr; // 世界坐标（Siddon 用）
            const SConeProjGeomVec* d_views_vox = nullptr; // 体素坐标系（Joseph 用）
            const SConeProjGeomVec* h_views = nullptr; // host，主轴分组用
            int K = 0;
            int angleOffset = 0;
        };

    } // namespace Fp
} // namespace YK