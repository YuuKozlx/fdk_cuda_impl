#pragma once
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "FDK/YkFdkStageTypes.hpp"
#include "global/YkGlobals.h"              // CUDA_PI
#include "global/YkMacro.hpp"

#include "FDK/kernels/YkFDKParkerLaunch.cuh"

namespace YK {
    namespace Fdk {

        // ============================================================
        // ParkerWeightProcessor
        //
        //   生命周期：prepare(config) -> apply(projection, chunk) -> release()
        // ============================================================
        class ParkerWeightProcessor {
        public:
            ParkerWeightProcessor() = default;
            ~ParkerWeightProcessor() { release(); }

            ParkerWeightProcessor(const ParkerWeightProcessor&) = delete;
            ParkerWeightProcessor& operator=(const ParkerWeightProcessor&) = delete;

            // 初始化配置只在 prepare 时传入；每个 chunk 的角度从 geometry 中
            // 提取，避免引入与几何不一致的第二份角度数组。
            bool prepare(const ParkerWeightConfig& config)
            {
                release();
                Nu_ = config.dims.iPU;
                Nv_ = config.dims.iPV;
                max_K_ = config.dims.iPAng;
                fScale_ = config.fScanRangeRad / static_cast<float>(CUDA_PI);
                fAngleBase_ = config.fStartAngleRad;
                nDirSign_ = config.nDirSign;
                if (Nu_ <= 0 || Nv_ <= 0 || max_K_ <= 0 ||
                    (nDirSign_ != 1 && nDirSign_ != -1)) {
                    std::fprintf(stderr, "[YK][Parker][E] prepare: invalid configuration.\n");
                    return false;
                }
                is_initialized_ = true;
                return true;
            }

            bool apply(float* d_projection, const ParkerWeightChunk& chunk,
                cudaStream_t stream)
            {
                if (!is_initialized_ || !d_projection || !chunk.h_geometry ||
                    !chunk.d_geometry || !chunk.d_gv ||
                    chunk.K <= 0 || chunk.K > max_K_) {
                    std::fprintf(stderr, "[YK][Parker][E] apply: invalid prepared state or chunk.\n");
                    return false;
                }
                std::vector<float> angles(chunk.K);
                for (int i = 0; i < chunk.K; ++i)
                    angles[i] = chunk.h_geometry[i].angle.x;
                detail::pk_uploadAngles(angles.data(), chunk.K, fAngleBase_, nDirSign_);
                detail::pk_launchParker(d_projection, Nu_, Nv_, chunk.K,
                    chunk.d_geometry, chunk.d_gv, fScale_, nDirSign_, stream);
                return true;
            }

            void release()
            {
                Nu_ = 0;
                Nv_ = 0;
                max_K_ = 0;
                fScale_ = 1.f;
                fAngleBase_ = 0.f;
                is_initialized_ = false;
            }

            bool isPrepared() const { return is_initialized_; }

        private:
            int   Nu_ = 0;
            int   Nv_ = 0;
            int   max_K_ = 0;
            float fScale_ = 1.f;
            int nDirSign_ = +1; // 方向符号，+1 或 -1，影响权重函数的正负号
            /// 基准角度：相对角 = 绝对角 - fAngleBase_
            float fAngleBase_ = 0.f;

            bool                     is_initialized_ = false;
        };

    }
} // namespace YK::Fdk
