#pragma once
#include <cstdio>

#include <cuda_runtime.h>

#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFdkStageTypes.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"

#include "Reconstruction/Analytic/Circular/Flat/FDK/kernels/YkFDKPreWeightHelpers.cuh"
#include "Reconstruction/Analytic/Circular/Flat/FDK/kernels/YkFDKPreWeightLaunch.cuh"

namespace YK {
    namespace Fdk {

        // ============================================================
        // PreweightProcessor
        //
        //   生命周期：prepare(config) -> apply(input, output, chunk) -> release()
        //
        //   支持原地：d_output == d_input 合法。
        // ============================================================
        class PreweightProcessor {
        public:
            PreweightProcessor() = default;
            ~PreweightProcessor() { release(); }

            PreweightProcessor(const PreweightProcessor&) = delete;
            PreweightProcessor& operator=(const PreweightProcessor&) = delete;

            // 初始化配置只在 prepare 时传入；chunk 数据由 apply 显式传入。
            bool prepare(const PreweightConfig& config)
            {
                release();
                Nu_ = static_cast<int>(config.dims.iPU);
                Nv_ = static_cast<int>(config.dims.iPV);
                max_K_ = static_cast<int>(config.dims.iPAng);
                policy_ = detail::normalizePreweightPolicy(config.policy);
                if (Nu_ <= 0 || Nv_ <= 0 || max_K_ <= 0) {
                    std::fprintf(stderr, "[YK][Preweight][E] prepare: invalid dimensions.\n");
                    return false;
                }
                is_initialized_ = true;
                return true;
            }

            bool apply(const float* d_input, float* d_output,
                const PreweightChunk& chunk, cudaStream_t stream)
            {
                if (!is_initialized_ || !d_input || !d_output ||
                    !chunk.d_geo || !chunk.d_gv || chunk.K <= 0 || chunk.K > max_K_) {
                    std::fprintf(stderr, "[YK][Preweight][E] apply: invalid prepared state or chunk.\n");
                    return false;
                }
                detail::pw_launchPreweight(d_input, d_output, chunk.d_geo,
                    chunk.d_gv, Nu_, Nv_, chunk.K, policy_, stream);
                return true;
            }

            // 无 GPU 资源，release 仅清除配置。
            void release()
            {
                Nu_ = Nv_ = max_K_ = 0;
                is_initialized_ = false;
            }

            bool isPrepared() const { return is_initialized_; }

        private:
            int                    Nu_ = 0;
            int                    Nv_ = 0;
            int                    max_K_ = 0;
            SKernelLaunchPolicy    policy_ = {};
            bool                   is_initialized_ = false;
        };

    }
} // namespace YK::Fdk
