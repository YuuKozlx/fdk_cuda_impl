#pragma once
#include <cuda_runtime.h>

#include "../YkVecGeo.hpp"
#include "../YkFdkPipelineContext.hpp"   // SKernelLaunchPolicy


#include "YkFDKPreWeightHelpers.cuh"
#include "YkFDKPreWeightKernels.cuh"
#include "../../global/YkMacro.hpp"

namespace YK {
    namespace Fdk {
        namespace detail {

            // ----------------------------------------------------------------
            // pw_launchPreweight
            //   启动锥束余弦预加权 kernel。
            //   d_src / d_dst 均为 [K, Nv, Nu] device buffer（支持原地）。
            // ----------------------------------------------------------------
            inline void pw_launchPreweight(
                const float* d_src,
                float* d_dst,
                const SConeProjGeomVec* d_geo,
                const SFDKGeoParamPerView* d_gv,
                int Nu, int Nv, int K,
                const SKernelLaunchPolicy& policy,
                cudaStream_t               stream)
            {
                const int blockThreads = policy.block_threads;
                const int warps_per_blk = blockThreads / 32;
                const int total_warps = K * Nv;
                const int blocks = (total_warps + warps_per_blk - 1) / warps_per_blk;
                const size_t smem_bytes = static_cast<size_t>(warps_per_blk)
                    * kPreweightSlot * sizeof(float);

                preweight_vec_chunk_rowwarp_kernel << <blocks, blockThreads, smem_bytes, stream >> > (
                    d_src, d_dst, d_geo, d_gv,
                    Nu, Nv, K,
                    policy.bounds_check ? 1 : 0);
                YK_CUDA_KERNEL_CHECK();
            }

        }
    }
} // namespace YK::Fdk::detail
