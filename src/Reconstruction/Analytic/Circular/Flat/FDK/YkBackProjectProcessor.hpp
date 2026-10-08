#pragma once


#include "global/YkGlobals.h"
#include "YkFdkStageTypes.hpp"

#include <driver_types.h>
#include <functional>
#include <texture_types.h>
#include "global/YkLog.h"
#include "kernels/YkFDKBpLaunch.cuh"

namespace YK {
    namespace Fdk {



        // ============================================================
        // BpProcessor
        //
        //   生命周期：prepare(config) -> apply(textures, volume, chunk) -> release()
        //
        //   两种模式由 BpConfig::use_precomputed 选择：
        //     true  — gC_coeffs constant memory 已在外部写入
        //     false — kernel 内直接计算，需要 d_geo / d_gv
        // ============================================================
        class BpProcessor {
        public:
            BpProcessor() = default;
            ~BpProcessor() { release(); }

            BpProcessor(const BpProcessor&) = delete;
            BpProcessor& operator=(const BpProcessor&) = delete;

            // 普通 FDK 只使用预计算系数路径；强类型入口避免每个 chunk 先写
            // 可变 context 再调用 process()。
            bool prepare(const BpConfig& config)
            {
                release();
                vol_geom_ = config.vol_geom;
                use_precomputed_ = config.use_precomputed;
                max_K_ = config.max_chunk_views;
                if (vol_geom_.Nx <= 0 || vol_geom_.Ny <= 0 || vol_geom_.Nz <= 0 ||
                    vol_geom_.vox_x <= 0.f || vol_geom_.vox_y <= 0.f || vol_geom_.vox_z <= 0.f ||
                    max_K_ <= 0) {
                    YK_LOGE("[YK][Bp][E] prepare: invalid volume geometry.");
                    return false;
                }
                is_initialized_ = true;
                return true;
            }

            bool apply(const cudaTextureObject_t* d_textures, float* d_volume,
                const BpChunk& chunk, cudaStream_t stream)
            {
                if (!is_initialized_ || !d_textures || !d_volume || chunk.K <= 0 || chunk.K > max_K_ ||
                    (!use_precomputed_ && (!chunk.d_geo || !chunk.d_gv))) {
                    YK_LOGE("[YK][Bp][E] apply: invalid prepared state or chunk.");
                    return false;
                }
                if (use_precomputed_)
                    Fdk::bp_launchBpPrecomputed(d_textures, d_volume, vol_geom_, chunk.K, stream);
                else
                    Fdk::bp_launchBpDirect(d_textures, chunk.d_geo, chunk.d_gv,
                        d_volume, vol_geom_, chunk.K, stream);
                return true;
            }

            void release()
            {
                vol_geom_ = {};
                use_precomputed_ = true;
                max_K_ = 0;
                is_initialized_ = false;
            }

            bool isPrepared() const { return is_initialized_; }

        private:
            SVolGeom        vol_geom_ = {};
            bool            use_precomputed_ = true;
            int             max_K_ = 0;
            bool            is_initialized_ = false;
        };

    }
} // namespace YK::Fdk
