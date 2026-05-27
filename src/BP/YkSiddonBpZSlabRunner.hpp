#pragma once
// YkSiddonZSlabReconstructor.hpp
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "YkSiddonBPRunner.hpp"

#include <cuda_runtime.h>

namespace YK {

    class SiddonZSlabReconstructor {
    public:
        SiddonZSlabReconstructor() = default;

        void reset()
        {
            total_received_ = 0;
            h_vol_buffer_ = nullptr;
        }

        bool isInitialized() const { return is_initialized_; }
        int  totalReceived() const { return total_received_; }

        void release()
        {
            if (d_vol_slab_) {
                cudaFree(d_vol_slab_);
                d_vol_slab_ = nullptr;
            }
            if (d_flt_proj_buf_) {
                cudaFree(d_flt_proj_buf_);
                d_flt_proj_buf_ = nullptr;
            }
            h_vol_buffer_ = nullptr;
            h_vol_slab_buffer_.clear();
            slabs_.clear();
            slab_params_.clear();
            reset();
            z_block_size_ = 0;
            is_initialized_ = false;
        }

        bool init(const SCBCTParams& params, ETask task,
            cudaStream_t stream, int z_block_size,
            int device_id = 0)
        {
            if (task != ETask::BP_Siddon_RayDriven &&
                task != ETask::BP_Siddon_VoxDriven &&
                task != ETask::Bp_Joseph)
            {
                YK_LOGE("[ConeBackprojector] init: task %d is not a BP task\n",
                    static_cast<int>(task));
                return false;
            }

            params_ = params;
            task_ = task;
            stream_ = stream;
            z_block_size_ = std::min(z_block_size, params.iVZ);

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;

            // ---- 设备端单块 vol slab buffer ────────────────────────
            {
                const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
                YK_CUDA_CHECK(cudaMalloc(&d_vol_slab_, slab_elems_max * sizeof(float)));
            }

            // ---- CPU feed 中转 buffer ───────────────────────────────
            {
                const size_t proj_elems_max = (size_t)iPU * iPV * iPA_total;
                YK_CUDA_CHECK(cudaMalloc(&d_flt_proj_buf_, proj_elems_max * sizeof(float)));
            }

            // ---- 按 z_block_size 分段，每段建子参数 ────────────────
            {
                const float vox_z = params.vox_z_mm;
                const float full_center_z = params.vol_offset_z_mm;
                const int   num_slabs = (iVZ + z_block_size_ - 1) / z_block_size_;
                slabs_.reserve(num_slabs);
                slab_params_.reserve(num_slabs);

                for (int z_start = 0; z_start < iVZ; z_start += z_block_size_) {
                    const int z_count = std::min(z_block_size_, iVZ - z_start);

                    SlabContext slab{};
                    slab.z_start = z_start;
                    slab.z_count = z_count;
                    slabs_.push_back(slab);

                    // 子 params：只改体积 Z 尺寸和 Z 中心偏移
                    SCBCTParams sp = params;
                    sp.iVZ = z_count;
                    const float slab_center_vox = z_start + z_count * 0.5f;
                    const float full_center_vox = iVZ * 0.5f;
                    sp.vol_offset_z_mm = full_center_z
                        + (slab_center_vox - full_center_vox) * vox_z;
                    slab_params_.push_back(sp);
                }
            }

            // ---- CPU 回读 buffer ────────────────────────────────────
            {
                const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
                h_vol_slab_buffer_.resize(slab_elems_max);
            }

            is_initialized_ = true;
            YK_LOGI("[SiddonZSlabReconstructor] init OK: {}x{}x{} vox, {} slabs, z_block={}",
                iVX, iVY, iVZ, (int)slabs_.size(), z_block_size_);
            return true;
        }

        // ── 设备端输入 ───────────────────────────────────────────────
        bool feed(const float* d_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out,
            TaskDumpCallback   onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!checkReady_(d_flt_proj, params)) return false;
            return feedImpl_(d_flt_proj, params, stream, h_vol_out, onDump, userdata);
        }

        // ── 主机端输入 ───────────────────────────────────────────────
        bool feedFromHost(const float* h_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out,
            TaskDumpCallback   onDump = nullptr,
            void* userdata = nullptr)
        {
            if (!checkReady_(h_flt_proj, params)) return false;

            const size_t proj_elems = (size_t)params.iPU * params.iPV * params.iPAng;
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_flt_proj_buf_, h_flt_proj,
                proj_elems * sizeof(float),
                cudaMemcpyHostToDevice, stream));

            return feedImpl_(d_flt_proj_buf_, params, stream, h_vol_out, onDump, userdata);
        }

    private:
        struct SlabContext {
            int z_start = 0;
            int z_count = 0;
        };

        bool checkReady_(const float* d_sino, const SCBCTParams& params)
        {
            if (!is_initialized_) {
                YK_LOGE("[SiddonZSlabReconstructor] not initialized"); return false;
            }
            if (!d_sino || params.iPAng <= 0) {
                YK_LOGE("[SiddonZSlabReconstructor] invalid input"); return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[SiddonZSlabReconstructor] angle_list size mismatch"); return false;
            }
            return true;
        }

        bool feedImpl_(const float* d_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out,
            TaskDumpCallback   onDump,
            void* userdata)
        {
            h_vol_buffer_ = h_vol_out;

            const int prev_total = total_received_;
            total_received_ += params.iPAng;

            const int iVX = params.iVX;
            const int iVY = params.iVY;

            const size_t view_elems = (size_t)params.iPU * params.iPV;

            // ---- 对每个 slab：清零 → 反投影 → 回读 → CPU 累加 ────
            for (int si = 0; si < (int)slabs_.size(); ++si) {
                const SlabContext& slab = slabs_[si];
                SCBCTParams& sp = slab_params_[si];

                // 把当前 batch 的角度同步到子 params
                sp.iPAng = params.iPAng;
                sp.angle_list = params.angle_list;

                const size_t slab_elems = (size_t)iVX * iVY * slab.z_count;

                // a. 清零
                YK_CUDA_CHECK(cudaMemsetAsync(
                    d_vol_slab_, 0, slab_elems * sizeof(float), stream));

                // b. 反投影（ConeBackprojector 每次 run 重建几何，无状态）
                ConeBackprojector bp;
                bp.init(sp, task_);
                bp.run(d_flt_proj, sp, d_vol_slab_, stream,
                    /*clear_vol=*/false);   // 已手动清零

                // c. 同步后回读
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                YK_CUDA_CHECK(cudaMemcpy(
                    h_vol_slab_buffer_.data(), d_vol_slab_,
                    slab_elems * sizeof(float),
                    cudaMemcpyDeviceToHost));

                if (onDump) {
                    DumpPayload payload{
                        si, "vol_slab",
                        static_cast<void*>(d_vol_slab_),
                        slab_elems, stream, userdata
                    };
                    onDump(&payload);
                }

                // d. 累加到外部 h_vol_buffer_ 对应段
                const bool is_first = (prev_total == 0);
                accumulateSlab_(slab.z_start, slab.z_count, iVX, iVY, is_first);
            }

            return true;
        }

        void accumulateSlab_(int z_start, int z_count,
            int iVX, int iVY, bool is_first)
        {
            const size_t slab_elems = (size_t)iVX * iVY * z_count;
            const size_t z_offset = (size_t)iVX * iVY * z_start;
            float* dst = h_vol_buffer_ + z_offset;
            const float* src = h_vol_slab_buffer_.data();

            if (is_first) {
                std::memcpy(dst, src, slab_elems * sizeof(float));
                return;
            }

#ifdef NDEBUG
            constexpr size_t kBlockSize = 1 << 14;
#pragma omp parallel for schedule(static)
            for (int block = 0; block < (int)slab_elems; block += (int)kBlockSize) {
                const size_t end = std::min((size_t)block + kBlockSize, slab_elems);
#pragma omp simd
                for (size_t i = block; i < end; ++i)
                    dst[i] += src[i];
            }
#else
            for (size_t i = 0; i < slab_elems; ++i)
                dst[i] += src[i];
#endif
        }

        bool         is_initialized_ = false;
        int          total_received_ = 0;
        int          z_block_size_ = 0;
        ETask        task_ = ETask::BP_Siddon_RayDriven;

        SCBCTParams  params_{};
        cudaStream_t stream_ = nullptr;

        float* d_vol_slab_ = nullptr;
        float* d_flt_proj_buf_ = nullptr;
        float* h_vol_buffer_ = nullptr;

        std::vector<SlabContext>  slabs_;
        std::vector<SCBCTParams>  slab_params_;
        std::vector<float>        h_vol_slab_buffer_;
    };

    // ================================================================
    // 便捷函数
    // ================================================================
    YK_INLINE bool siddon_zslab_recon(
        const float* d_flt_proj,
        float* h_vol_out,
        const SCBCTParams& params,
        ETask              task,
        cudaStream_t       stream,
        int                z_block_size,
        TaskDumpCallback   onDump = nullptr,
        void* userdata = nullptr)
    {
        SiddonZSlabReconstructor recon;
        if (!recon.init(params, task, stream, z_block_size)) return false;
        return recon.feed(d_flt_proj, params, stream, h_vol_out, onDump, userdata);
    }

    YK_INLINE bool siddon_zslab_recon_from_host(
        const float* h_flt_proj,
        float* h_vol_out,
        const SCBCTParams& params,
        ETask              task,
        cudaStream_t       stream,
        int                z_block_size,
        TaskDumpCallback   onDump = nullptr,
        void* userdata = nullptr)
    {
        SiddonZSlabReconstructor recon;
        if (!recon.init(params, task, stream, z_block_size)) return false;
        return recon.feedFromHost(h_flt_proj, params, stream, h_vol_out, onDump, userdata);
    }

} // namespace YK