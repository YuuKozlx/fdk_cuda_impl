#pragma once
#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>
#include <vector_functions.hpp>
#include <vector_types.h>

#include "FDK/YkBackProjectProcessor.hpp"
#include "FDK/YkFDKVecGeoDerived.hpp"
#include "FDK/YkFdkPipelineContext.hpp"
#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkLog.h"
#include "global/YkMacro.hpp"
#include "YkBPGpuContext.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace YK {

    class BpZSlabReconstructor {
    public:
        BpZSlabReconstructor() = default;

        void reset()
        {
            total_received_ = 0;
            h_vol_buffer_ = nullptr;
        }

        bool isInitialized() const { return is_initialized_; }
        int  totalReceived() const { return total_received_; }

        void release()
        {
            for (auto& bp : slab_bps_)
                if (bp) bp->release();
            slab_bps_.clear();
            slabs_.clear();

            if (d_vol_slab_) {
                cudaFree(d_vol_slab_);
                d_vol_slab_ = nullptr;
            }

            if (d_flt_proj_buf_) {
                cudaFree(d_flt_proj_buf_);
                d_flt_proj_buf_ = nullptr;
            }

            gpu_ctx_.release();
            h_vol_buffer_ = nullptr;
            h_vol_slab_buffer_.clear();

            reset();
            Kchunk_ = 0;
            is_initialized_ = false;
        }

        bool init(const SCBCTParams& params, int Kchunk,
            cudaStream_t stream, int z_block_size, int device_id = 0)
        {
            Kchunk_ = std::min(Kchunk, kMaxChunkAng);
            params_ = params;
            stream_ = stream;
            z_block_size_ = std::min(z_block_size, params.iVZ);

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;

            // ---- gpu_ctx_：geo 按 iPAngTotal，proj 按 Kchunk ----
            {
                SProjDims dims{ iPU, iPV, iPA_total };
                gpu_ctx_.init(dims, iPA_total, stream, device_id);
            }

            // ---- 设备端单块 vol slab buffer ----
            {
                const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
                YK_CUDA_CHECK(cudaMalloc(&d_vol_slab_, slab_elems_max * sizeof(float)));
            }

            // ---- CPU feed 中转 buffer（按 iPAngTotal 预分配，避免运行时分配）----
            {
                const size_t proj_elems_max = (size_t)iPU * iPV * iPA_total;
                YK_CUDA_CHECK(cudaMalloc(&d_flt_proj_buf_, proj_elems_max * sizeof(float)));
            }

            // ---- 按 z_block_size 分段，每段建一个 BpProcessor ----
            {
                const float vox_z = params.vox_z_mm;
                const float full_center_z = params.vol_offset_z_mm;

                const int num_slabs = (iVZ + z_block_size_ - 1) / z_block_size_;
                slabs_.reserve(num_slabs);
                slab_bps_.reserve(num_slabs);

                for (int z_start = 0; z_start < iVZ; z_start += z_block_size_) {
                    const int z_count = std::min(z_block_size_, iVZ - z_start);

                    SlabContext slab{};
                    slab.z_start = z_start;
                    slab.z_count = z_count;
                    slabs_.push_back(slab);

                    SVolGeom sub_geom = SVolGeom::make_centered(
                        iVX, iVY, z_count, params.vox_x_mm, vox_z);

                    const float slab_center_vox = z_start + z_count * 0.5f;
                    const float full_center_vox = iVZ * 0.5f;
                    const float z_offset_mm = (slab_center_vox - full_center_vox) * vox_z;

                    sub_geom.center = make_float3(
                        params.vol_offset_x_mm,
                        params.vol_offset_y_mm,
                        full_center_z + z_offset_mm);

                    auto bp_ptr = std::make_unique<Fdk::BpProcessor>();
                    BpInitContext bctx{};
                    bctx.vol_geom = sub_geom;
                    bctx.use_precomputed = true;
                    bp_ptr->setInitContext(&bctx);
                    if (!bp_ptr->init()) {
                        YK_LOGE("[BpZSlabReconstructor] BpProcessor init failed for slab z={}",
                            z_start);
                        return false;
                    }
                    slab_bps_.push_back(std::move(bp_ptr));
                }
            }

            // ---- CPU 单段回读 buffer ----
            {
                const size_t slab_elems_max = (size_t)iVX * iVY * z_block_size_;
                h_vol_slab_buffer_.resize(slab_elems_max);
            }

            is_initialized_ = true;
            YK_LOGI("[BpZSlabReconstructor] init OK: {}x{}x{} vox, {} slabs, z_block={}",
                iVX, iVY, iVZ, (int)slabs_.size(), z_block_size_);
            return true;
        }

        // ----------------------------------------------------------------
        // feed（设备端输入）
        // d_flt_proj：已滤波投影，设备端，iPU×iPV×iPAng 连续排列
        // ----------------------------------------------------------------
        bool feed(const float* d_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out, TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
        {
            if (!is_initialized_) {
                YK_LOGE("[BpZSlabReconstructor] not initialized");
                return false;
            }
            if (!d_flt_proj || params.iPAng <= 0) {
                YK_LOGE("[BpZSlabReconstructor] invalid input");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[BpZSlabReconstructor] angle_list size mismatch");
                return false;
            }

            return feedImpl_(d_flt_proj, params, stream, h_vol_out, onDump, dumpUserData);
        }

        // ----------------------------------------------------------------
        // feed（主机端输入）
        // h_flt_proj：已滤波投影，CPU 端，内部上传至 d_flt_proj_buf_ 后调用设备端版本
        // ----------------------------------------------------------------
        bool feedFromHost(const float* h_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out, TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
        {
            if (!is_initialized_) {
                YK_LOGE("[BpZSlabReconstructor] not initialized");
                return false;
            }
            if (!h_flt_proj || params.iPAng <= 0) {
                YK_LOGE("[BpZSlabReconstructor] invalid input");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[BpZSlabReconstructor] angle_list size mismatch");
                return false;
            }

            const size_t proj_elems = (size_t)params.iPU * params.iPV * params.iPAng;

            // 上传到预分配的设备端中转 buffer
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_flt_proj_buf_, h_flt_proj,
                proj_elems * sizeof(float),
                cudaMemcpyHostToDevice, stream));

            return feedImpl_(d_flt_proj_buf_, params, stream, h_vol_out, onDump, dumpUserData);
        }

    private:

        struct SlabContext {
            int z_start = 0;
            int z_count = 0;
        };

        // ----------------------------------------------------------------
        // feedImpl_：设备端投影 → slab 反投影 → 回读 → CPU 累加
        // ----------------------------------------------------------------
        bool feedImpl_(const float* d_flt_proj,
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* h_vol_out, TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
        {
            h_vol_buffer_ = h_vol_out;

            const int batch_count = params.iPAng;
            const int prev_total = total_received_;
            total_received_ += batch_count;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            // ---- 建当前 batch 的 geo ----
            std::vector<SConeProjGeomVec>    h_geo(batch_count);
            std::vector<SFDKGeoParamPerView> h_gv(batch_count);

            build_circular_vec_geometry_from_theta(
                h_geo, params.angle_list, batch_count,
                iPU, iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, 0.f, params.offsetV_mm),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_geo, h_gv);

            gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, prev_total, batch_count, stream);

            const size_t view_elems = (size_t)iPU * iPV;

            // ---- chunk 循环 ----
            for (int base = 0; base < batch_count; base += Kchunk_) {
                const int K = std::min(Kchunk_, batch_count - base);
                const int global_base = prev_total + base;

                auto triggerDump = [&](const char* stage, float* d_base) {
                    if (!onDump) return;
                    cudaStreamSynchronize(stream);
                    for (int i = 0; i < K; ++i) {
                        DumpPayload payload{
                            global_base + i, stage,
                            static_cast<void*>(d_base + i * view_elems),
                            view_elems, stream, dumpUserData
                        };
                        onDump(&payload);
                    }
                    };

                // 拷入 d_sino（texture 绑定在此）
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    gpu_ctx_.proj.d_sino.data(),
                    d_flt_proj + (size_t)base * view_elems,
                    K * view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));

                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + global_base, K, stream);

                // ---- 对每个 slab：清零 → 反投影 → 回读 → CPU 累加 ----
                for (int si = 0; si < (int)slabs_.size(); ++si) {
                    const SlabContext& slab = slabs_[si];
                    Fdk::BpProcessor& bp = *slab_bps_[si];

                    const size_t slab_elems = (size_t)iVX * iVY * slab.z_count;

                    // a. 清零
                    YK_CUDA_CHECK(cudaMemsetAsync(
                        d_vol_slab_, 0, slab_elems * sizeof(float), stream));

                    // b. 反投影
                    BpChunkContext bctx{};
                    bctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                    bctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                    bctx.K = K;
                    bp.setContext(&bctx);
                    bp.process(gpu_ctx_.proj.d_texObjs(), d_vol_slab_, stream);

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
                            slab_elems, stream, dumpUserData
                        };
                        onDump(&payload);
                    }

                    // d. 累加到外部 h_vol_buffer_ 对应段
                    const bool is_first_chunk = (prev_total == 0 && base == 0);
                    accumulateSlab_(slab.z_start, slab.z_count, iVX, iVY, is_first_chunk);
                }
            }

            return true;
        }

        void accumulateSlab_(int z_start, int z_count, int iVX, int iVY, bool is_first)
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

        int          Kchunk_ = 0;
        bool         is_initialized_ = false;
        int          total_received_ = 0;
        int          z_block_size_ = 0;

        SCBCTParams  params_{};
        cudaStream_t stream_ = nullptr;

        YK::Bp::BpGpuContext gpu_ctx_;

        float* d_vol_slab_ = nullptr;
        float* d_flt_proj_buf_ = nullptr;  // CPU feed 上传中转，init 时预分配
        float* h_vol_buffer_ = nullptr;  // 不持有，指向外部

        std::vector<SlabContext>                        slabs_;
        std::vector<std::unique_ptr<Fdk::BpProcessor>> slab_bps_;
        std::vector<float>                              h_vol_slab_buffer_;
    };

    // ================================================================
    // 便携函数（设备端输入）
    // ================================================================
    YK_INLINE bool bp_zslab_recon(
        const float* d_flt_proj,
        float* h_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        int                z_block_size, TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
    {
        BpZSlabReconstructor recon;
        if (!recon.init(params, Kchunk, stream, z_block_size))
            return false;
        return recon.feed(d_flt_proj, params, stream, h_vol_out, onDump, dumpUserData);
    }

    // ================================================================
    // 便携函数（主机端输入）
    // ================================================================
    YK_INLINE bool bp_zslab_recon_from_host(
        const float* h_flt_proj,
        float* h_vol_out,
        const SCBCTParams& params,
        int                Kchunk,
        cudaStream_t       stream,
        int                z_block_size, TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
    {
        BpZSlabReconstructor recon;
        if (!recon.init(params, Kchunk, stream, z_block_size))
            return false;
        return recon.feedFromHost(h_flt_proj, params, stream, h_vol_out, onDump, dumpUserData);
    }

} // namespace YK