// YkFdkTaskHandle.hpp
#pragma once
#include <cuda_runtime.h>

#include "../global/YkCBCTParams.h"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../interface/IYkTask.hpp"
#include "../interface/YkTaskTypes.hpp"
#include "YkFdkReconstructor.hpp"

namespace YK {

    // ----------------------------------------------------------------
    // FdkTaskHandle
    //
    //   ITask 的 FDK 重建实现
    //   stream 生命周期：init() 创建，release() / 析构时销毁
    //   Kchunk：内部固定为 kMaxChunkAng，外部通过 TaskBatchParams.K 控制分批
    //
    //   输出模式（vol_mode / sino_mode）：
    //     DevicePtr — 外部提供 GPU 指针，直接写入，无回拷
    //     HostPtr   — 外部提供 CPU 指针，内部分配显存暂存，
    //                 最后一包（totalReceived >= iPAngTotal）时自动回拷
    // ----------------------------------------------------------------
    class FdkTaskHandle : public ITask {
    public:
        FdkTaskHandle() = default;
        ~FdkTaskHandle() override { release(); }

        FdkTaskHandle(const FdkTaskHandle&) = delete;
        FdkTaskHandle& operator=(const FdkTaskHandle&) = delete;

        // ----------------------------------------------------------------
        // init
        // ----------------------------------------------------------------
        bool init(const TaskInitParams& p) override
        {
            release();

            if (p.task != ETask::FDK) {
                YK_LOGE("[FdkTaskHandle] init: wrong task type {}\n", (int)p.task);
                return false;
            }
            if (p.gpu.empty()) {
                YK_LOGE("[FdkTaskHandle] init: no GPU specified.\n");
                return false;
            }
            device_id_ = p.gpu[0];

            SCBCTParams cp = mapParams(p);

            if (p.algoParams && p.algoParamSize >= sizeof(SFdkAlgoParams)) {
                const auto& ap = *static_cast<const SFdkAlgoParams*>(p.algoParams);
                cp.desc = SFilterKernelDesc{ mapFilter(ap.filter) };
            }
            else {
                cp.desc = SFilterKernelDesc{ EFilterKernel::RamLak };
            }

            YK_CUDA_CHECK(cudaStreamCreate(&stream_));

            if (!fdk_.init(cp, kMaxChunkAng, stream_, device_id_)) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            params_ = cp;
            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // reset / release
        // ----------------------------------------------------------------
        void reset() override
        {
            fdk_.reset();
            h_vol_pending_ = nullptr;
        }

        void release() override
        {
            fdk_.release();


            h_vol_pending_ = nullptr;
            vol_size_bytes_ = 0;

            if (stream_) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }

            params_ = {};
            is_initialized_ = false;
        }

        // ----------------------------------------------------------------
        // 状态查询
        // ----------------------------------------------------------------
        bool  isInitialized() const override { return is_initialized_; }
        ETask task()          const override { return ETask::FDK; }
        int   totalReceived() const { return fdk_.totalReceived(); }

        bool run(
            const FdkBatchParams& p,
            TaskDumpCallback       dump_cb = nullptr,
            void* userdata = nullptr) override
        {
            if (!is_initialized_) {
                YK_LOGE("[FdkTaskHandle] run: not initialized.");
                return false;
            }
            if (!p.h_proj || !p.h_angles || p.K <= 0) {
                YK_LOGE("[FdkTaskHandle] run: invalid params.");
                return false;
            }

            switch (p.vol_mode) {
            case EBufferMode::DevicePtr:
                return runDevice_(p, dump_cb, userdata);
            case EBufferMode::HostPtr:
                return runHost_(p, dump_cb, userdata);
            default:
                YK_LOGE("[FdkTaskHandle] run: unknown vol_mode.");
                return false;
            }
        }


    private:

        bool runDevice_(
            const FdkBatchParams& p,
            TaskDumpCallback       dump_cb,
            void* userdata)
        {
            if (!p.d_vol_out) {
                YK_LOGE("[FdkTaskHandle] runDevice_: d_vol_out is null.\n");
                return false;
            }

            SCBCTParams cp = buildCp_(p);
            return feedImpl_(p.h_proj, cp, p.d_vol_out, p.clearOut, dump_cb, userdata);
        }

        bool runHost_(
            const FdkBatchParams& p,
            TaskDumpCallback       dump_cb,
            void* userdata)
        {
            if (!p.h_vol_out) {
                fprintf(stderr, "[FdkTaskHandle] runHost_: h_vol_out is null.\n");
                return false;
            }

            // 懒分配内部显存
            if (!d_vol_internal_) {
                Mem::MemoryController mc;
                d_vol_internal_ = mc.allocateDevice3D<float>(
                    params_.iVX, params_.iVY, params_.iVZ, 0, true);

            }

            SCBCTParams cp = buildCp_(p);
            if (!feedImpl_(p.h_proj, cp, d_vol_internal_.data(), p.clearOut,
                dump_cb, userdata))
                return false;

            // 最后一包时回拷
            const bool is_last = (fdk_.totalReceived() >= params_.iPAngTotal);
            if (is_last) {
                cudaStreamSynchronize(stream_);

                YK_CUDA_CHECK(cudaMemcpy(
                    p.h_vol_out, d_vol_internal_.data(),
                    d_vol_internal_.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }

            return true;
        }

        // 公共：构建 SCBCTParams
        SCBCTParams buildCp_(const FdkBatchParams& p) const
        {
            SCBCTParams cp = params_;
            cp.iPAng = p.K;
            cp.angle_list.assign(p.h_angles, p.h_angles + p.K);
            return cp;
        }

        // 公共：调 fdk_.feed
        bool feedImpl_(
            const float* h_proj,
            const SCBCTParams& cp,
            float* d_out,
            bool             clear_vol,
            TaskDumpCallback dump_cb = nullptr,
            void* userdata = nullptr)
        {
            return fdk_.feed(h_proj, cp, stream_, d_out, clear_vol, dump_cb, userdata);
        }



    private:
        FdkReconstructor fdk_;
        SCBCTParams      params_{};
        cudaStream_t     stream_ = nullptr;
        bool             is_initialized_ = false;
        int              device_id_ = 0;

        // 体数据内部显存（HostPtr 模式）
        Mem::DeviceLinearBuffer3D<float> d_vol_internal_;
        float* h_vol_pending_;
        size_t  vol_size_bytes_ = 0;

        // ----------------------------------------------------------------
        // 公共几何参数映射
        // ----------------------------------------------------------------
        static SCBCTParams mapParams(const TaskInitParams& p)
        {
            SCBCTParams cp{};
            cp.iPU = p.scan.Nu;
            cp.iPV = p.scan.Nv;
            cp.iPAngTotal = p.scan.NAng;
            cp.du_mm = p.scan.du_mm;
            cp.dv_mm = p.scan.dv_mm;
            cp.offsetU_mm = p.scan.offsetU_mm;
            cp.offsetV_mm = p.scan.offsetV_mm;
            cp.tiltn_angle_rad = p.scan.tiltN_rad;
            cp.tiltu_angle_rad = p.scan.tiltU_rad;
            cp.tiltv_angle_rad = p.scan.tiltV_rad;
            cp.SID = p.scan.SOD_mm;
            cp.SDD = p.scan.SDD_mm;
            cp.scan_range_rad = p.scan.scanRangeRad;
            cp.scan_start_angle_rad = p.scan.startAngleRad;
            cp.bShortScan = p.scan.shortScan;
            cp.iVX = p.volume.Nx;
            cp.iVY = p.volume.Ny;
            cp.iVZ = p.volume.Nz;
            cp.vox_x_mm = p.volume.voxX_mm;
            cp.vox_y_mm = p.volume.voxY_mm;
            cp.vox_z_mm = p.volume.voxZ_mm;
            cp.vol_offset_x_mm = p.volume.offsetX_mm;
            cp.vol_offset_y_mm = p.volume.offsetY_mm;
            cp.vol_offset_z_mm = p.volume.offsetZ_mm;
            return cp;
        }

        // ----------------------------------------------------------------
        // EFdkFilter → EFilterKernel
        // ----------------------------------------------------------------
        static EFilterKernel mapFilter(EFdkFilter f)
        {
            switch (f) {
            case EFdkFilter::None:       return EFilterKernel::None;
            case EFdkFilter::RamLak:     return EFilterKernel::RamLak;
            case EFdkFilter::SheppLogan: return EFilterKernel::SheppLogan;
            case EFdkFilter::Cosine:     return EFilterKernel::Cosine;
            case EFdkFilter::Hann:       return EFilterKernel::Hann;
            case EFdkFilter::Hamming:    return EFilterKernel::Hamming;
            case EFdkFilter::Blackman:   return EFilterKernel::Blackman;
            default:                     return EFilterKernel::RamLak;
            }
        }
    };

} // namespace YK