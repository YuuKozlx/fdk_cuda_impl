// YkFpTaskHandle.hpp
#pragma once
#include <cuda_runtime.h>

#include <cstring>
#include "global/YkCBCTParams.h"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "global/YkMem3d.hpp"
#include "YKCBCT/interface/IYkTask.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "FP/YkFpRunner.hpp"

namespace YK {

    // ----------------------------------------------------------------
    // FpTaskHandle
    //
    //   ITask 的 FP 正投影实现
    //   支持 FP_Joseph / FP_Siddon / FP_CVP 三种方法
    //   stream 生命周期：init() 创建，release() / 析构时销毁
    //
    //   输入模式（vol_in_mode）：
    //     DevicePtr — 外部直接提供 GPU 体数据指针
    //     HostPtr   — 外部提供 CPU 体数据指针，内部自动上传到显存
    //
    //   输出模式（sino_mode）：
    //     DevicePtr — 直接写入外部 GPU 指针
    //     HostPtr   — 内部显存暂存，最后一包自动回拷到外部 CPU 指针
    // ----------------------------------------------------------------
    class FpTaskHandle : public ITask {
    public:
        explicit FpTaskHandle(ETask task = ETask::FP_Joseph)
            : task_(task) {
        }

        ~FpTaskHandle() override { release(); }

        FpTaskHandle(const FpTaskHandle&) = delete;
        FpTaskHandle& operator=(const FpTaskHandle&) = delete;

        // ----------------------------------------------------------------
        // init
        // ----------------------------------------------------------------
        bool init(const TaskInitParams& p) override
        {
            release();

            if (p.task != ETask::FP_Joseph &&
                p.task != ETask::FP_Siddon &&
                p.task != ETask::FP_CVP)
            {
                YK_LOGE("[FpTaskHandle] init: wrong task type {}\n", (int)p.task);
                return false;
            }

            if (p.gpu.empty()) {
                YK_LOGE("[FpTaskHandle] init: no GPU specified.\n");
                return false;
            }

            device_id_ = p.gpu[0];

            task_ = p.task;

            SCBCTParams cp = mapParams(p);

            if (p.algoParams && p.algoParamSize >= sizeof(SFpAlgoParams)) {
                const auto& ap = *static_cast<const SFpAlgoParams*>(p.algoParams);
                stepSS_ = ap.stepSS;
                detSS_ = ap.detSS;
            }

            YK_CUDA_CHECK(cudaStreamCreate(&stream_));

            if (!fp_.init(cp, p.task, device_id_)) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            params_ = cp;
            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // run：按 vol_in_mode / sino_mode 分发
        // ----------------------------------------------------------------
        bool run(
            const FpBatchParams& p,
            TaskDumpCallback     dump_cb = nullptr,
            void* userdata = nullptr) override
        {
            if (!is_initialized_) {
                YK_LOGE("[FpTaskHandle] run: not initialized.\n");
                return false;
            }
            if (!p.h_angles || p.K <= 0) {
                YK_LOGE("[FpTaskHandle] run: invalid params.\n");
                return false;
            }


            if (p.vol_in_mode == EBufferMode::HostPtr && !p.h_vol_in) {
                YK_LOGE("[FpTaskHandle] run: vol_in_mode=HostPtr but h_vol_in is null.\n");
                return false;
            }

            if (p.vol_in_mode == EBufferMode::DevicePtr && !p.d_vol_in) {
                YK_LOGE("[FpTaskHandle] run: vol_in_mode=DevicePtr but d_vol_in is null.\n");
                return false;
            }


            if (p.sino_mode == EBufferMode::HostPtr && !p.h_sino_out) {
                YK_LOGE("[FpTaskHandle] run: sino_mode=HostPtr but h_sino_out is null.\n");
                return false;
            }

            if (p.sino_mode == EBufferMode::DevicePtr && !p.d_sino_out) {
                YK_LOGE("[FpTaskHandle] run: sino_mode=DevicePtr but d_sino_out is null.\n");
                return false;
            }


            if (p.clearOut) {
                // 根据输出模式清零输出缓冲
                switch (p.sino_mode) {
                case EBufferMode::DevicePtr:
                    if (p.d_sino_out)
                        YK_CUDA_CHECK(cudaMemsetAsync(p.d_sino_out, 0, sizeof(float) * params_.iPU * params_.iPV * params_.iPAngTotal, stream_));
                    break;
                case EBufferMode::HostPtr:
                    if (p.h_sino_out)
                        memset(p.h_sino_out, 0, sizeof(float) * params_.iPU * params_.iPV * params_.iPAngTotal);
                    break;
                }
            }

            // ---- 确定体数据输入 ----
            const float* d_vol = nullptr;
            switch (p.vol_in_mode) {
            case EBufferMode::DevicePtr:
                if (!p.d_vol_in) {
                    YK_LOGE("[FpTaskHandle] run: vol_in_mode=DevicePtr but d_vol_in is null.\n");
                    return false;
                }
                d_vol = p.d_vol_in;
                break;

            case EBufferMode::HostPtr:
                if (!p.h_vol_in) {
                    YK_LOGE("[FpTaskHandle] run: vol_in_mode=HostPtr but h_vol_in is null.\n");
                    return false;
                }
                d_vol = ensureVolInput_(p.h_vol_in);
                if (!d_vol) return false;
                break;
            }

            // ---- 确定正弦图输出目标 ----
            switch (p.sino_mode) {
            case EBufferMode::DevicePtr:
                return runDevice_(p, d_vol, dump_cb, userdata);
            case EBufferMode::HostPtr:
                return runHost_(p, d_vol, dump_cb, userdata);
            default:
                YK_LOGE("[FpTaskHandle] run: unknown sino_mode.\n");
                return false;
            }
        }

        // ----------------------------------------------------------------
        // reset / release
        // ----------------------------------------------------------------
        void reset() override
        {
            fp_.reset();
            total_received_ = 0;
        }

        void release() override
        {
            fp_.release();
            d_vol_input_.reset();
            sino_internal_.reset();

            if (stream_) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }

            total_received_ = 0;
            params_ = {};
            is_initialized_ = false;
        }

        bool  isInitialized() const override { return is_initialized_; }
        ETask task()          const override { return task_; }

    private:
        ETask            task_ = ETask::FP_Joseph;
        EFpStepSample    stepSS_ = EFpStepSample::x1;
        EFpDetSample     detSS_ = EFpDetSample::x1;
        ConeProjector  fp_;
        SCBCTParams      params_{};
        cudaStream_t     stream_ = nullptr;
        bool             is_initialized_ = false;
        int              total_received_ = 0;
        int              device_id_ = 0;

        // 体数据输入暂存（HostPtr 模式）
        Mem::DeviceLinearBuffer3D<float> d_vol_input_;

        // 正弦图输出暂存（HostPtr 模式）
        Mem::DeviceLinearBuffer3D<float> sino_internal_;

        // ----------------------------------------------------------------
        // DevicePtr 输出：直接写外部 GPU 指针
        // ----------------------------------------------------------------
        bool runDevice_(
            const FpBatchParams& p,
            const float* d_vol,
            TaskDumpCallback     dump_cb,
            void* userdata)
        {
            if (!p.d_sino_out) {
                YK_LOGE("[FpTaskHandle] runDevice_: d_sino_out is null.\n");
                return false;
            }
            return feedImpl_(p, d_vol, p.d_sino_out, dump_cb, userdata);
        }

        // ----------------------------------------------------------------
        // HostPtr 输出：内部显存暂存，最后一包回拷
        // ----------------------------------------------------------------
        bool runHost_(
            const FpBatchParams& p,
            const float* d_vol,
            TaskDumpCallback     dump_cb,
            void* userdata)
        {
            if (!p.h_sino_out) {
                YK_LOGE("[FpTaskHandle] runHost_: h_sino_out is null.\n");
                return false;
            }

            // 懒分配正弦图显存
            if (!sino_internal_) {
                Mem::MemoryController mc;
                sino_internal_ = mc.allocateDevice3D<float>(
                    (size_t)params_.iPU, params_.iPV, params_.iPAngTotal, device_id_, true);
            }

            if (!feedImpl_(p, d_vol, sino_internal_.data(), dump_cb, userdata))
                return false;

            // 最后一包时回拷
            const bool is_last = (total_received_ >= params_.iPAngTotal);
            if (is_last) {
                cudaStreamSynchronize(stream_);
                YK_CUDA_CHECK(cudaMemcpy(
                    p.h_sino_out, sino_internal_.data(),
                    sino_internal_.size() * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }

            return true;
        }

        // ----------------------------------------------------------------
        // HostPtr 输入：上传体数据到显存，返回 GPU 指针
        // ----------------------------------------------------------------
        const float* ensureVolInput_(const float* h_vol_in)
        {
            if (!d_vol_input_) {
                Mem::MemoryController mc;
                d_vol_input_ = mc.allocateDevice3D<float>(
                    (size_t)params_.iVX, params_.iVY, params_.iVZ, device_id_, true);
            }
            YK_CUDA_CHECK(cudaMemcpyAsync(
                d_vol_input_.data(), h_vol_in,
                d_vol_input_.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream_));
            return d_vol_input_.data();
        }

        // ----------------------------------------------------------------
        // 公共 feed 逻辑
        // ----------------------------------------------------------------
        bool feedImpl_(
            const FpBatchParams& p,
            const float* d_vol,
            float* d_sino,
            TaskDumpCallback     dump_cb,
            void* userdata)
        {
            SCBCTParams cp = params_;
            cp.iPAng = p.K;
            cp.iPAngTotal = p.K;
            cp.angle_list.assign(p.h_angles, p.h_angles + p.K);

            bool ok = fp_.run(d_vol, cp, d_sino, stream_, device_id_, dump_cb, userdata);
            if (ok) total_received_ += p.K;
            return ok;
        }
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
    };

} // namespace YK