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
                fprintf(stderr, "[FdkTaskHandle] init: wrong task type %d\n", (int)p.task);
                return false;
            }

            SCBCTParams cp = mapParams(p);

            // 解析 FDK 专属参数
            if (p.algoParams && p.algoParamSize >= sizeof(SFdkAlgoParams)) {
                const auto& ap = *static_cast<const SFdkAlgoParams*>(p.algoParams);
                cp.desc = SFilterKernelDesc{ mapFilter(ap.filter) };
            }
            else {
                cp.desc = SFilterKernelDesc{ EFilterKernel::RamLak };
            }

            YK_CUDA_CHECK(cudaStreamCreate(&stream_));

            if (!recon_.init(cp, kMaxChunkAng, stream_)) {
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
                return false;
            }

            params_ = cp;
            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // run
        // K 可以小于 kMaxChunkAng（尾包），内部自动处理
        // ----------------------------------------------------------------
        bool run(
            const TaskBatchParams& p,
            TaskDumpCallback       dump_cb = nullptr,
            void* userdata = nullptr) override
        {
            if (!is_initialized_) {
                fprintf(stderr, "[FdkTaskHandle] run: not initialized.\n");
                return false;
            }
            if (!p.h_proj || !p.h_angles || p.K <= 0 || !p.d_vol_out) {
                fprintf(stderr, "[FdkTaskHandle] run: invalid params.\n");
                return false;
            }

            SCBCTParams cp = params_;
            cp.iPAng = p.K;
            cp.iPAngTotal = p.K;
            cp.angle_list.assign(p.h_angles, p.h_angles + p.K);

            if (dump_cb) {
                auto onDump = [&](int idx, const char* stage, float* d, size_t n) {
                    dump_cb(userdata, idx, stage, d, n);
                    };
                return recon_.feed(p.h_proj, cp, stream_,
                    p.d_vol_out, p.clearOut, onDump);
            }

            return recon_.feed(p.h_proj, cp, stream_,
                p.d_vol_out, p.clearOut);
        }

        // ----------------------------------------------------------------
        // reset / release
        // ----------------------------------------------------------------
        void reset() override
        {
            recon_.reset();
        }

        void release() override
        {
            recon_.release();

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
        bool      isInitialized() const override { return is_initialized_; }
        ETask task()          const override { return ETask::FDK; }
        int       totalReceived() const { return recon_.totalReceived(); }

    private:
        FdkReconstructor recon_;
        SCBCTParams      params_{};
        cudaStream_t     stream_ = nullptr;
        bool             is_initialized_ = false;

        // ----------------------------------------------------------------
        // 公共几何参数映射（与 FpTaskHandle 共用逻辑）
        // ----------------------------------------------------------------
        static SCBCTParams mapParams(const TaskInitParams& p)
        {
            SCBCTParams cp{};
            cp.iPU = p.scan.Nu;
            cp.iPV = p.scan.Nv;
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
        // EFdkFilter（对外枚举）→ EFilterKernel（内部枚举）
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