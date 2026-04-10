#pragma once
#include <cuda_runtime.h>

#include "../global/YkCBCTParams.h"
#include "../global/YkGlobals.h"
#include "../global/YkMacro.hpp"
#include "../interface/IFdkReconstructor.hpp"
#include "../interface/YkFdkTypes.hpp"
#include "YkFdkReconstructor.hpp"

namespace YK {

    // ----------------------------------------------------------------
    // FdkReconstructorImpl
    //
    //   IReconstructor 的唯一实现，经由 FdkFactory 对外提供。
    //   本文件不对外分发。
    //
    //   stream 生命周期：init() 创建，release() / 析构时销毁。
    //   Kchunk：内部固定为 kMaxChunkAng，外部通过 FdkFeedParams.K 控制分批。
    // ----------------------------------------------------------------
    class FdkReconstructorHandle : public IFdkReconstructor {
    public:
        FdkReconstructorHandle() = default;

        ~FdkReconstructorHandle() override { release(); }

        FdkReconstructorHandle(const FdkReconstructorHandle&) = delete;
        FdkReconstructorHandle& operator=(const FdkReconstructorHandle&) = delete;

        // ----------------------------------------------------------------
        // init
        // ----------------------------------------------------------------
        bool init(const FdkInitParams& p) override
        {
            release();   // 幂等：允许重复 init

            // --- 映射到内部参数 ---
            SCBCTParams cp{};
            cp.iPU = p.Nu;
            cp.iPV = p.Nv;
            cp.du_mm = p.du_mm;
            cp.dv_mm = p.dv_mm;
            cp.offsetU_mm = p.offsetU_mm;
            cp.offsetV_mm = p.offsetV_mm;
            cp.tiltn_angle_rad = p.tiltN_rad;
            cp.tiltu_angle_rad = p.tiltU_rad;
            cp.tiltv_angle_rad = p.tiltV_rad;
            cp.SID = p.SOD_mm;
            cp.SDD = p.SDD_mm;
            cp.scan_range_rad = p.scanRangeRad;
            cp.scan_start_angle_rad = p.startAngleRad;
            cp.bShortScan = p.shortScan;
            cp.iVX = p.Nx;
            cp.iVY = p.Ny;
            cp.iVZ = p.Nz;
            cp.vox_xy_mm = p.voxXY_mm;
            cp.vox_z_mm = p.voxZ_mm;
            cp.vol_offset_x_mm = p.offsetX_mm;
            cp.vol_offset_y_mm = p.offsetY_mm;
            cp.vol_offset_z_mm = p.offsetZ_mm;


            cp.desc = SFilterKernelDesc{ mapFilter(p.filter) };
            // --- 创建内部 stream ---
            YK_CUDA_CHECK(cudaStreamCreate(&stream_));

            // --- 初始化重建器（Kchunk = kMaxChunkAng，内部常量）---
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
        // recon
        //   K 可以小于 kMaxChunkAng（尾包），FdkReconstructor::feed 内部
        //   会自动调用 reinitProcessors 处理不足 chunk 的情况。
        // ----------------------------------------------------------------
        bool recon(const FdkBatchParams& fp,
            FdkDumpCallback dump_cb, void* userdata) override
        {
            if (!is_initialized_) {
                fprintf(stderr, "[FdkReconstructorImpl] recon: not initialized.\n");
                return false;
            }
            if (!fp.h_proj || !fp.h_angles || fp.K <= 0 || !fp.d_vol) {
                fprintf(stderr, "[FdkReconstructorImpl] recon: invalid FdkFeedParams.\n");
                return false;
            }

            // 把测量角度填入内部临时参数（不跨 DLL）
            SCBCTParams cp = params_;
            cp.iPAng = fp.K;
            cp.iPAngTotal = fp.K;
            cp.angle_list.assign(fp.h_angles, fp.h_angles + fp.K);

            if (dump_cb) {
                // C 函数指针在 Impl 内包成 lambda，std::function 不出 DLL
                auto onDump = [&](int idx, const char* stage,
                    float* d, size_t n) {
                        dump_cb(userdata, idx, stage, d, n);
                    };
                return recon_.feed(fp.h_proj, cp, stream_,
                    fp.d_vol, fp.clearVol, onDump);
            }

            return recon_.feed(fp.h_proj, cp, stream_,
                fp.d_vol, fp.clearVol);
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
            recon_.release();     // 不再用赋值，直接调内部 release

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
        bool isInitialized() const override { return is_initialized_; }
        int  totalReceived() const override { return recon_.totalReceived(); }

    private:
        FdkReconstructor recon_;
        SCBCTParams      params_{};
        cudaStream_t     stream_ = nullptr;
        bool             is_initialized_ = false;

        // EFdkFilter（对外枚举）→ EFilterKernel（内部枚举）
        static EFilterKernel mapFilter(EFdkFilter f)
        {
            /*  None,        // no filtering (identity)
                RamLak,
                SheppLogan,
                Cosine,
                Hann,        // Hann == Hanning
                Hamming,
                Blackman
            */
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