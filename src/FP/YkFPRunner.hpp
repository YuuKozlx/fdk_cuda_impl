#pragma once
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

#include "common/YkVecGeo.hpp"
#include "global/YkCBCTParams.h"
#include "global/YkCudaTextureController.hpp"
#include "global/YkGlobals.h"
#include "global/YkMacro.hpp"
#include "util/YkCudaTimer.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"      // ETask 定义，按实际路径调整

#include <cuda_runtime_api.h>
#include "global/YkLog.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"
#include "YkFPGpuContext.hpp"
#include "Fp/kernels/YkFPLaunch.cuh"

namespace YK {

    // ConeProjector 专属
    struct FpDumpPayload {
        int          viewIdx;
        const char* stage;
        void* d_buf;
        size_t       n;
        cudaStream_t stream;
        void* userdata;
    };

    using FpDumpCallback = std::function<void(void*)>;

    // ====================================================================
    // ConeProjector
    //
    // 职责：给定体数据指针 + CBCTParams + ETask，执行对应正投影算法，
    //       将结果写入调用方提供的 d_sino_out。
    //
    // 支持的 ETask：
    //   FP_Joseph  —— 带纹理，三线性插值
    //   FP_Siddon  —— 带纹理，Siddon 步进
    //   FP_CVP     —— 无纹理，体素裸指针
    //
    // 其他 ETask（FDK / SART / OSEM）会直接报错返回 false。
    // ====================================================================
    class ConeProjector {
    public:
        bool isInitialized() const { return is_initialized_; }

        // release：释放所有资源，恢复初始状态
        void release()
        {
            is_initialized_ = false;
            fp_type_ = ETask::FP_Joseph;
            YK_LOGD("[ConeProjector] release: resources released, back to initial state\n");
        }

        // reset：保留初始化状态，只清空运行时累积量
        // ConeProjector 目前没有跨 run 的累积状态，
        // 但为接口一致性保留
        void reset()
        {
            // 无累积状态需要清空
            YK_LOGD("[ConeProjector] reset: no internal state to reset\n");
        }

        bool init(const SCBCTParams& params, ETask task = ETask::FP_Joseph, int deviceId = 0)
        {
            if (params.iPU <= 0 || params.iPV <= 0
                || params.iVX <= 0 || params.iVY <= 0 || params.iVZ <= 0)
            {
                YK_LOGE("[ConeProjector] init: invalid geometry params\n");
                return false;
            }

            // ── 任务类型检查 ─────────────────────────────────────────────
            if (task != ETask::FP_Joseph &&
                task != ETask::FP_Siddon &&
                task != ETask::FP_CVP)
            {
                YK_LOGE(
                    "[ConeProjector] run: task %d is not a FP task\n",
                    static_cast<int>(task));
                return false;
            }

            fp_type_ = task;

            is_initialized_ = true;
            return true;
        }

        bool run(const float* d_vol,
            const SCBCTParams& params,
            float* d_sino_out,
            cudaStream_t        stream,
            int                 deviceId = 0,
            TaskDumpCallback     onDump = nullptr,
            void* userdata = nullptr)
        {

            // ── 基本就绪检查 ─────────────────────────────────────────────
            if (!is_initialized_ || !d_vol || !d_sino_out) {
                fprintf(stderr, "[ConeProjector] run: not ready\n");
                return false;
            }
            if (params.iPAng <= 0
                || static_cast<int>(params.angle_list.size()) != params.iPAng)
            {
                fprintf(stderr, "[ConeProjector] run: invalid angle params\n");
                return false;
            }

            // ── 构建几何 ─────────────────────────────────────────────────
            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            SVolGeom vol_geom = SVolGeom::make_centered(
                params.iVX, params.iVY, params.iVZ,
                params.vox_x_mm, params.vox_y_mm, params.vox_z_mm);
            vol_geom.center = make_float3(
                params.vol_offset_x_mm,
                params.vol_offset_y_mm,
                params.vol_offset_z_mm);

            std::vector<SConeProjGeomVec> h_views(params.iPAng);
            build_circular_vec_geometry_from_theta(
                h_views, params.angle_list, params.iPAng,
                params.iPU, params.iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            // ── 按任务决定是否创建纹理 ───────────────────────────────────
            //   Joseph / Siddon：三线性插值，必须走纹理路径
            //   CVP            ：直接操作体素指针，不建纹理节省显存和时间
            ETask task = fp_type_;
            const bool needTex = (task == ETask::FP_Joseph || task == ETask::FP_CVP || task == ETask::FP_Siddon);

            Fp::FpGpuContext gpuctx;
            if (needTex)
                gpuctx.init(d_vol, vol_geom, h_views, deviceId);
            else
                gpuctx.initNoTex(d_vol, vol_geom, h_views, deviceId);

            // ── 清零输出 ─────────────────────────────────────────────────
            const size_t sino_elems =
                static_cast<size_t>(params.iPAng) * params.iPV * params.iPU;
            YK_CUDA_CHECK(cudaMemsetAsync(
                d_sino_out, 0, sino_elems * sizeof(float), stream));

            // ── src_dirs（Joseph / Siddon 需要，CVP 不需要）─────────────
            std::vector<float4> h_src_dirs;
            if (needTex) {
                const int Na = params.iPAng;
                h_src_dirs.resize(Na);
                for (int i = 0; i < Na; ++i)
                    h_src_dirs[i] = gpuctx.geo.h_views_vox()[i].src;
            }

            const int Na = params.iPAng;

            // ── Dispatch ─────────────────────────────────────────────────
            switch (task)
            {
            case ETask::FP_Joseph:
            {
                //Util::CudaTimer timer{ "FP_Joseph", stream };
                Fp::fp_joseph_launch(
                    gpuctx.volTex.tex,
                    gpuctx.geo.h_views_vec(),
                    gpuctx.geo.d_views_vox(),
                    d_sino_out,
                    vol_geom,
                    Na, params.iPU, params.iPV,
                    false,
                    stream, Fp::FpStepSuperSample::x1);
                break;
            }
            case ETask::FP_Siddon:
            {
                //Util::CudaTimer timer{ "FP_Siddon", stream };
                Fp::fp_siddon_launch(
                    gpuctx.volTex.tex,
                    d_sino_out,
                    gpuctx.geo.d_views(),
                    vol_geom,
                    params.iPU, params.iPV, Na,
                    false,
                    stream
                );
                break;
            }


            case ETask::FP_CVP:
                Fp::fp_cvp_launch(
                    gpuctx.volTex.tex,
                    d_sino_out,
                    gpuctx.geo.h_views(),
                    vol_geom,
                    Na, params.iPU, params.iPV,
                    stream);
                break;

            default:
                // 前面已拦截，理论上不可达
                return false;
            }

            // ── 可选 dump（逐 view 回调）────────────────────────────────
            if (onDump) {
                YK_CUDA_CHECK(cudaStreamSynchronize(stream));
                const size_t view_elems = (size_t)params.iPU * params.iPV;
                for (int i = 0; i < Na; ++i) {
                    DumpPayload payload{
                        i, "sino",
                        static_cast<void*>(d_sino_out + i * view_elems),
                        view_elems, stream, userdata
                    };
                    onDump(&payload);
                }
            }

            return true;
        }

    private:
        bool is_initialized_ = false;
        ETask fp_type_ = ETask::FP_Joseph;
    };

    // ====================================================================
    // fp_project
    // 便捷函数：一次性调用，内部自动 init + run
    // ====================================================================
    YK_INLINE bool fp_project(
        const float* d_vol,
        float* d_sino_out,
        const SCBCTParams& params,
        ETask              task,
        cudaStream_t       stream,
        int                deviceId = 0,
        TaskDumpCallback onDump = nullptr,
        void* dumpUserData = nullptr)
    {
        ConeProjector fp;
        if (!fp.init(params, task, deviceId)) return false;
        return fp.run(d_vol, params, d_sino_out, stream, deviceId, onDump, dumpUserData);
    }

} // namespace YK