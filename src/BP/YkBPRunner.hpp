#pragma once
// YkBpReconstructor.hpp
#include <cuda_runtime_api.h>
#include <vector>
#include "../FDK/YkBackProjectProcessor.hpp"
#include "../FDK/YkFDKVecGeoDerived.hpp"
#include "../FDK/YkFdkPipelineContext.hpp"
#include "../FDK/YkVecGeo.hpp"
#include "../global/YkCBCTParams.h"
#include "../global/YkGlobals.h"
#include "../global/YkLog.h"
#include "../global/YkMacro.hpp"
#include "YkBPGpuContext.hpp"

namespace YK {


    struct BpDumpPayload {
        int          viewIdx;
        const char* stage;    // "flt_in"（输入的滤波投影）/ "vol"（体数据）
        void* d_buf;
        size_t       n;
        cudaStream_t stream;
        void* userdata;
    };

    using BpDumpCallback = std::function<void(void*)>;

    class BpReconstructor {
    public:
        BpReconstructor() = default;

        bool isInitialized() const { return is_initialized_; }
        int  totalReceived() const { return total_received_; }

        void reset()
        {
            total_received_ = 0;
        }

        void release()
        {
            bp_.release();
            gpu_ctx_.release();
            reset();
            Kchunk_ = 0;
            is_initialized_ = false;
        }

        // ----------------------------------------------------------------
        // init
        // d_flt_proj 已经是滤波后的投影，外部负责滤波
        // ----------------------------------------------------------------
        bool init(const SCBCTParams& params, int Kchunk, cudaStream_t stream,
            int device_id = 0)
        {
            Kchunk_ = std::min(Kchunk, kMaxChunkAng);

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;
            const int iPA_total = params.iPAngTotal;

            // BpProcessor
            {
                SVolGeom vol_geom = SVolGeom::make_centered(
                    iVX, iVY, iVZ, params.vox_x_mm, params.vox_z_mm);
                vol_geom.center = make_float3(
                    params.vol_offset_x_mm,
                    params.vol_offset_y_mm,
                    params.vol_offset_z_mm);

                BpInitContext ictx{};
                ictx.vol_geom = vol_geom;
                ictx.use_precomputed = false;
                bp_.setInitContext(&ictx);
                if (!bp_.init()) {
                    YK_LOGE("[BpReconstructor] BpProcessor init failed");
                    return false;
                }
            }

            // GpuContext：geo 按 iPAngTotal 分配，proj 按 Kchunk 分配
            {
                SProjDims dims{ iPU, iPV, iPA_total };
                gpu_ctx_.init(dims, iPA_total, stream, device_id);
            }

            is_initialized_ = true;
            return true;
        }

        // ----------------------------------------------------------------
        // feed：输入已滤波投影（GPU 指针），执行反投影累加到 d_vol_out
        // ----------------------------------------------------------------
        bool feed(
            const float* d_flt_proj,   // 已滤波投影，GPU 指针
            const SCBCTParams& params,
            cudaStream_t       stream,
            float* d_vol_out,
            bool               clear_vol = false,
            TaskDumpCallback onDump = nullptr,
            void* dumpUserData = nullptr)
        {
            if (!is_initialized_) {
                YK_LOGE("[BpReconstructor] not initialized");
                return false;
            }
            if (!d_flt_proj || params.iPAng <= 0) {
                YK_LOGE("[BpReconstructor] invalid input");
                return false;
            }
            if ((int)params.angle_list.size() != params.iPAng) {
                YK_LOGE("[BpReconstructor] angle_list size mismatch");
                return false;
            }

            const int batch_count = params.iPAng;
            const int prev_total = total_received_;
            total_received_ += batch_count;

            const int iPU = params.iPU;
            const int iPV = params.iPV;
            const int iVX = params.iVX;
            const int iVY = params.iVY;
            const int iVZ = params.iVZ;

            auto rad2deg = [](float r) { return r * 180.f / CUDA_PI; };

            // geo 只建当前 batch
            std::vector<SConeProjGeomVec>    h_geo(batch_count);
            std::vector<SFDKGeoParamPerView> h_gv(batch_count);

            build_circular_vec_geometry_from_theta(
                h_geo, params.angle_list, batch_count,
                iPU, iPV,
                params.du_mm, params.dv_mm,
                params.SID, params.SDD - params.SID,
                f3(params.offsetU_mm, params.offsetV_mm, 0.f),
                f3(rad2deg(params.tiltu_angle_rad),
                    rad2deg(params.tiltn_angle_rad),
                    rad2deg(params.tiltv_angle_rad)));

            GeoDerivedManagerVec{}.build_geo_params(
                iPU, iPV, params.scan_range_rad, h_geo, h_gv);

            // 增量上传 geo
            gpu_ctx_.uploadGeoIncremental(h_geo, h_gv, prev_total, batch_count, stream);

            if (clear_vol) {
                YK_CUDA_CHECK(cudaMemsetAsync(d_vol_out, 0,
                    (size_t)iVX * iVY * iVZ * sizeof(float), stream));
            }

            const size_t view_elems = (size_t)iPU * iPV;

            for (int base = 0; base < batch_count; base += Kchunk_) {
                const int K = std::min(Kchunk_, batch_count - base);
                const int global_base = prev_total + base;

                // 外部滤波数据拷入 d_sino（texture 绑定在此）
                YK_CUDA_CHECK(cudaMemcpyAsync(
                    gpu_ctx_.proj.d_sino.data(),
                    d_flt_proj + (size_t)base * view_elems,
                    K * view_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));

                gpu_ctx_.geo.uploadCoeffsChunk(
                    gpu_ctx_.geo.d_coeffs() + global_base, K, stream);

                BpChunkContext bctx{};
                bctx.d_geo = gpu_ctx_.geo.d_geo() + global_base;
                bctx.d_gv = gpu_ctx_.geo.d_gv() + global_base;
                bctx.K = K;
                bp_.setContext(&bctx);
                bp_.process(gpu_ctx_.proj.d_texObjs(), d_vol_out, stream);

                // dump 最终体数据
                if (onDump) {
                    DumpPayload payload{
                        prev_total, "vol",
                        static_cast<void*>(d_vol_out),
                        (size_t)params.iVX * params.iVY * params.iVZ,
                        stream, dumpUserData
                    };
                    onDump(&payload);
                }
            }

            return true;
        }

    private:
        int  Kchunk_ = 0;
        bool is_initialized_ = false;
        int  total_received_ = 0;

        YK::Fdk::BpProcessor bp_;
        YK::Bp::BpGpuContext    gpu_ctx_;
    };


    // ====================================================================
// fp_project
// 便捷函数：一次性调用，内部自动 init + run
// ====================================================================
    YK_INLINE bool bp_project(const float* d_flt_proj, const SCBCTParams& params,
        cudaStream_t stream, float* d_vol_out, bool clear_vol = false,
        TaskDumpCallback onDump = nullptr, void* dumpUserData = nullptr)
    {
        BpReconstructor recon;
        if (!recon.init(params, 16, stream)) {
            YK_LOGE("[YK][bp_project] init failed");
            return false;
        }
        return recon.feed(d_flt_proj, params, stream, d_vol_out, clear_vol, onDump, dumpUserData);
    }


};// namespace YK