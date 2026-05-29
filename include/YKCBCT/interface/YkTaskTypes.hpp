// YkTaskTypes.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <vector>



namespace YK {

    // ----------------------------------------------------------------
    // 任务类型
    // ----------------------------------------------------------------
    enum class ETask : int32_t {
        FDK = 0,
        FP_Joseph = 1,
        FP_Siddon = 2,
        FP_CVP = 3,
        BP_Siddon_RayDriven = 4,
        BP_Siddon_VoxDriven = 5,
        BP_FDK = 6,
        Bp_Joseph = 7,   //
        Bp_Joseph_v2 = 8, // 
        Bp_Joseph_v3 = 9, // 
        SART = 10,
        OSEM = 11,
    };

    enum class EFdkFilter : int32_t {
        None = 0,
        RamLak = 1,
        SheppLogan = 2,
        Cosine = 3,
        Hann = 4,
        Hamming = 5,
        Blackman = 6,
    };

    enum class EBufferMode {
        DevicePtr,  // 外部提供 GPU 指针
        HostPtr,    // 外部提供 CPU 指针，内部自动管理回拷/预上传
    };

    enum class EFpStepSample : int32_t { x1 = 1, x2 = 2, x4 = 4 };
    enum class EFpDetSample : int32_t { x1 = 1, x2 = 2, x4 = 4 };

    // ================================================================
    // 几何与体积
    // ================================================================

    struct SScanParams {
        float SOD_mm = 0.f;
        float SDD_mm = 0.f;
        int   Nu = 0;
        int   Nv = 0;
        float du_mm = 1.f;
        float dv_mm = 1.f;
        float offsetU_mm = 0.f;
        float offsetV_mm = 0.f;
        float tiltN_rad = 0.f;
        float tiltU_rad = 0.f;
        float tiltV_rad = 0.f;
        float scanRangeRad = 6.2832f;
        float startAngleRad = 0.f;
        bool  shortScan = false;
        int  NAng = 0; // 仅 FDK 用，表示总视图数（非批次大小）
    };

    struct SVolumeParams {
        int   Nx = 0;
        int   Ny = 0;
        int   Nz = 0;
        float voxX_mm = 1.f;
        float voxY_mm = 1.f;
        float voxZ_mm = 1.f;
        float offsetX_mm = 0.f;
        float offsetY_mm = 0.f;
        float offsetZ_mm = 0.f;
    };

    // ================================================================
    // 算法专属参数
    // ================================================================

    struct SFdkAlgoParams {
        EFdkFilter filter = EFdkFilter::RamLak;
    };

    struct SFpAlgoParams {
        EFpStepSample stepSS = EFpStepSample::x1;
        EFpDetSample  detSS = EFpDetSample::x1;
    };

    struct SIterAlgoParams {
        int   iterations = 10;
        float relaxation = 1.f;
        int   subsets = 1;
    };


    struct GPURes {
        std::vector<int> deviceIds;

        // 自动探测所有可用 GPU
        static GPURes autoDetect()
        {
            int count = 0;
            cudaGetDeviceCount(&count);
            GPURes res;
            for (int i = 0; i < count; ++i)
                res.deviceIds.push_back(i);
            return res;
        }

        // 手动指定
        static GPURes fromList(std::initializer_list<int> ids)
        {
            GPURes res;
            res.deviceIds.assign(ids);
            return res;
        }

        int count()              const { return (int)deviceIds.size(); }
        int operator[](int i)    const { return deviceIds[i]; }
        bool empty()             const { return deviceIds.empty(); }
    };

    // ================================================================
    // 接口参数
    // ================================================================
    struct TaskInitParams {
        ETask     task = ETask::FDK;
        SScanParams   scan;
        SVolumeParams volume;
        const void* algoParams = nullptr;
        size_t        algoParamSize = 0;
        GPURes  gpu = GPURes::fromList({ 0 });  // 默认用卡 0
    };

    // 公共基类
    struct BatchParams {
        const float* h_angles = nullptr;
        int          K = 0;
        bool         clearOut = true;
    };

    // FDK 重建参数
    struct FdkBatchParams : BatchParams {
        // 投影输入（仅 CPU）
        const float* h_proj = nullptr;

        // 体数据输出
        float* d_vol_out = nullptr;
        float* h_vol_out = nullptr;
        EBufferMode vol_mode = EBufferMode::DevicePtr;
    };

    struct FpBatchParams : BatchParams {
        // 体数据输入
        const float* d_vol_in = nullptr;
        const float* h_vol_in = nullptr;
        EBufferMode  vol_in_mode = EBufferMode::DevicePtr;

        // 正弦图输出
        float* d_sino_out = nullptr;
        float* h_sino_out = nullptr;
        EBufferMode sino_mode = EBufferMode::DevicePtr;
    };

    // YkDump.hpp
    struct DumpPayload {
        int          viewIdx;
        const char* stage;
        void* buf;
        size_t       n;
        cudaStream_t stream;
        void* userdata;
        bool         isDevicePtr = true;  // true=GPU, false=CPU
    };


    using TaskDumpCallback = std::function<void(void*)>;

} // namespace YK