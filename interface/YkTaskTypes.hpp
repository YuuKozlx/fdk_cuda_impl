// YkTaskTypes.hpp
#pragma once
#include <cstdint>

namespace YK {

    // ----------------------------------------------------------------
    // 任务类型
    // ----------------------------------------------------------------
    enum class ETask : int32_t {
        FDK = 0,
        FP_Joseph = 1,
        FP_Siddon = 2,
        FP_CVP = 3,
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

    // ================================================================
    // 接口参数
    // ================================================================

    struct TaskInitParams {
        ETask     task = ETask::FDK;
        SScanParams   scan;
        SVolumeParams volume;
        const void* algoParams = nullptr;
        size_t        algoParamSize = 0;
    };

    struct TaskBatchParams {
        const float* h_proj = nullptr;
        const float* d_vol_in = nullptr;
        float* d_vol_out = nullptr;
        float* d_sino_out = nullptr;
        const float* h_angles = nullptr;
        int          K = 0;
        bool         clearOut = true;
    };

    using TaskDumpCallback = void(*)(
        void* userdata,
        int         viewIndex,
        const char* stageName,
        float* d_data,
        size_t      count);

} // namespace YK