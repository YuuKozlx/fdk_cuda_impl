// YkTaskTypes.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>
#include "YKCBCT/geometry/YkProjectionGeometry.hpp"



namespace YK {

    // Low-level projector/backprojector implementations.  These are used by
    // AlgorithmDesc; they are not independent DLL tasks.
    enum class ETask : int32_t {
        FDK = 0,
        FP_Joseph = 1,
        FP_Siddon = 2,
        FP_CVP = 3,
        BP_Siddon_RayDriven = 4,
        BP_Siddon_VoxDriven = 5,
        BP_FDK = 6,
        BP_FDK_matched = 7,
        BP_Joseph = 8,   //
        BP_Joseph_v2 = 9, // 
        BP_Joseph_v3 = 10, // 
        SART = 11,
        OSEM = 12,
    };

    enum class EFdkFilter : int32_t {
        None = 0,
        RamLak = 1,
        SheppLogan = 2,
        Cosine = 3,
        Hann = 4,
        Hamming = 5,
        Blackman = 6,
        // Parameterized windows.  Keep existing numeric values stable because
        // this enum is part of the public task interface.
        Butterworth = 7,
        Kaiser = 8,
        Tukey = 9,
    };

    enum class EPipeline : int32_t {
        FDK,
        ForwardProjection,
        SIRT,
        OSSART,
        CGLS,
    };

    enum class EMemoryLocation : int32_t {
        Host,
        Device,
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
        int nDirSign = 1; // 1角度递增，-1角度递减
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

        // Shared frequency-domain window parameters.  cutoff is normalized
        // to Nyquist: 0.5 means the complete representable band.
        float cutoff = 0.5f;
        float gain = 1.f;
        // Finite discrete Ram-Lak always suppresses DC; this is fixed
        // algorithm behavior and is not exposed as a runtime option.

        // Filter-specific parameters.  Unused fields are ignored.
        float butterworth_order = 2.f;
        float kaiser_beta = 8.6f;
        float tukey_alpha = 0.5f;
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

    struct AlgorithmDesc {
        EPipeline pipeline = EPipeline::FDK;
        ETask forward_projector = ETask::FP_Joseph;
        ETask back_projector = ETask::BP_Joseph_v2;
        SFdkAlgoParams fdk{};
        SIterAlgoParams iterative{};
    };

    struct SessionDesc {
        SScanParams scan{};
        SVolumeParams volume{};
        AlgorithmDesc algorithm{};
        GPURes gpu = GPURes::fromList({ 0 });
        // FDK 优先使用完整逐视图 geometry。geometry 非空时是唯一的几何和
        // 角度来源，长度必须等于 scan.NAng；当前公开 FDK 仅接受初始化时的
        // 完整 geometry，不接受 execute() 逐批改变几何。
        std::vector<SConeProjGeomVec> geometry{};
        // 圆轨迹回退输入；geometry 为空时，FDK 用它构造完整 geometry。
        // 迭代管线当前仍需要完整 angles 建立工作区。
        std::vector<float> angles{};
    };

    // All buffers are contiguous float arrays in the library's canonical
    // layouts: projection [angle][v][u], volume [z][y][x].
    struct Buffer {
        float* data = nullptr;
        EMemoryLocation location = EMemoryLocation::Device;
    };

    // 对 FDK 和 FP，angles/K 描述一个批次。FDK 批次按 execute() 调用顺序
    // 追加，总视图数必须恰好达到 SessionDesc::scan.NAng；重建完成后，在
    // reset() 前 session 会拒绝继续提交 FDK 批次。clear_output 仅允许用于
    // initialize() 或 reset() 后的首批，防止意外清掉此前的在线累积结果。
    // 对迭代管线，K 必须等于 SessionDesc::scan.NAng，projection 包含全视图。
    struct ExecuteRequest {
        const float* angles = nullptr; // host-resident radians
        int K = 0;
        Buffer projection{};
        Buffer volume{};
        bool clear_output = true;
        // Zero uses AlgorithmDesc::iterative.iterations.
        int iteration_count = 0;
    };

} // namespace YK
