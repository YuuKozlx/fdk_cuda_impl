// YkTaskTypes.hpp
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <vector>



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
        bool  force_dc_zero = false;

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
        // Required by iterative pipelines because their workspace is built
        // for the complete acquisition during initialize().
        std::vector<float> angles{};
    };

    // All buffers are contiguous float arrays in the library's canonical
    // layouts: projection [angle][v][u], volume [z][y][x].
    struct Buffer {
        float* data = nullptr;
        EMemoryLocation location = EMemoryLocation::Device;
    };

    // For FDK and FP, angles/K define one batch.  For iterative pipelines K
    // must equal SessionDesc::scan.NAng and projection contains all views.
    struct ExecuteRequest {
        const float* angles = nullptr; // host-resident radians
        int K = 0;
        Buffer projection{};
        Buffer volume{};
        bool clear_output = true;
        // Zero uses AlgorithmDesc::iterative.iterations.
        int iteration_count = 0;
    };

    // Optional diagnostics hook used by the internal FDK/FP runners.
    using TaskDumpCallback = std::function<void(void*)>;

    // Internal runners share this payload shape.  It remains public only
    // because callbacks are intentionally part of the C++ extension surface.
    struct DumpPayload {
        int viewIdx = 0;
        const char* stage = nullptr;
        void* buf = nullptr;
        size_t n = 0;
        cudaStream_t stream = nullptr;
        void* userdata = nullptr;
        bool isDevicePtr = true;
    };

} // namespace YK
