// YkTaskTypes.hpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
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
        // Flat 体素驱动 Siddon 的两个既有优化版本。追加枚举值以保持前面
        // 已公开任务的 ABI 数值不变；三者数值语义不同，不做隐式替换。
        BP_Siddon_VoxDriven_v2 = 13,
        BP_Siddon_VoxDriven_v3 = 14,
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
        PWLS,
        XFDK,
        CFDK,
        CylAnalyticFDK,
        WFBP,
        TigreGradient,
        SART,
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
        // 源端校准偏移，只用于前端构造 geometry，不由 kernel 解释。
        float sourceOffsetX_mm = 0.f;
        float sourceOffsetY_mm = 0.f;
        float sourceOffsetZ_mm = 0.f;
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

    // CGLS 的停止条件独立于通用迭代次数。0 表示关闭对应条件，避免用
    // 魔数或负数在配置层隐式表达“禁用”。
    struct SCglsAlgoParams {
        int minimum_iterations = 0;
        int check_interval = 1;
        int patience = 1;
        float relative_residual = 0.f;
        bool robust_restart = true;
    };

    enum class EPwlsRegularizerSpec : int32_t {
        None,
        Quadratic,
        Huber,
    };

    struct SPwlsAlgoParams {
        EPwlsRegularizerSpec regularizer = EPwlsRegularizerSpec::Quadratic;
        float regularization = 1e-3f;
        float huber_delta = 3e-3f;
        float epsilon = 1e-6f;
        float lower_bound = 0.f;
        float upper_bound = std::numeric_limits<float>::max();
        // 投影统计权重 W 默认全 1；实际数组由执行请求提供，配置只声明
        // 是否消费该数组，避免在算法描述中保存有生命周期的设备指针。
        bool use_projection_weights = false;
    };

    enum class ETigreGradientMethodSpec : int32_t {
        Sart,
        OsSart,
        Sirt,
        AsdPocs,
        OsAsdPocs,
        BAsdPocsBeta,
        Pcsd,
        OsPcsd,
        AwPcsd,
        OsAwPcsd,
        AwAsdPocs,
        OsAwAsdPocs,
    };

    struct STigreGradientAlgoParams {
        ETigreGradientMethodSpec method = ETigreGradientMethodSpec::OsSart;
        int block_size = 20;
        float lambda_reduction = 1.f;
        bool nesterov_relaxation = false;
        bool fdk_initialization = false;
        bool non_negative = true;
        int tv_iterations = 20;
        float tv_alpha = 0.002f;
        float tv_alpha_reduction = 0.95f;
        float maximum_update_ratio = 0.95f;
        float max_l2_error = -1.f;
    };

    enum class EWfbpInputDetectorSpec : int32_t {
        EquiangularArc,
        FlatPanel,
        CylindricalArc,
    };

    enum class EWfbpFocalSpotSpec : int32_t {
        None,
        Phi,
        Z,
        PhiAndZ,
    };

    struct SWfbpAlgoParams {
        EWfbpInputDetectorSpec input_detector = EWfbpInputDetectorSpec::FlatPanel;
        EWfbpFocalSpotSpec focal_spot = EWfbpFocalSpotSpec::None;
        float redundancy_flat = 0.6f;
        float filter_cutoff = 1.f;
        float filter_apodization = 1.f;
        float anode_angle_rad = 0.f;
        bool reverse_row_interleave = false;
    };

    // 柱面解析 FDK 的后端始终消费 R=SDD 的规范几何。该配置描述前端
    // 是否自动执行物理曲率到规范曲率的重排，以及短扫冗余权重策略。
    struct SCylAnalyticFdkParams {
        bool map_to_canonical_radius = true;
        bool short_scan_parker = true;
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
        SCglsAlgoParams cgls{};
        SPwlsAlgoParams pwls{};
        STigreGradientAlgoParams tigre{};
        SWfbpAlgoParams wfbp{};
        SCylAnalyticFdkParams cyl_analytic_fdk{};
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
        // 柱面逐视图几何，与 Flat geometry 分开保存，供
        // CylAnalyticFDK 使用；两者不应同时作为同一请求的真源。
        std::vector<SCylConeProjGeomVec> cyl_geometry{};
        // 将 Scanner 坐标下的逐视图几何转换到固定 Object 坐标系。
        // 空数组表示恒等变换；长度 1 表示所有视图共用；否则必须等于 NAng。
        // 模体体素数组不会被重采样，volume.offsetX/Y/Z 仍是 Object 坐标系下
        // 的真实体积中心。若输入的是 scannerFromObject，请先调用 inverse()。
        std::vector<SRigidTransform> objectFromScanner{};
        // 圆轨迹回退输入；geometry 为空时，FDK 用它构造完整 geometry。
        // 迭代管线当前仍需要完整 angles 建立工作区。
        std::vector<float> angles{};
    };

    // All buffers are contiguous float arrays in the library's canonical
    // layouts: projection [angle][v][u], volume [z][y][x].
    struct Buffer {
        float* data = nullptr;
        EMemoryLocation location = EMemoryLocation::Device;
        // 连续 float 元素容量。新 DLL API 要求调用方填写，用于在进入 CUDA
        // 后端前拒绝尺寸不足的投影或体缓冲；旧内部调用可暂时保留 0。
        size_t element_count = 0;
    };

    // 对 FDK 和 FP，angles/K 描述一个批次。FDK 批次按 execute() 调用顺序
    // 追加，总视图数必须恰好达到 SessionDesc::scan.NAng；重建完成后，在
    // reset() 前 session 会拒绝继续提交 FDK 批次。clear_output 仅允许用于
    // initialize() 或 reset() 后的首批，防止意外清掉此前的在线累积结果。
    // 对迭代管线，K 必须等于 SessionDesc::scan.NAng，projection 包含全视图。
    // 公共 Session 的 execute() 采用同步请求语义：返回 true 后本次 GPU 工作
    // 已完成，输入缓冲可立即复用，输出缓冲可立即读取。需要跨请求流水并行时，
    // 应使用各算法底层的异步接口及其 completion fence/event。
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
