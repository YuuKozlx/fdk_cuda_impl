#pragma once
#include <cstdint>

namespace YK {

    enum class EFdkFilter : int32_t {
        None,        // no filtering (identity)
        RamLak,
        SheppLogan,
        Cosine,
        Hann,        // Hann == Hanning
        Hamming,
        Blackman
    };

    // ----------------------------------------------------------------
    // 系统固定配置，init() 时传一次
    // 全部 POD，无内部类型，无 std:: 容器，DLL 边界安全
    // ----------------------------------------------------------------
    struct FdkInitParams {
        // 探测器
        int   Nu = 0;
        int   Nv = 0;
        float du_mm = 1.0f;
        float dv_mm = 1.0f;
        float offsetU_mm = 0.0f;
        float offsetV_mm = 0.0f;
        float tiltN_rad = 0.0f;
        float tiltU_rad = 0.0f;
        float tiltV_rad = 0.0f;

        // 扫描几何
        float SOD_mm = 0.0f;
        float SDD_mm = 0.0f;
        float scanRangeRad = 6.2832f;
        float startAngleRad = 0.0f;
        bool  shortScan = false;

        // 重建体积
        int   Nx = 0;
        int   Ny = 0;
        int   Nz = 0;
        float voxXY_mm = 1.0f;
        float voxZ_mm = 1.0f;
        float offsetX_mm = 0.0f;
        float offsetY_mm = 0.0f;
        float offsetZ_mm = 0.0f;

        // 滤波
        EFdkFilter filter = EFdkFilter::RamLak;

    };

    // ----------------------------------------------------------------
    // 每次 feed 时传入的动态数据
    // stream 由库内部管理，调用方不传
    // ----------------------------------------------------------------
    struct FdkBatchParams {
        const float* h_proj;    // host，[K * Nv * Nu]
        const float* h_angles;  // host，K 个绝对角度（rad），测量值
        int          K;         // 本次视角数，库内部按 kMaxChunkAng 自动分批
        float* d_vol;     // device，[Nx * Ny * Nz]，累加写入
        bool         clearVol;  // true = 先清零再累加，第一批时通常为 true，后续批次为 false
    };

    // ----------------------------------------------------------------
    // dump 回调（调试用，可为 nullptr）
    // 用 C 函数指针而非 std::function，保证 DLL 边界安全
    // ----------------------------------------------------------------
    using FdkDumpCallback = void(*)(
        void* userdata,   // 调用方自定义上下文
        int         viewIndex,
        const char* stageName,  // "pw" / "parker" / "flt"
        float* d_data,     // device pointer
        size_t      count);

} // namespace YK