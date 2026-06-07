#pragma once
#pragma once

// ============================================================
// CUDA Common Macros & Utilities
// ============================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <vector_types.h>

namespace YK {
    struct SDimensions3D {
        unsigned int iVX;
        unsigned int iVY;
        unsigned int iVZ;
        unsigned int iPAng;
        unsigned int iPU; // number of detectors in the U direction
        unsigned int iPV; // number of detectors in the V direction
    };

    struct SProjDims {
        int iPU = 0;
        int iPV = 0;
        int iPAng = 0;

        static SProjDims from(const SDimensions3D& d) {
            return { (int)d.iPU, (int)d.iPV, (int)d.iPAng };
        }
        SProjDims() = default;
        SProjDims(int iPU, int iPV, int iPAng) : iPU(iPU), iPV(iPV), iPAng(iPAng) {}
    };

    struct SVolDims {
        int Nx = 0;
        int Ny = 0;
        int Nz = 0;

        static SVolDims from(const SDimensions3D& d) {
            return { (int)d.iVX, (int)d.iVY, (int)d.iVZ };
        }
        SVolDims() = default;
        SVolDims(int Nx, int Ny, int Nz) : Nx(Nx), Ny(Ny), Nz(Nz) {}
    };


    struct SVolGeom {
        // volume 尺寸（体素数）
        int Nx = 0, Ny = 0, Nz = 0;

        // 体素大小（mm）
        float vox_x = 1.f, vox_y = 1.f, vox_z = 1.f;

        float tmp_rcp_vox_x = 1.f, tmp_rcp_vox_y = 1.f, tmp_rcp_vox_z = 1.f; // 体素大小的倒数，预计算加速

        // volume 中心的世界坐标（mm），默认原点
        float3 center = make_float3(0.f, 0.f, 0.f);

        // 便捷构造：等体素，中心对齐世界原点
        static SVolGeom make_centered(int Nx, int Ny, int Nz, float vox) {
            SVolGeom g;
            g.Nx = Nx; g.Ny = Ny; g.Nz = Nz;
            g.vox_x = g.vox_y = g.vox_z = vox;
            g.center = make_float3(0.f, 0.f, 0.f);


            g.tmp_rcp_vox_x = g.tmp_rcp_vox_y = 1.f / vox;
            g.tmp_rcp_vox_z = 1.f / vox;
            return g;
        }

        static SVolGeom make_centered(int Nx, int Ny, int Nz, float vox_xy, float vox_z) {
            SVolGeom g;
            g.Nx = Nx; g.Ny = Ny; g.Nz = Nz;
            g.vox_x = g.vox_y = vox_xy; g.vox_z = vox_z;
            g.center = make_float3(0.f, 0.f, 0.f);


            g.tmp_rcp_vox_x = g.tmp_rcp_vox_y = 1.f / vox_xy;
            g.tmp_rcp_vox_z = 1.f / vox_z;
            return g;
        }

        static SVolGeom make_centered(int Nx, int Ny, int Nz, float vox_x, float vox_y, float vox_z) {
            SVolGeom g;
            g.Nx = Nx; g.Ny = Ny; g.Nz = Nz;
            g.vox_x = vox_x; g.vox_y = vox_y; g.vox_z = vox_z;
            g.center = make_float3(0.f, 0.f, 0.f);


            g.tmp_rcp_vox_x = 1.f / vox_x;
            g.tmp_rcp_vox_y = 1.f / vox_y;
            g.tmp_rcp_vox_z = 1.f / vox_z;
            return g;
        }

        // 推导 volume (0,0,0) 体素的世界坐标
        __host__ __device__  float3 origin() const {
            return make_float3(
                center.x - (Nx - 1) * 0.5f * vox_x,
                center.y - (Ny - 1) * 0.5f * vox_y,
                center.z - (Nz - 1) * 0.5f * vox_z);
        }
    };


    struct alignas(16) SConeProjGeomVec {
        float4 src;     // xyz = source position, w = unused
        float4 srcCR;   // xyz = center ray direction, w = unused
        float4 detS;    // xyz = detector (0,0) position, w = unused
        float4 detU;    // xyz = per-pixel U vector, w = unused
        float4 detV;    // xyz = per-pixel V vector, w = unused
        float4 angle;   // x = gantry angle, y/z = reserved, w = unused
    };

    // ---------------------- launch policy ----------------------
    struct SKernelLaunchPolicy {
        int  block_threads = 256;  // must be multiple of 32
        bool bounds_check = true;

        // 从 occupancy API 自动查询最优 block_threads
        template<typename KernelFunc>
        static SKernelLaunchPolicy fromKernel(KernelFunc* kernel,
            size_t dynamic_smem = 0,
            bool   bounds_check = true)
        {
            int block_size, min_grid_size;
            YK_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
                &min_grid_size, &block_size, kernel, dynamic_smem, 0));

            // 对齐到 32 的倍数
            block_size = (block_size / 32) * 32;
            block_size = std::max(block_size, 32);

            SKernelLaunchPolicy policy;
            policy.block_threads = block_size;
            policy.bounds_check = bounds_check;
            return policy;
        }
    };





    struct alignas(16) SFDKGeoParamPerView {
        // ---- 标量组，打包在一起 ----
        float theta = 0.f;
        float dtheta = 0.f;
        float fScaleDTheta = 1.f;// 扫描角度为2Pi是为1.0f，短扫描时为 2Pi / (maxTheta - minTheta)，权重用
        float SOD_mm = 0.f;
        // 由源到旋转轴的距离（mm）计算得出
        // 以下两个参数在 compute_SDD_offsets() 中计算得出，供仿射系数构造和求交使用：
        // 理想几何时，SDD_mm == SDD_plane_mm == 标称 SDD；探测器倾斜时，SDD_mm 略大于 SDD_plane_mm（差 1/cos(tilt) 倍），必须区分使用
        // 源点沿主射线方向到探测器平面交点的距离（mm）
        // = |principal_point - src|
        // 其中 principal_point = src + t * srcCR，t = SDD_plane_mm / (srcCR · det_n)
        // 理想几何下等于标称 SDD；探测器倾斜时略大于 SDD_plane_mm（差 1/cos(tilt) 倍）
        // 用于：预计算仿射系数 Cu/Cv/Cd 的构造（透视投影的缩放基准）

        float SDD_mm = 0.f;
        // 源点到探测器平面沿法向量方向的投影距离（mm）
        // = (detS - src) · det_n
        // 不是物理意义上的 SDD，是求交参数 t 的分子项
        // t = SDD_plane_mm / (d · det_n)，保证射线与探测器平面正确求交
        // 理想几何（探测器垂直于主射线）下等于 SDD_mm
        // 探测器倾斜时二者不同，必须用此值求交而不能用 SDD_mm
        float SDD_plane_mm = 0.f;
        float offsetU_pix = 0.f;
        float offsetV_pix = 0.f;

        float du_mm = 1.f;
        float dv_mm = 1.f;
        float inv_du_mm = 1.f;
        float inv_dv_mm = 1.f;

        int   Nu = 0;
        int   Nv = 0;
        float UU = 0.f;
        float VV = 0.f;

        float UV = 0.f;
        float invDetUV = 0.f;
        float detS_sub_src_dot_dU = 0.f;
        float detS_sub_src_dot_dV = 0.f;

        // ---- float4，天然 16 字节对齐，放在标量之后 ----
        float4 ray_center = { 0,0,0,0 };  // xyz = 方向，w = unused
        float4 det_n = { 0,0,0,0 };
        float4 det_u = { 0,0,0,0 };
        float4 det_v = { 0,0,0,0 };
    };


    // 该结构体用于__constan__ cuda全局常量的生命，不能初始化，否则编译报错
    // __constant__ 变量不能有动态初始化，只能零初始化或者没有初始化器。
    struct alignas(16)  FdkAffineCoeff {
        float Cu_x, Cu_y, Cu_z, Cu_w;
        float Cv_x, Cv_y, Cv_z, Cv_w;
        float Cd_x, Cd_y, Cd_z, Cd_w;
        float dtheta;
        float SID2;
        float fScaleDTheta;
        float SDD2;
        float du_mm;
        float dv_mm;
        float nReserved1;
        float nReserved2;

    };

    // constant 内存，按 chunk 上传
    // 1024 角度 × 64 bytes = 64KB，刚好在限制内
    static constexpr int kMaxChunkAng = 32;
    // YkFDKBackProject.cuh —— 只放 extern 声明



    // ============================================================
    // Filter options
    // ============================================================
    enum class EFilterKernel {
        None,        // no filtering (identity)
        RamLak,
        SheppLogan,
        Cosine,
        Hann,        // Hann == Hanning
        Hamming,
        Blackman,
        Custom
    };

    // 权重构建来源（保留两条路径）
    enum class EWeightsBuildSource {
        AnalyticFreq,     // 直接频域写 H(f)=|f|*window
        DiscreteRLFFT     // 空域RL->FFT 提取 ramp -> 乘窗
    };

    // 从离散RLFFT提取 ramp 的方式（仅对 EWeightsBuildSource::DiscreteRLFFT 有效）
    enum class ERampExtractMode {
        RealPart = 0,
        Magnitude = 1
    };

    struct SFilterKernelDesc {
        EFilterKernel        kind = EFilterKernel::RamLak;
        EWeightsBuildSource  source = EWeightsBuildSource::AnalyticFreq;
        ERampExtractMode     extract_mode = ERampExtractMode::RealPart;
        float                gain = 1.0f;
        float                cutoff = 0.5f;
        bool                 force_dc_zero = false;


        // Custom 路径专用，长度 = n_complex = paddedN/2+1  单边频域核
        std::vector<float>   custom_weights;

        static SFilterKernelDesc Custom(
            std::vector<float> weights, float gain = 1.0f)
        {
            SFilterKernelDesc d;
            d.kind = EFilterKernel::Custom;
            d.custom_weights = std::move(weights);
            d.gain = gain;
            return d;
        }

        // ── 工厂函数 ─────────────────────────────────────────
        static SFilterKernelDesc RamLak(
            EWeightsBuildSource src = EWeightsBuildSource::DiscreteRLFFT,
            float gain = 1.0f)
        {
            SFilterKernelDesc d;
            d.kind = EFilterKernel::RamLak;
            d.source = src;
            d.gain = gain;
            return d;
        }

        static SFilterKernelDesc Hamming(float cutoff = 0.5f, float gain = 1.0f)
        {
            SFilterKernelDesc d;
            d.kind = EFilterKernel::Hamming;
            d.source = EWeightsBuildSource::DiscreteRLFFT;
            d.cutoff = cutoff;
            d.gain = gain;
            return d;
        }

        static SFilterKernelDesc Hann(float cutoff = 0.5f, float gain = 1.0f)
        {
            SFilterKernelDesc d;
            d.kind = EFilterKernel::Hann;
            d.source = EWeightsBuildSource::DiscreteRLFFT;
            d.cutoff = cutoff;
            d.gain = gain;
            return d;
        }

        static SFilterKernelDesc Identity(float gain = 1.0f)
        {
            SFilterKernelDesc d;
            d.kind = EFilterKernel::None;
            d.gain = gain;
            return d;
        }
    };


}

