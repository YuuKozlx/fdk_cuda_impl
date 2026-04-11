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

        // volume 中心的世界坐标（mm），默认原点
        float3 center = make_float3(0.f, 0.f, 0.f);

        // 便捷构造：等体素，中心对齐世界原点
        static SVolGeom make_centered(int Nx, int Ny, int Nz, float vox) {
            SVolGeom g;
            g.Nx = Nx; g.Ny = Ny; g.Nz = Nz;
            g.vox_x = g.vox_y = g.vox_z = vox;
            g.center = make_float3(0.f, 0.f, 0.f);
            return g;
        }

        static SVolGeom make_centered(int Nx, int Ny, int Nz, float vox_xy, float vox_z) {
            SVolGeom g;
            g.Nx = Nx; g.Ny = Ny; g.Nz = Nz;
            g.vox_x = g.vox_y = vox_xy; g.vox_z = vox_z;
            g.center = make_float3(0.f, 0.f, 0.f);
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
        float3 src;      // tube params : source position (world);
        float3 srcCR; // tube params：center ray direction [unit vector];
        float3 detS;     // detector pixel (0,0) position (world)
        float3 detU;     // per-pixel vector in U direction (world), length = du
        float3 detV;    // per-pixel vector in V direction (world), length = dv
        float3 angle;   // angle.x : gantry rotation angle [rad], measured (encoder)
        // angle.y : reserved (e.g. angle velocity / jitter / 0)
        // angle.z : reserved
    };

    // ---------------------- launch policy ----------------------
    struct SKernelLaunchPolicy {
        int block_threads = 256;   // warp-row: must be multiple of 32
        bool bounds_check = true;  // 是否检查 a in [0, Ang)
    };




    struct alignas(16) SFDKGeoParamPerView
    {
        float theta = 0.0f;
        float dtheta = 0.0f;
        float fScaleDTheta = 1.0f; // 扫描角度为2Pi是为1.0f，短扫描时为 2Pi / (maxTheta - minTheta)，权重用



        float SOD_mm = 0.0f;   // 由源到旋转轴的距离（mm）计算得出
        // 以下两个参数在 compute_SDD_offsets() 中计算得出，供仿射系数构造和求交使用：
        // 理想几何时，SDD_mm == SDD_plane_mm == 标称 SDD；探测器倾斜时，SDD_mm 略大于 SDD_plane_mm（差 1/cos(tilt) 倍），必须区分使用
        // 源点沿主射线方向到探测器平面交点的距离（mm）
        // = |principal_point - src|
        // 其中 principal_point = src + t * srcCR，t = SDD_plane_mm / (srcCR · det_n)
        // 理想几何下等于标称 SDD；探测器倾斜时略大于 SDD_plane_mm（差 1/cos(tilt) 倍）
        // 用于：预计算仿射系数 Cu/Cv/Cd 的构造（透视投影的缩放基准）
        float SDD_mm = 0.0f;

        // 源点到探测器平面沿法向量方向的投影距离（mm）
        // = (detS - src) · det_n
        // 不是物理意义上的 SDD，是求交参数 t 的分子项
        // t = SDD_plane_mm / (d · det_n)，保证射线与探测器平面正确求交
        // 理想几何（探测器垂直于主射线）下等于 SDD_mm
        // 探测器倾斜时二者不同，必须用此值求交而不能用 SDD_mm
        float SDD_plane_mm = 0.0f;

        int   Nu = 0;
        int   Nv = 0;

        float offsetU_pix = 0.0f;
        float offsetV_pix = 0.0f;

        float3 ray_center = { 0,0,0 };// from source to principal point (mm)

        float3 det_n = { 0,0,0 }; // detector plane normal (unit vector)


        float du_mm = 1.0f;
        float dv_mm = 1.0f;
        float inv_du_mm = 1.0f;
        float inv_dv_mm = 1.0f;

        float detS_sub_src_dot_dU = 0.0f;   // (detS - src) · detU
        float detS_sub_src_dot_dV = 0.0f;   // (detS - src) · detV

        float UU = 0.0f;
        float VV = 0.0f;
        float UV = 0.0f;
        float invDetUV = 0.0f;
    };



    struct FdkAffineCoeff {
        float Cu_x, Cu_y, Cu_z, Cu_w;
        float Cv_x, Cv_y, Cv_z, Cv_w;
        float Cd_x, Cd_y, Cd_z, Cd_w;
        float dtheta;
        float SID2;
        float fScaleDTheta;
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
        Blackman
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
        EFilterKernel kind = EFilterKernel::RamLak;

        // cutoff in DFT-normalized frequency:
        // f = k/N in [0,0.5], Nyquist=0.5
        float cutoff = 0.5f;

        float gain = 1.0f;


        // DC 处理（让离散/解析对齐）
        bool force_dc_zero = false;

        // 选择构建来源：保留两条路
        EWeightsBuildSource source = EWeightsBuildSource::DiscreteRLFFT;

        ERampExtractMode extract_mode = ERampExtractMode::RealPart; // 仅对 DiscreteRLFFT 有效

        SFilterKernelDesc() {}
        SFilterKernelDesc(EFilterKernel fkenel) :kind(fkenel) {}
    };



}

