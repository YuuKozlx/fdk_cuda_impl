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

    // IMPORTANT: this stores SID by your definition (NOT classic SOD)
    float SOD_mm = 0.0f;   // == SID_mm
    float SDD_mm = 0.0f;

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
    float4 Cu;  // u 分子系数 (cx, cy, cz, cw)
    float4 Cv;  // v 分子系数
    float4 Cd;  // 分母系数
    float  dtheta;
    float  SID2; // SOD^2，权重用
    float fScaleDTheta; // 权重用，等于 2Pi / scan_range_rad，扫描角度为2Pi时为1.0f
};

// constant 内存，按 chunk 上传
// 1024 角度 × 64 bytes = 64KB，刚好在限制内
static constexpr int kMaxChunkAng = 32;
__constant__ FdkAffineCoeff gC_coeffs[kMaxChunkAng];


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


