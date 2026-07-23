#include "cvp_geometry.cuh"
#include "cvp_forward.cuh"
#include "cvp_back.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>



// ============================================================
//  示例：在你的 FDK pipeline 中集成 CVP projector
//
//  假设你已有：
//    - d_volume[Nz*Ny*Nx]      （device float*）
//    - d_sino[n_views][N*M]    （device float*，per-view）
//    - geom_vec[n_views]       （host SConeProjGeomVec array）
// ============================================================

// ---- 你需要填写的 volume 参数 ----
struct ReconParams {
    int   Nx, Ny, Nz;
    float a1, a2, a3;  // 体素尺寸 [mm]
    int   M, N;        // 探测器尺寸（cols U, rows V）
};

SVolumeDesc make_vol_desc(const ReconParams& p) {
    SVolumeDesc v;
    // 体素(0,0,0)中心 = volume 中心对齐到 isocenter
    v.origin = make_float3(
        -(p.Nx - 1) * 0.5f * p.a1,
        -(p.Ny - 1) * 0.5f * p.a2,
        -(p.Nz - 1) * 0.5f * p.a3);
    v.a1 = p.a1; v.a2 = p.a2; v.a3 = p.a3;
    v.Nx = p.Nx; v.Ny = p.Ny; v.Nz = p.Nz;
    return v;
}

// ============================================================
//  Forward projection：volume → sinogram（所有views）
// ============================================================
void project_all_views(
    const float* d_volume,
    float** d_sinos,          // [n_views] device ptrs
    const SConeProjGeomVecCVP* geom_vec,
    int                     n_views,
    const ReconParams& p,
    cudaStream_t            stream = 0)
{
    SVolumeDesc vol = make_vol_desc(p);

    // 预分配 cos_theta buffer（每view共用一块，串行views时复用）
    float* d_cos_theta;
    cudaMalloc(&d_cos_theta, p.M * p.N * sizeof(float));

    for (int v = 0; v < n_views; ++v) {
        // 清零当前 view 的 sinogram
        cudaMemsetAsync(d_sinos[v], 0, p.M * p.N * sizeof(float), stream);

        // 构建 view cache（host端计算，开销可忽略）
        SCVPViewCache cache = make_view_cache(geom_vec[v], p.M, p.N, vol);

        // 启动 forward kernels（3步：cos_theta → scatter → normalize）
        launch_cvp_forward(d_volume, d_sinos[v], d_cos_theta, cache, stream);
    }

    cudaFree(d_cos_theta);
}

// ============================================================
//  Back projection：sinogram → volume（所有views，累加）
//  FDK 流程：sino 已经过 ramp filter + Parker weight
// ============================================================
void backproject_all_views(
    float** d_sinos,   // [n_views] 已滤波的 sinogram
    float* d_volume,  // 输出，需在调用前清零
    const SConeProjGeomVecCVP* geom_vec,
    int                     n_views,
    const ReconParams& p,
    cudaStream_t            stream = 0)
{
    SVolumeDesc vol = make_vol_desc(p);

    for (int v = 0; v < n_views; ++v) {
        SCVPViewCache cache = make_view_cache(geom_vec[v], p.M, p.N, vol);

        // fdk_weight=true：使用 SDD²/r² 的 FDK 标准权重
        launch_cvp_back(d_sinos[v], d_volume, cache, /*fdk_weight=*/true, stream);
    }
}

// ============================================================
//  Quick sanity check（host端，无 GPU 编译时用来检查几何）
// ============================================================
void debug_print_cache(const SCVPViewCache& c) {
    printf("=== SCVPViewCache ===\n");
    printf("src       : (%.2f, %.2f, %.2f)\n", c.src.x, c.src.y, c.src.z);
    printf("srcCR     : (%.4f, %.4f, %.4f)\n", c.srcCR.x, c.srcCR.y, c.srcCR.z);
    printf("det_center: (%.2f, %.2f, %.2f)\n", c.det_center.x, c.det_center.y, c.det_center.z);
    printf("SDD       : %.2f mm\n", c.SDD);
    printf("du/dv     : %.4f / %.4f mm\n", c.du, c.dv);
    printf("M x N     : %d x %d\n", c.M, c.N);
    printf("vol_origin: (%.2f, %.2f, %.2f)\n", c.vol_origin.x, c.vol_origin.y, c.vol_origin.z);
    printf("a1/a2/a3  : %.4f / %.4f / %.4f mm\n", c.a1, c.a2, c.a3);
    printf("Nx/Ny/Nz  : %d / %d / %d\n", c.Nx, c.Ny, c.Nz);
}

/*
============================================================
  已知限制 & TODO
============================================================

[1] atomicAdd 竞争（forward kernel）
    - 体素 footprint 覆盖 1~4 个像素时基本无问题
    - 若体素很小、像素很大（footprint < 1 像素），竞争激烈
    - 优化方案：将 sinogram 拆分成 tile，每个 tile 用 shared memory 累加，最后原子写回

[2] footprint AABB 近似
    - 当 volume 有旋转（非标准朝向）时，需改用8角点投影取 min/max
    - 对于标准 CBCT（volume Z ∥ det_v），当前近似误差 < 0.5%

[3] depth 因子
    - 当前 cvp_forward_kernel 里 depth = a3（体素Z方向厚度）
    - 精确值应为 a3 * |dot(vol_z_axis, srcCR)|
    - 若每个view的 srcCR 与 Z 轴夹角恒定（标准C-arm），可预计算一次

[4] back projector 与 forward 的转置关系
    - 当前 back 用 Siddon 路径长度，forward 用 footprint 面积
    - 两者不是严格数学转置（对 FDK 无影响，对 SART 迭代需注意）
    - 若需严格转置：back 也改用 voxel-driven + footprint 面积权重

[5] 多 view 并行
    - 当前是 view 串行。若 GPU 显存足够，可以：
      a. 多 stream 并行（不同 view 用不同 stream）
      b. 或将多 view 合并成一个大 kernel（grid.z = n_views）
*/

