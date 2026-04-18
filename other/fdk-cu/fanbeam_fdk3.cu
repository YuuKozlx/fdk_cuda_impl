#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#include "../YkGlobals.h"

//// 几何结构定义
//struct SConeProjection {
//    // the source
//    double fSrcX, fSrcY, fSrcZ;
//
//    // the origin ("bottom left") of the (flat-panel) detector
//    double fDetSX, fDetSY, fDetSZ;
//
//    // the U-edge of a detector pixel
//    double fDetUX, fDetUY, fDetUZ;
//
//    // the V-edge of a detector pixel
//    double fDetVX, fDetVY, fDetVZ;
//};
//
//struct SDimensions3D {
//    unsigned int iVolX;
//    unsigned int iVolY;
//    unsigned int iVolZ;
//    unsigned int iProjAngles;
//    unsigned int iProjU; // number of detectors in the U direction
//    unsigned int iProjV; // number of detectors in the V direction
//};

// 1. 三线性插值
__device__ float trilinear_interpolate(const float* vol, float x, float y, float z, int Nx, int Ny, int Nz) {
    float ix = x + (Nx - 1) * 0.5f;
    float iy = y + (Ny - 1) * 0.5f;
    float iz = z + (Nz - 1) * 0.5f;
    int x0 = (int)floorf(ix), y0 = (int)floorf(iy), z0 = (int)floorf(iz);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    if (x0 < 0 || x1 >= Nx || y0 < 0 || y1 >= Ny || z0 < 0 || z1 >= Nz) return 0.0f;
    float dx = ix - x0, dy = iy - y0, dz = iz - z0;
    float v000 = vol[z0 * Ny * Nx + y0 * Nx + x0], v100 = vol[z0 * Ny * Nx + y0 * Nx + x1];
    float v010 = vol[z0 * Ny * Nx + y1 * Nx + x0], v110 = vol[z0 * Ny * Nx + y1 * Nx + x1];
    float v001 = vol[z1 * Ny * Nx + y0 * Nx + x0], v101 = vol[z1 * Ny * Nx + y0 * Nx + x1];
    float v011 = vol[z1 * Ny * Nx + y1 * Nx + x0], v111 = vol[z1 * Ny * Nx + y1 * Nx + x1];
    return (1-dx)*(1-dy)*(1-dz)*v000 + dx*(1-dy)*(1-dz)*v100 + (1-dx)*dy*(1-dz)*v010 + dx*dy*(1-dz)*v110 +
           (1-dx)*(1-dy)*dz*v001 + dx*(1-dy)*dz*v101 + (1-dx)*dy*dz*v011 + dx*dy*dz*v111;
}

// 2. 正向投影内核
__global__ void astra_fp_kernel(const float* vol, float* proj, int Nx, int Ny, int Nz, int Nu, int Nv, int Ntheta, const SConeProjection* geom) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu || v >= Nv || t >= Ntheta) return;
    SConeProjection g = geom[t];
    float fDetX = g.fDetSX + u * g.fDetUX + v * g.fDetVX;
    float fDetY = g.fDetSY + u * g.fDetUY + v * g.fDetVY;
    float fDetZ = g.fDetSZ + u * g.fDetUZ + v * g.fDetVZ;
    float dx = fDetX - g.fSrcX, dy = fDetY - g.fSrcY, dz = fDetZ - g.fSrcZ;
    float adx = fabsf(dx), ady = fabsf(dy), adz = fabsf(dz), sum = 0.0f, step_len = 0.0f;
    if (adx >= ady && adx >= adz) {
        float ay = dy / dx, az = dz / dx; step_len = sqrtf(1.0f + ay*ay + az*az);
        for (int i = 0; i < Nx; ++i) { float curX = (i - (Nx-1)*0.5f); float alpha = (curX - g.fSrcX) / dx;
            sum += trilinear_interpolate(vol, curX, g.fSrcY + alpha*dy, g.fSrcZ + alpha*dz, Nx, Ny, Nz); }
    } else if (ady >= adx && ady >= adz) {
        float ax = dx / dy, az = dz / dy; step_len = sqrtf(1.0f + ax*ax + az*az);
        for (int i = 0; i < Ny; ++i) { float curY = (i - (Ny-1)*0.5f); float alpha = (curY - g.fSrcY) / dy;
            sum += trilinear_interpolate(vol, g.fSrcX + alpha*dx, curY, g.fSrcZ + alpha*dz, Nx, Ny, Nz); }
    } else {
        float ax = dx / dz, ay = dy / dz; step_len = sqrtf(1.0f + ax*ax + ay*ay);
        for (int i = 0; i < Nz; ++i) { float curZ = (i - (Nz-1)*0.5f); float alpha = (curZ - g.fSrcZ) / dz;
            sum += trilinear_interpolate(vol, g.fSrcX + alpha*dx, g.fSrcY + alpha*dy, curZ, Nx, Ny, Nz); }
    }
    proj[t * Nu * Nv + v * Nu + u] = sum * step_len;
}

// 3. FDK 预权重修正
__global__ void fdk_preweight_kernel(float* proj,
    const SDimensions3D dims,
    const float SID, const float SDD,
    const float du, const float dv) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;

    int Nu = dims.iProjU, Nv = dims.iProjV, Ntheta = dims.iProjAngles;
    if (u >= Nu || v >= Nv || t >= Ntheta) return;

    // 1. 计算当前像素相对于探测器中心的物理坐标 (假设中心对称)
    float u_pos = (u - (Nu - 1) / 2.0f) * du;
    float v_pos = (v - (Nv - 1) / 2.0f) * dv;

    // 2. 计算源到探测器像素的距离 L
    // 在理想几何下：L = sqrt(SDD^2 + u_pos^2 + v_pos^2)
    float L = sqrtf(SDD * SDD + u_pos * u_pos + v_pos * v_pos);

    // 3. FDK 预加权公式: Proj = Proj * (SID / L)
    int idx = t * Nu * Nv + v * Nu + u;
    proj[idx] *= (SID / L);
}

// 4. 归一化滤波内核 (修正了 scaling 和量纲)
__global__ void ramp_filter_kernel(cufftComplex* freq, int Nu_complex, int Nv, int Ntheta, float du, int Nu) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;

    if (u >= Nu_complex || v >= Nv || t >= Ntheta) return;

    // 1. 频率映射：u 是从 0 到 Nu/2 的离散索引
    // 归一化频率 f_norm = u / Nu, 物理频率 f = u / (Nu * du)
    float f = (float)u / (float)(Nu * du);

    // 2. 窗口函数 (可选): 为了抑制高频噪声，通常会加一个 Hamming 或 Hann 窗
    // 这里仅实现纯 Ram-Lak: |f|
    // 注意：cuFFT 的 C2R 变换需要乘以 1/Nu 归一化，由于我们在频域操作，直接乘进去
    float w = f ;

    // 3. 索引计算
    int idx = (t * Nv + v) * Nu_complex + u;

    freq[idx].x *= w;
    freq[idx].y *= w;
}

// 5. 反投影内核 (修正了权重累加系数)
__global__ void astra_bp_kernel(float* vol, int Nx, int Ny, int Nz,
    const float* proj, int Nu, int Nv, int Ntheta,
    const SConeProjectionVec* geom,
    const float SID, const float voxel_size) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    // 1. 引入物理尺寸 (以中心为原点)
    float Px = (ix - (Nx - 1) * 0.5f) * voxel_size;
    float Py = (iy - (Ny - 1) * 0.5f) * voxel_size;
    float Pz = (iz - (Nz - 1) * 0.5f) * voxel_size;

    float sum = 0.0f;

    for (int t = 0; t < Ntheta; ++t) {
        SConeProjection g = geom[t];

        // 射线向量: 源 -> 体素
        float Lx = Px - g.fSrcX;
        float Ly = Py - g.fSrcY;
        float Lz = Pz - g.fSrcZ;

        // 探测器法线 (建议在外部预计算并直接存入 g)
        float nX = g.fDetUY * g.fDetVZ - g.fDetUZ * g.fDetVY;
        float nY = g.fDetUZ * g.fDetVX - g.fDetUX * g.fDetVZ;
        float nZ = g.fDetUX * g.fDetVY - g.fDetUY * g.fDetVX;

        // U: 射线向量在法线方向的投射长度
        float U = Lx * nX + Ly * nY + Lz * nZ;
        if (fabsf(U) < 1e-6f) continue;

        // alpha: 射线与平面相交的比例系数
        float alpha = ((g.fDetSX - g.fSrcX) * nX + (g.fDetSY - g.fSrcY) * nY + (g.fDetSZ - g.fSrcZ) * nZ) / U;

        // Q: 探测器平面上的交点坐标 (相对于探测器原点 S)
        float QX = g.fSrcX + alpha * Lx - g.fDetSX;
        float QY = g.fSrcY + alpha * Ly - g.fDetSY;
        float QZ = g.fSrcZ + alpha * Lz - g.fDetSZ;

        // 映射到 U, V 索引
        float u_idx = (QX * g.fDetUX + QY * g.fDetUY + QZ * g.fDetUZ) / (g.fDetUX * g.fDetUX + g.fDetUY * g.fDetUY + g.fDetUZ * g.fDetUZ);
        float v_idx = (QX * g.fDetVX + QY * g.fDetVY + QZ * g.fDetVZ) / (g.fDetVX * g.fDetVX + g.fDetVY * g.fDetVY + g.fDetVZ * g.fDetVZ);

        if (u_idx >= 0 && u_idx < Nu - 1 && v_idx >= 0 && v_idx < Nv - 1) {
            int iu = (int)u_idx, iv = (int)v_idx;
            float wu = u_idx - iu, wv = v_idx - iv;

            const float* p = proj + t * Nv * Nu;

            // 双线性插值
            float val = (1 - wu) * (1 - wv) * p[iv * Nu + iu] +
                wu * (1 - wv) * p[iv * Nu + iu + 1] +
                (1 - wu) * wv * p[(iv + 1) * Nu + iu] +
                wu * wv * p[(iv + 1) * Nu + iu + 1];

            // FDK 权重修正: (SID / U_virtual)^2
            // 这里 U 实际上跟源到体素的距离相关，alpha 是缩放比例
            // 标准 FDK 权重通常使用源到体素投影点的距离
            float dep_weight = SID / (alpha * sqrtf(Lx * Lx + Ly * Ly + Lz * Lz));
            sum += val * (dep_weight * dep_weight);
        }
    }

    vol[iz * Ny * Nx + iy * Nx + ix] = sum * ( M_PI / (float)Ntheta)/10;
}

void save_raw(const char* filename, const std::vector<float>& data) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(data.data(), sizeof(float), data.size(), f);
        fclose(f);
        printf("Saved: %s\n", filename);
    }
}

int main00() {
    // 1. 参数定义
    SDimensions3D dims;
    dims.iProjU = 256; dims.iProjV = 256; dims.iProjAngles = 360;
    dims.iVolX = 512;  dims.iVolY = 512,dims.iVolZ = 100;

    float SID = 500.0f;
    float SDD = 1000.0f;
    float du = 1.0f;        // 探测器像素尺寸
    float dv = 1.0f;
    float voxel_size = 0.5f; // 重建体素尺寸

    size_t proj_elements = dims.iProjU * dims.iProjV * dims.iProjAngles;
    size_t vol_elements = dims.iVolX * dims.iVolY * dims.iVolZ;

    // 2. 分配主机内存并读取数据
    std::vector<float> h_proj(proj_elements);
    FILE* fp = fopen("cat515_projection.raw", "rb");
    if (!fp) { printf("Error: Cannot find proj.raw\n"); return -1; }
    fread(h_proj.data(), sizeof(float), proj_elements, fp);
    fclose(fp);

    // 3. 分配 GPU 内存
    float* d_proj, * d_vol;
    cudaMalloc(&d_proj, proj_elements * sizeof(float));
    cudaMalloc(&d_vol, vol_elements * sizeof(float));
    cudaMemcpy(d_proj, h_proj.data(), proj_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_vol, 0, vol_elements * sizeof(float));

    // 4. 构建几何轨迹 (圆周)
    std::vector<SConeProjection> h_geom(dims.iProjAngles);
    for (unsigned int t = 0; t < dims.iProjAngles; ++t) {
        float angle = t * (2.0f * M_PI / dims.iProjAngles);
        // 源位置
        h_geom[t].fSrcX = SID * cosf(angle);
        h_geom[t].fSrcY = SID * sinf(angle);
        h_geom[t].fSrcZ = 0;

        // 探测器 U, V 向量 (旋转坐标系)
        h_geom[t].fDetUX = -sinf(angle); h_geom[t].fDetUY = cosf(angle); h_geom[t].fDetUZ = 0;
        h_geom[t].fDetVX = 0; h_geom[t].fDetVY = 0; h_geom[t].fDetVZ = 1.0;

        // 探测器中心位置 (在源的对面)
        float det_center_x = -(SDD - SID) * cosf(angle);
        float det_center_y = -(SDD - SID) * sinf(angle);

        // 计算探测器左下角 S = Center - (U*halfWidth) - (V*halfHeight)
        h_geom[t].fDetSX = det_center_x - (dims.iProjU - 1) * 0.5f * h_geom[t].fDetUX - (dims.iProjV - 1) * 0.5f * h_geom[t].fDetVX;
        h_geom[t].fDetSY = det_center_y - (dims.iProjU - 1) * 0.5f * h_geom[t].fDetUY - (dims.iProjV - 1) * 0.5f * h_geom[t].fDetVY;
        h_geom[t].fDetSZ = 0 - (dims.iProjV - 1) * 0.5f * h_geom[t].fDetVZ;
    }
    SConeProjectionVec* d_geom;
    cudaMalloc(&d_geom, dims.iProjAngles * sizeof(SConeProjectionVec));
    cudaMemcpy(d_geom, h_geom.data(), dims.iProjAngles * sizeof(SConeProjectionVec), cudaMemcpyHostToDevice);

    // ==========================================================
    // 第一步：预加权
    // ==========================================================
    dim3 blockP(16, 16, 1);
    dim3 gridP((dims.iProjU + 15) / 16, (dims.iProjV + 15) / 16, dims.iProjAngles);
    fdk_preweight_kernel << <gridP, blockP >> > (d_proj, dims, SID, SDD, du, dv);
    cudaDeviceSynchronize();

    // 保存预加权结果
    cudaMemcpy(h_proj.data(), d_proj, proj_elements * sizeof(float), cudaMemcpyDeviceToHost);
    save_raw("preweighted.raw", h_proj);

    // ==========================================================
    // 第二步：频谱滤波 (R2C -> Kernel -> C2R)
    // ==========================================================
    int Nu_complex = dims.iProjU / 2 + 1;
    int total_rows = dims.iProjV * dims.iProjAngles;
    cufftHandle plan_fwd, plan_inv;
    cufftPlanMany(&plan_fwd, 1, (int*)&dims.iProjU, NULL, 1, dims.iProjU, NULL, 1, dims.iProjU, CUFFT_R2C, total_rows);
    cufftPlanMany(&plan_inv, 1, (int*)&dims.iProjU, NULL, 1, dims.iProjU, NULL, 1, dims.iProjU, CUFFT_C2R, total_rows);

    cufftComplex* d_freq;
    cudaMalloc(&d_freq, Nu_complex * total_rows * sizeof(cufftComplex));

    cufftExecR2C(plan_fwd, (cufftReal*)d_proj, d_freq);

    dim3 blockF(16, 8, 8);
    dim3 gridF((Nu_complex + 15) / 16, (dims.iProjV + 7) / 8, (dims.iProjAngles + 7) / 8);
    ramp_filter_kernel << <gridF, blockF >> > (d_freq, Nu_complex, dims.iProjV, dims.iProjAngles, du, dims.iProjU);

    cufftExecC2R(plan_inv, d_freq, (cufftReal*)d_proj);
    cudaDeviceSynchronize();

    // 保存滤波结果
    cudaMemcpy(h_proj.data(), d_proj, proj_elements * sizeof(float), cudaMemcpyDeviceToHost);
    save_raw("filtered.raw", h_proj);

    // ==========================================================
    // 第三步：反投影
    // ==========================================================
    dim3 blockB(8, 8, 8);
    dim3 gridB((dims.iVolX + 7) / 8, (dims.iVolY + 7) / 8, (dims.iVolZ + 7) / 8);
    astra_bp_kernel << <gridB, blockB >> > (d_vol, dims.iVolX, dims.iVolY, dims.iVolZ, d_proj, dims.iProjU, dims.iProjV, dims.iProjAngles, d_geom, SID, voxel_size);
    cudaDeviceSynchronize();

    // 保存重建结果
    std::vector<float> h_vol(vol_elements);
    cudaMemcpy(h_vol.data(), d_vol, vol_elements * sizeof(float), cudaMemcpyDeviceToHost);
    save_raw("reconstruction.raw", h_vol);

    // 5. 释放资源
    cufftDestroy(plan_fwd); cufftDestroy(plan_inv);
    cudaFree(d_proj); cudaFree(d_vol); cudaFree(d_geom); cudaFree(d_freq);

    printf("FDK Reconstruction Pipeline Finished.\n");
    return 0;
}