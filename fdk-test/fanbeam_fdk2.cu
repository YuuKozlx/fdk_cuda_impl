#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================
// 0. 工具：归一化 (用于修正 cuFFT 缩放)
// ============================================================
__global__ void normalize_kernel(float* data, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ============================================================
// 1. 向量化反投影 Kernel (FDK Voxel-Driven)
// ============================================================
__global__ void backproj_vector_kernel(
    float* vol, int Nx, int Ny, int Nz, float dx, float dy, float dz,
    const float* proj, int Nu, int Nv, int Ntheta,
    const float* geom) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    // 像素 P 的物理位置
    float Px = (ix - (Nx - 1) * 0.5f) * dx;
    float Py = (iy - (Ny - 1) * 0.5f) * dy;
    float Pz = (iz - (Nz - 1) * 0.5f) * dz;

    float sum = 0.0f;

    for (int t = 0; t < Ntheta; ++t) {
        // 获取当前角度的 12 个几何向量参数
        const float* g = &geom[t * 12];
        float sX = g[0], sY = g[1], sZ = g[2]; // Source
        float dX = g[3], dY = g[4], dZ = g[5]; // Detector Center
        float uX = g[6], uY = g[7], uZ = g[8]; // Unit U
        float vX = g[9], vY = g[10], vZ = g[11]; // Unit V

        // 1. 射线向量 L = P - S
        float Lx = Px - sX;
        float Ly = Py - sY;
        float Lz = Pz - sZ;

        // 2. 计算探测器法向量 n = u x v
        float nX = uY * vZ - uZ * vY;
        float nY = uZ * vX - uX * vZ;
        float nZ = uX * vY - uY * vX;

        // 3. 计算射线与平面的交点比例 alpha = ((D-S).n) / (L.n)
        float num = (dX - sX) * nX + (dY - sY) * nY + (dZ - sZ) * nZ;
        float den = Lx * nX + Ly * nY + Lz * nZ;

        if (fabsf(den) < 1e-6f) continue;
        float alpha = num / den;
        if (alpha < 0) continue; // 确保在探测器前方

        // 4. 交点相对中心向量 Q = (S + alpha*L) - D
        float QX = sX + alpha * Lx - dX;
        float QY = sY + alpha * Ly - dY;
        float QZ = sZ + alpha * Lz - dZ;

        // 5. 将 Q 投影到 u 和 v 轴获得像素索引
        float magU2 = uX*uX + uY*uY + uZ*uZ;
        float magV2 = vX*vX + vY*vY + vZ*vZ;
        float u_coord = (QX * uX + QY * uY + QZ * uZ) / magU2;
        float v_coord = (QX * vX + QY * vY + QZ * vZ) / magV2;

        float u_idx = u_coord + (Nu - 1) * 0.5f;
        float v_idx = v_coord + (Nv - 1) * 0.5f;

        // 6. 双线性插值
        if (u_idx >= 0 && u_idx < Nu - 1 && v_idx >= 0 && v_idx < Nv - 1) {
            int iu = (int)u_idx, iv = (int)v_idx;
            float wu = u_idx - iu, wv = v_idx - iv;
            const float* p = proj + t * Nv * Nu;
            float val = (1-wu)*(1-wv)*p[iv*Nu+iu] + wu*(1-wv)*p[iv*Nu+iu+1] +
                        (1-wu)*wv*p[(iv+1)*Nu+iu] + wu*wv*p[(iv+1)*Nu+iu+1];

            // FDK 距离平方反比权重
            sum += val / (alpha * alpha);
        }
    }
    vol[iz * Ny * Nx + iy * Nx + ix] = sum * (2.0f * M_PI / Ntheta);
}

// ============================================================
// 2. 正向投影 Kernel (沿用 slice-driven 逻辑)
// ============================================================
__global__ void astra_projection_kernel(
    const float* vol, float* proj,
    int Nx, int Ny, int Nz, float dx, float dy, float dz,
    int Nu, int Nv, int Ntheta, float du, float dv,
    float SID, float SDD) 
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu || v >= Nv || t >= Ntheta) return;

    float theta = t * 2.0f * M_PI / Ntheta;
    float c = cosf(theta), s = sinf(theta);
    float src_x = -SID * c, src_y = -SID * s, src_z = 0.0f;
    float up = (u - (Nu - 1) * 0.5f) * du, vp = (v - (Nv - 1) * 0.5f) * dv;
    float det_x = (SDD - SID) * c - up * s, det_y = (SDD - SID) * s + up * c, det_z = vp;

    float dxr = det_x - src_x, dyr = det_y - src_y, dzr = det_z - src_z;
    float step_len = sqrtf(dxr*dxr + dyr*dyr + dzr*dzr);
    float vx = dxr / step_len, vy = dyr / step_len, vz = dzr / step_len;

    float sum = 0.0f;
    float current_t = 0.0f;
    while(current_t < step_len * 1.5f) { // 简单射线步进
        float px = src_x + vx * current_t;
        float py = src_y + vy * current_t;
        float pz = src_z + vz * current_t;
        int ix = (int)(px / dx + Nx * 0.5f);
        int iy = (int)(py / dy + Ny * 0.5f);
        int iz = (int)(pz / dz + Nz * 0.5f);
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz)
            sum += vol[iz * Ny * Nx + iy * Nx + ix];
        current_t += 0.5f; // 步长
    }
    proj[t * Nv * Nu + v * Nu + u] = sum * 0.5f;
}

// ============================================================
// 3. FDK 滤波器 Kernels
// ============================================================
__global__ void fdk_preweight_kernel(float* proj, int Nu, int Nv, int Ntheta, float du, float dv, float SID) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu || v >= Nv || t >= Ntheta) return;
    float up = (u - (Nu - 1) * 0.5f) * du;
    float vp = (v - (Nv - 1) * 0.5f) * dv;
    float w = SID / sqrtf(SID*SID + up*up + vp*vp);
    proj[t * Nv * Nu + v * Nu + u] *= w;
}

__global__ void ramp_filter_kernel(cufftComplex* freq, int Nu_c, int Nv, int Ntheta, float du) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu_c || v >= Nv || t >= Ntheta) return;
    float f = (float)u / (float)(2 * (Nu_c - 1)); 
    float w = f / du; 
    int idx = t * Nv * Nu_c + v * Nu_c + u;
    freq[idx].x *= w; freq[idx].y *= w;
}

// ============================================================
// 4. MAIN 函数 (全写出)
// ============================================================
int main() {
    // 基础维度参数
    int Nx=128, Ny=128, Nz=128;
    int Nu=256, Nv=256, Ntheta=360;
    float dx=1.0f, dy=1.0f, dz=1.0f, du=1.0f, dv=1.0f;
    float SID=500.0f, SDD=1000.0f;

    // --- 1. 初始化几何向量 (Vector Geometry) ---
    std::vector<float> h_geom(Ntheta * 12);
    for (int t = 0; t < Ntheta; ++t) {
        float angle = t * 2.0f * M_PI / Ntheta;
        float c = cosf(angle), s = sinf(angle);
        // Source S
        h_geom[t*12+0]=-SID*c; h_geom[t*12+1]=-SID*s; h_geom[t*12+2]=0;
        // Detector Center D
        h_geom[t*12+3]=(SDD-SID)*c; h_geom[t*12+4]=(SDD-SID)*s; h_geom[t*12+5]=0;
        // Vector U
        h_geom[t*12+6]=-du*s; h_geom[t*12+7]=du*c; h_geom[t*12+8]=0;
        // Vector V
        h_geom[t*12+9]=0; h_geom[t*12+10]=0; h_geom[t*12+11]=dv;
    }
    float *d_geom; cudaMalloc(&d_geom, Ntheta*12*sizeof(float));
    cudaMemcpy(d_geom, h_geom.data(), Ntheta*12*sizeof(float), cudaMemcpyHostToDevice);

    // --- 2. 内存分配与 Phantom 生成 ---
    float *d_vol, *d_proj, *d_recon;
    cudaMalloc(&d_vol, Nx*Ny*Nz*sizeof(float));
    cudaMalloc(&d_proj, Nu*Nv*Ntheta*sizeof(float));
    cudaMalloc(&d_recon, Nx*Ny*Nz*sizeof(float));
    cudaMemset(d_recon, 0, Nx*Ny*Nz*sizeof(float));

    std::vector<float> h_phantom(Nx*Ny*Nz, 0.0f);
    for(int z=0; z<Nz; z++)
        for(int y=0; y<Ny; y++)
            for(int x=0; x<Nx; x++) {
                float vx=(x-63.5f), vy=(y-63.5f), vz=(z-63.5f);
                if (sqrtf((vx-20)*(vx-20)+vy*vy+vz*vz) < 15) h_phantom[z*Ny*Nx+y*Nx+x]=1.0f; // 偏心球
            }
    cudaMemcpy(d_vol, h_phantom.data(), Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);

    // --- 3. 执行流程 ---
    dim3 block(8,8,8);
    dim3 gridP((Nu+7)/8, (Nv+7)/8, (Ntheta+7)/8);
    dim3 gridV((Nx+7)/8, (Ny+7)/8, (Nz+7)/8);

    printf("Step 1: Forward Projecting...\n");
    astra_projection_kernel<<<gridP, block>>>(d_vol, d_proj, Nx, Ny, Nz, dx, dy, dz, Nu, Nv, Ntheta, du, dv, SID, SDD);

    printf("Step 2: Pre-weighting...\n");
    fdk_preweight_kernel<<<gridP, block>>>(d_proj, Nu, Nv, Ntheta, du, dv, SID);

    printf("Step 3: FFT Filtering...\n");
    int Nu_c = Nu / 2 + 1;
    cufftHandle pf, pi;
    cufftPlanMany(&pf, 1, &Nu, NULL, 1, Nu, NULL, 1, Nu_c, CUFFT_R2C, Nv * Ntheta);
    cufftPlanMany(&pi, 1, &Nu, NULL, 1, Nu_c, NULL, 1, Nu, CUFFT_C2R, Nv * Ntheta);
    cufftComplex* d_freq; cudaMalloc(&d_freq, Ntheta * Nv * Nu_c * sizeof(cufftComplex));

    cufftExecR2C(pf, (cufftReal*)d_proj, d_freq);
    dim3 gridF((Nu_c+7)/8, (Nv+7)/8, (Ntheta+7)/8);
    ramp_filter_kernel<<<gridF, block>>>(d_freq, Nu_c, Nv, Ntheta, du);
    cufftExecC2R(pi, d_freq, (cufftReal*)d_proj);
    normalize_kernel<<<(Nu*Nv*Ntheta+255)/256, 256>>>(d_proj, Nu*Nv*Ntheta, 1.0f/Nu);

    printf("Step 4: Vector Backprojection...\n");
    backproj_vector_kernel<<<gridV, block>>>(d_recon, Nx, Ny, Nz, dx, dy, dz, d_proj, Nu, Nv, Ntheta, d_geom);

    // --- 4. 数据保存 ---
    std::vector<float> h_res(Nx*Ny*Nz);
    cudaMemcpy(h_res.data(), d_recon, Nx*Ny*Nz*sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen("vector_fdk_final.raw", "wb");
    fwrite(h_res.data(), sizeof(float), h_res.size(), f);
    fclose(f);

    printf("Done. Saved to vector_fdk_final.raw\n");

    // 清理
    cufftDestroy(pf); cufftDestroy(pi);
    cudaFree(d_vol); cudaFree(d_proj); cudaFree(d_recon); cudaFree(d_geom); cudaFree(d_freq);
    return 0;
}