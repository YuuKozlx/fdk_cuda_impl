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
// 工具：归一化 (用于 cuFFT 后的缩放)
// ============================================================
__global__ void normalize_kernel(float* data, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= factor;
}

// ============================================================
// 1. 正向投影 (ASTRA-style slice-driven)
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

    float src_x = -SID * c;
    float src_y = -SID * s;
    float src_z = 0.0f;

    float up = (u - (Nu - 1) * 0.5f) * du;
    float vp = (v - (Nv - 1) * 0.5f) * dv;

    float det_x = (SDD - SID) * c - up * s;
    float det_y = (SDD - SID) * s + up * c;
    float det_z = vp;

    float dxr = det_x - src_x, dyr = det_y - src_y, dzr = det_z - src_z;
    float adx = fabsf(dxr), ady = fabsf(dyr), adz = fabsf(dzr);

    int dir = (adx >= ady && adx >= adz) ? 0 : (ady >= adx && ady >= adz ? 1 : 2);
    float a1, a2, step;
    int start, end;

    if (dir == 0) { 
        float inv = 1.0f / dxr; a1 = dyr * inv; a2 = dzr * inv;
        start = 0; end = Nx; step = dx * sqrtf(1.0f + a1*a1 + a2*a2);
    } else if (dir == 1) {
        float inv = 1.0f / dyr; a1 = dxr * inv; a2 = dzr * inv;
        start = 0; end = Ny; step = dy * sqrtf(1.0f + a1*a1 + a2*a2);
    } else {
        float inv = 1.0f / dzr; a1 = dxr * inv; a2 = dyr * inv;
        start = 0; end = Nz; step = dz * sqrtf(1.0f + a1*a1 + a2*a2);
    }

    float sum = 0.0f;
    for (int s0 = start; s0 < end; ++s0) {
        float x, y, z;
        if (dir == 0) {
            x = (s0 + 0.5f) * dx - Nx * 0.5f * dx;
            y = src_y + (x - src_x) * a1; z = src_z + (x - src_x) * a2;
        } else if (dir == 1) {
            y = (s0 + 0.5f) * dy - Ny * 0.5f * dy;
            x = src_x + (y - src_y) * a1; z = src_z + (y - src_y) * a2;
        } else {
            z = (s0 + 0.5f) * dz - Nz * 0.5f * dz;
            x = src_x + (z - src_z) * a1; y = src_y + (z - src_z) * a2;
        }
        int ix = (int)(x / dx + Nx * 0.5f), iy = (int)(y / dy + Ny * 0.5f), iz = (int)(z / dz + Nz * 0.5f);
        if (ix >= 0 && ix < Nx && iy >= 0 && iy < Ny && iz >= 0 && iz < Nz)
            sum += vol[iz * Ny * Nx + iy * Nx + ix];
    }
    proj[t * Nv * Nu + v * Nu + u] = sum * step;
}

// ============================================================
// 2. FDK 预权重 (Cosine weight)
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

// ============================================================
// 3. Ramp 滤波器 (频域)
// ============================================================
__global__ void ramp_filter_kernel(cufftComplex* freq, int Nu_complex, int Nv, int Ntheta, float du) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu_complex || v >= Nv || t >= Ntheta) return;

    float f = (float)u / (float)(2 * (Nu_complex - 1)); 
    float w = f / du; 

    int idx = t * Nv * Nu_complex + v * Nu_complex + u;
    freq[idx].x *= w;
    freq[idx].y *= w;
}

// ============================================================
// 4. FDK 反投影 (Voxel-driven)
// ============================================================
__global__ void backproj_kernel(
    float* vol, int Nx, int Ny, int Nz, float dx, float dy, float dz,
    const float* proj, int Nu, int Nv, int Ntheta, float du, float dv,
    float SID, float SDD) 
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    float x = (ix - (Nx - 1) * 0.5f) * dx;
    float y = (iy - (Ny - 1) * 0.5f) * dy;
    float z = (iz - (Nz - 1) * 0.5f) * dz;

    float sum = 0.0f;
    for (int t = 0; t < Ntheta; ++t) {
        float theta = t * 2.0f * M_PI / Ntheta;
        float c = cosf(theta), s = sinf(theta);

        float xr = x * c + y * s;
        float yr = -x * s + y * c;
        float L = SID + xr;

        if (L > 1e-3f) {
            float mag = SDD / L;
            float u_idx = yr * mag / du + (Nu - 1) * 0.5f;
            float v_idx = z * mag / dv + (Nv - 1) * 0.5f;

            if (u_idx >= 0 && u_idx < Nu - 1 && v_idx >= 0 && v_idx < Nv - 1) {
                int iu = (int)u_idx, iv = (int)v_idx;
                float wu = u_idx - iu, wv = v_idx - iv;
                const float* p = proj + t * Nv * Nu;
                float val = (1-wu)*(1-wv)*p[iv*Nu+iu] + wu*(1-wv)*p[iv*Nu+iu+1] +
                            (1-wu)*wv*p[(iv+1)*Nu+iu] + wu*wv*p[(iv+1)*Nu+iu+1];
                sum += val * (SID * SID) / (L * L);
            }
        }
    }
    vol[iz * Ny * Nx + iy * Nx + ix] = sum * (2.0f * M_PI / Ntheta);
}

// ============================================================
// 5. 主函数
// ============================================================
int main() {
    // 参数设置
    int Nx=128, Ny=128, Nz=128;
    int Nu=256, Nv=256, Ntheta=360;
    float dx=1.0f, dy=1.0f, dz=1.0f, du=1.0f, dv=1.0f;
    float SID=500.0f, SDD=1000.0f;

    size_t vol_sz = Nx * Ny * Nz * sizeof(float);
    size_t proj_sz = Nu * Nv * Ntheta * sizeof(float);

    float *d_vol, *d_proj, *d_recon;
    cudaMalloc(&d_vol, vol_sz);
    cudaMalloc(&d_proj, proj_sz);
    cudaMalloc(&d_recon, vol_sz);
    cudaMemset(d_recon, 0, vol_sz);

    // --- A. 生成并保存 Phantom ---
    // --- A. 生成并保存偏心球 Phantom ---
    std::vector<float> h_phantom(Nx * Ny * Nz, 0.0f);

    // 定义球体参数：{x_center, y_center, z_center, radius, intensity}
    struct Sphere { float x, y, z, r, i; };
    std::vector<Sphere> spheres = {
        {0, 0, 0, 20, 0.5f},      // 中心大球（半透明）
        {30, 0, 0, 10, 1.0f},     // X轴偏移小球
        {0, 30, 0, 8, 0.8f},      // Y轴偏移小球
        {-20, -20, 25, 12, 0.6f}, // 空间偏移球（Z轴偏移）
        {0, 0, -35, 5, 1.0f}      // 底部小球
    };

    for(int z=0; z<Nz; z++) {
        for(int y=0; y<Ny; y++) {
            for(int x=0; x<Nx; x++) {
                // 将索引转换为物理坐标 (相对于体积中心)
                float vx = (x - (Nx - 1) * 0.5f) * dx;
                float vy = (y - (Ny - 1) * 0.5f) * dy;
                float vz = (z - (Nz - 1) * 0.5f) * dz;

                for(const auto& s : spheres) {
                    float distSq = (vx - s.x)*(vx - s.x) + (vy - s.y)*(vy - s.y) + (vz - s.z)*(vz - s.z);
                    if (distSq < s.r * s.r) {
                        h_phantom[z * Ny * Nx + y * Nx + x] += s.i; 
                    }
                }
            }
        }
    }
    cudaMemcpy(d_vol, h_phantom.data(), vol_sz, cudaMemcpyHostToDevice);
    FILE* f1 = fopen("phantom_original.raw", "wb");
    fwrite(h_phantom.data(), sizeof(float), h_phantom.size(), f1);
    fclose(f1);
    printf("Phantom saved: phantom_original.raw\n");

    // --- B. 正向投影 ---
    dim3 block(8, 8, 8);
    dim3 gridP((Nu+7)/8, (Nv+7)/8, (Ntheta+7)/8);
    dim3 gridV((Nx+7)/8, (Ny+7)/8, (Nz+7)/8);

    printf("Forward Projecting...\n");
    astra_projection_kernel<<<gridP, block>>>(d_vol, d_proj, Nx, Ny, Nz, dx, dy, dz, Nu, Nv, Ntheta, du, dv, SID, SDD);

    // 保存投影数据
    std::vector<float> h_proj(Nu * Nv * Ntheta);
    cudaMemcpy(h_proj.data(), d_proj, proj_sz, cudaMemcpyDeviceToHost);
    FILE* f2 = fopen("projection.raw", "wb");
    fwrite(h_proj.data(), sizeof(float), h_proj.size(), f2);
    fclose(f2);
    printf("Projection saved: projection.raw\n");

    // --- C. FDK 滤波链条 ---
    printf("Filtering...\n");
    fdk_preweight_kernel<<<gridP, block>>>(d_proj, Nu, Nv, Ntheta, du, dv, SID);

    int Nu_complex = Nu / 2 + 1;
    cufftHandle plan_fwd, plan_inv;
    cufftPlanMany(&plan_fwd, 1, &Nu, NULL, 1, Nu, NULL, 1, Nu_complex, CUFFT_R2C, Nv * Ntheta);
    cufftPlanMany(&plan_inv, 1, &Nu, NULL, 1, Nu_complex, NULL, 1, Nu, CUFFT_C2R, Nv * Ntheta);

    cufftComplex* d_freq;
    cudaMalloc(&d_freq, Ntheta * Nv * Nu_complex * sizeof(cufftComplex));

    cufftExecR2C(plan_fwd, (cufftReal*)d_proj, d_freq);
    dim3 gridF((Nu_complex+7)/8, (Nv+7)/8, (Ntheta+7)/8);
    ramp_filter_kernel<<<gridF, block>>>(d_freq, Nu_complex, Nv, Ntheta, du);
    cufftExecC2R(plan_inv, d_freq, (cufftReal*)d_proj);

    normalize_kernel<<<(Nu*Nv*Ntheta+255)/256, 256>>>(d_proj, Nu*Nv*Ntheta, 1.0f/Nu);

    // --- D. 反投影重建 ---
    printf("Backprojecting...\n");
    backproj_kernel<<<gridV, block>>>(d_recon, Nx, Ny, Nz, dx, dy, dz, d_proj, Nu, Nv, Ntheta, du, dv, SID, SDD);

    // 保存重建体积
    std::vector<float> h_recon(Nx * Ny * Nz);
    cudaMemcpy(h_recon.data(), d_recon, vol_sz, cudaMemcpyDeviceToHost);
    FILE* f3 = fopen("reconstruction_fdk.raw", "wb");
    fwrite(h_recon.data(), sizeof(float), h_recon.size(), f3);
    fclose(f3);
    printf("Reconstruction saved: reconstruction_fdk.raw\n");

    // 清理
    cufftDestroy(plan_fwd); cufftDestroy(plan_inv);
    cudaFree(d_vol); cudaFree(d_proj); cudaFree(d_recon); cudaFree(d_freq);
    printf("Success!\n");

    return 0;
}