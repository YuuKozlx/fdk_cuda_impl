#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// 几何结构定义
struct SConeProjection {
    float fSrcX, fSrcY, fSrcZ;
    float fDetSX, fDetSY, fDetSZ;
    float fDetUX, fDetUY, fDetUZ;
    float fDetVX, fDetVY, fDetVZ;
};

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
__global__ void fdk_preweight_kernel(float* proj, int Nu, int Nv, int Ntheta, const SConeProjection* geom, const float SID) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu || v >= Nv || t >= Ntheta) return;

    SConeProjection g = geom[t];
    float pX = g.fDetSX + u*g.fDetUX + v*g.fDetVX;
    float pY = g.fDetSY + u*g.fDetUY + v*g.fDetVY;
    float pZ = g.fDetSZ + u*g.fDetUZ + v*g.fDetVZ;
    float L = sqrtf(powf(pX-g.fSrcX, 2) + powf(pY-g.fSrcY, 2) + powf(pZ-g.fSrcZ, 2));

    proj[t * Nu * Nv + v * Nu + u] *= (SID / L);
}

// 4. 归一化滤波内核 (修正了 scaling 和量纲)
__global__ void ramp_filter_kernel(cufftComplex* freq, int Nu_complex, int Nv, int Ntheta, float du, int Nu) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.z * blockDim.z + threadIdx.z;
    if (u >= Nu_complex || v >= Nv || t >= Ntheta) return;

    // 归一化频率 [0, 0.5]
    float f = (float)u / (float)Nu; 
    // Ramp: |f| / (2*du)，同时除以 Nu 以抵消 cuFFT 的增益
    float w = (f / (2.0f * du)) / (float)Nu; 

    int idx = t * Nv * Nu_complex + v * Nu_complex + u;
    freq[idx].x *= w;
    freq[idx].y *= w;
}

// 5. 反投影内核 (修正了权重累加系数)
__global__ void astra_bp_kernel(float* vol, int Nx, int Ny, int Nz, const float* proj, int Nu, int Nv, int Ntheta, const SConeProjection* geom, const float SID) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;

    float Px = (ix - (Nx - 1) * 0.5f), Py = (iy - (Ny - 1) * 0.5f), Pz = (iz - (Nz - 1) * 0.5f);
    float sum = 0.0f;

    for (int t = 0; t < Ntheta; ++t) {
        SConeProjection g = geom[t];
        float Lx = Px - g.fSrcX, Ly = Py - g.fSrcY, Lz = Pz - g.fSrcZ;
        
        // 计算探测器法线 (U x V)
        float nX = g.fDetUY*g.fDetVZ - g.fDetUZ*g.fDetVY;
        float nY = g.fDetUZ*g.fDetVX - g.fDetUX*g.fDetVZ;
        float nZ = g.fDetUX*g.fDetVY - g.fDetUY*g.fDetVX;
        float norm_l = sqrtf(nX*nX + nY*nY + nZ*nZ);
        nX /= norm_l; nY /= norm_l; nZ /= norm_l;

        float U = Lx*nX + Ly*nY + Lz*nZ;
        float alpha = ((g.fDetSX-g.fSrcX)*nX + (g.fDetSY-g.fSrcY)*nY + (g.fDetSZ-g.fSrcZ)*nZ) / U;
        float QX = g.fSrcX + alpha*Lx - g.fDetSX;
        float QY = g.fSrcY + alpha*Ly - g.fDetSY;
        float QZ = g.fSrcZ + alpha*Lz - g.fDetSZ;
        float dep_weight = SID / U;

        float magU2 = g.fDetUX*g.fDetUX + g.fDetUY*g.fDetUY + g.fDetUZ*g.fDetUZ;
        float magV2 = g.fDetVX*g.fDetVX + g.fDetVY*g.fDetVY + g.fDetVZ*g.fDetVZ;
        float u_idx = (QX*g.fDetUX + QY*g.fDetUY + QZ*g.fDetUZ) / magU2;
        float v_idx = (QX*g.fDetVX + QY*g.fDetVY + QZ*g.fDetVZ) / magV2;

        if (u_idx >= 0 && u_idx < Nu-1 && v_idx >= 0 && v_idx < Nv-1) {
            int iu = (int)u_idx, iv = (int)v_idx;
            float wu = u_idx-iu, wv = v_idx-iv;
            const float* p = proj + t*Nv*Nu;
            float val = (1-wu)*(1-wv)*p[iv*Nu+iu] + wu*(1-wv)*p[iv*Nu+iu+1] + (1-wu)*wv*p[(iv+1)*Nu+iu] + wu*wv*p[(iv+1)*Nu+iu+1];
            
            // FDK 距离平方反比加权
  
            sum += val * dep_weight * dep_weight;
        }
    }

    vol[iz * Ny * Nx + iy * Nx + ix] = sum * (2.0f * M_PI / (float)Ntheta);
}

void save_raw(const char* filename, const std::vector<float>& data) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(data.data(), sizeof(float), data.size(), f);
        fclose(f);
        printf("Saved: %s\n", filename);
    }
}

int main() {
    int Nx=128, Ny=128, Nz=128, Nu=256, Nv=256, Ntheta=360;
    float SID=500.0f, SDD=1000.0f, du=1.0f, dv=1.0f;

    // 1. 初始化几何
    std::vector<SConeProjection> h_geom(Ntheta);
    for (int t = 0; t < Ntheta; ++t) {
        float a = t * 2.0f * M_PI / Ntheta;
        h_geom[t].fSrcX = -SID*cosf(a); h_geom[t].fSrcY = -SID*sinf(a); h_geom[t].fSrcZ = 0;
        h_geom[t].fDetUX = -du*sinf(a); h_geom[t].fDetUY = du*cosf(a); h_geom[t].fDetUZ = 0;
        h_geom[t].fDetVX = 0; h_geom[t].fDetVY = 0; h_geom[t].fDetVZ = dv;
        float cx = (SDD-SID)*cosf(a), cy = (SDD-SID)*sinf(a);
        h_geom[t].fDetSX = cx - (Nu-1)*0.5f*h_geom[t].fDetUX; 
        h_geom[t].fDetSY = cy - (Nu-1)*0.5f*h_geom[t].fDetUY; 
        h_geom[t].fDetSZ = -(Nv-1)*0.5f*dv;
    }

    SConeProjection *d_geom; cudaMalloc(&d_geom, Ntheta*sizeof(SConeProjection));
    cudaMemcpy(d_geom, h_geom.data(), Ntheta*sizeof(SConeProjection), cudaMemcpyHostToDevice);

    float *d_vol, *d_proj, *d_recon;
    cudaMalloc(&d_vol, Nx*Ny*Nz*sizeof(float));
    cudaMalloc(&d_proj, Nu*Nv*Ntheta*sizeof(float));
    cudaMalloc(&d_recon, Nx*Ny*Nz*sizeof(float));

    // 2. 创建测试球体 (密度 1.0)
    std::vector<float> h_ph(Nx*Ny*Nz, 0.0f);
    for(int i=0; i<Nx*Ny*Nz; i++) {
        int x=i%Nx, y=(i/Nx)%Ny, z=i/(Nx*Ny);
        if(pow(x-64,2)+pow(y-64,2)+pow(z-64,2) < 1600) h_ph[i]=1.0f;
    }
    cudaMemcpy(d_vol, h_ph.data(), Nx*Ny*Nz*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blk(8,8,8);
    dim3 gridP((Nu+7)/8, (Nv+7)/8, (Ntheta+7)/8);

    // 3. 正向投影
    printf("Forward Projecting...\n");
    astra_fp_kernel<<<gridP, blk>>>(d_vol, d_proj, Nx, Ny, Nz, Nu, Nv, Ntheta, d_geom);
    
    // 4. FDK 滤波链条
    printf("Filtering...\n");
    fdk_preweight_kernel<<<gridP, blk>>>(d_proj, Nu, Nv, Ntheta, d_geom, SID);

    int Nu_c = Nu/2 + 1;
    cufftHandle plan_fwd, plan_inv;
    cufftPlanMany(&plan_fwd, 1, &Nu, NULL, 1, Nu, NULL, 1, Nu_c, CUFFT_R2C, Nv*Ntheta);
    cufftPlanMany(&plan_inv, 1, &Nu, NULL, 1, Nu_c, NULL, 1, Nu, CUFFT_C2R, Nv*Ntheta);
    
    cufftComplex* d_f; cudaMalloc(&d_f, Nv*Ntheta*Nu_c*sizeof(cufftComplex));
    cufftExecR2C(plan_fwd, d_proj, d_f);
    
    // 核心修正：传入 Nu 进行归一化
    ramp_filter_kernel<<<(dim3((Nu_c+7)/8,(Nv+7)/8,(Ntheta+7)/8)), blk>>>(d_f, Nu_c, Nv, Ntheta, du, Nu);
    
    cufftExecC2R(plan_inv, d_f, d_proj);

    // 5. 反投影
    printf("Backprojecting...\n");
    cudaMemset(d_recon, 0, Nx*Ny*Nz*sizeof(float));
    astra_bp_kernel<<<(dim3((Nx+7)/8,(Ny+7)/8,(Nz+7)/8)), blk>>>(d_recon, Nx, Ny, Nz, d_proj, Nu, Nv, Ntheta, d_geom, SID);

    // 6. 保存结果
    std::vector<float> h_res(Nx*Ny*Nz);
    cudaMemcpy(h_res.data(), d_recon, Nx*Ny*Nz*sizeof(float), cudaMemcpyDeviceToHost);
    save_raw("reconstruction_fixed.raw", h_res);

    printf("Success!\n");

    cufftDestroy(plan_fwd); cufftDestroy(plan_inv);
    cudaFree(d_vol); cudaFree(d_proj); cudaFree(d_recon); cudaFree(d_f); cudaFree(d_geom);
    return 0;
}