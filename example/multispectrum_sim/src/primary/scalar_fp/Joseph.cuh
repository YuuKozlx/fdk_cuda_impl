// Independently maintained copy from fdk-test f63a8fe: Joseph kernel and axis sampling.
#pragma once
struct KernelGrid {
    int Nx, Ny, Nz;
    float vox_x, vox_y, vox_z;
};
constexpr int kAnglesPerBlock = 2, kBlockSlices = 256, kDetBlockU = 16, kDetBlockV = 4;
struct AxisX {
    static constexpr int axis = 0;
    __host__ __device__ static float c0(float x, float, float) { return x; }
    __host__ __device__ static float c1(float, float y, float) { return y; }
    __host__ __device__ static float c2(float, float, float z) { return z; }
    __host__ __device__ static int nSlices(int nx, int, int) { return nx; }
    __host__ __device__ static int nDim1(int, int ny, int) { return ny; }
    __host__ __device__ static int nDim2(int, int, int nz) { return nz; }
    __host__ __device__ static float vox0(float vx, float, float) { return vx; }
    __host__ __device__ static float vox1(float, float vy, float) { return vy; }
    __host__ __device__ static float vox2(float, float, float vz) { return vz; }
    __host__ __device__ static float voxSize(const KernelGrid& g) { return g.vox_x; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f0, f1, f2); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
    {
        return make_int3(s, d1, d2);
    }
};

struct AxisY {
    static constexpr int axis = 1;
    __host__ __device__ static float c0(float, float y, float) { return y; }
    __host__ __device__ static float c1(float x, float, float) { return x; }
    __host__ __device__ static float c2(float, float, float z) { return z; }
    __host__ __device__ static int nSlices(int, int ny, int) { return ny; }
    __host__ __device__ static int nDim1(int nx, int, int) { return nx; }
    __host__ __device__ static int nDim2(int, int, int nz) { return nz; }
    __host__ __device__ static float vox0(float, float vy, float) { return vy; }
    __host__ __device__ static float vox1(float vx, float, float) { return vx; }
    __host__ __device__ static float vox2(float, float, float vz) { return vz; }
    __host__ __device__ static float voxSize(const KernelGrid& g) { return g.vox_y; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f1, f0, f2); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
    {
        return make_int3(d1, s, d2);
    }
};

struct AxisZ {
    static constexpr int axis = 2;
    __host__ __device__ static float c0(float, float, float z) { return z; }
    __host__ __device__ static float c1(float x, float, float) { return x; }
    __host__ __device__ static float c2(float, float y, float) { return y; }
    __host__ __device__ static int nSlices(int, int, int nz) { return nz; }
    __host__ __device__ static int nDim1(int nx, int, int) { return nx; }
    __host__ __device__ static int nDim2(int, int ny, int) { return ny; }
    __host__ __device__ static float vox0(float, float, float vz) { return vz; }
    __host__ __device__ static float vox1(float vx, float, float) { return vx; }
    __host__ __device__ static float vox2(float, float vy, float) { return vy; }
    __host__ __device__ static float voxSize(const KernelGrid& g) { return g.vox_z; }
#ifdef __CUDACC__
    __device__ static float sample(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return tex3D<float>(texture, f1, f2, f0); }
    __device__ static float sampleVol(cudaTextureObject_t texture,
        float f0, float f1, float f2) { return sample(texture, f0, f1, f2); }
#endif
    __host__ __device__ static int3 toVoxel(int s, int d1, int d2)
    {
        return make_int3(d1, d2, s);
    }
};

template <typename DIR, int kStepDenom = 1>
__global__ void fp_joseph_kernel(
    cudaTextureObject_t volTex,
    const DeviceView* d_views_vox,
    float* d_sino,
    int nSlices, int nDim1, int nDim2,
    int Nu, int Nv,
    int startSlice,
    int startAngle, int endAngle,
    KernelGrid vg,
    bool accumulate)
{
    const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
    const int angle = startAngle + localAngle;
    if (angle >= endAngle)
        return;

    const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
    const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
    if (detU >= Nu)
        return;

    const int detV = (blockIdx.x / nUBlocks) * kDetBlockV + threadIdx.z;
    if (detV >= Nv)
        return;

    const DeviceView& v = d_views_vox[angle];
    if (v.axis != DIR::axis)
        return;
    const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;

    const float fDetX = v.detS.x + detU * v.detU.x + detV * v.detV.x;
    const float fDetY = v.detS.y + detU * v.detU.y + detV * v.detV.y;
    const float fDetZ = v.detS.z + detU * v.detU.z + detV * v.detV.z;

    const float rcp_dc0 = __frcp_rn(
        DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
    const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_dc0;
    const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_dc0;

    const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
    const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

    const float v0 = DIR::vox0(vg.vox_x, vg.vox_y, vg.vox_z);
    const float v1 = DIR::vox1(vg.vox_x, vg.vox_y, vg.vox_z);
    const float v2 = DIR::vox2(vg.vox_x, vg.vox_y, vg.vox_z);
    const float a1v1 = a1 * v1;
    const float a2v2 = a2 * v2;
    const float fPhysLen = __fsqrt_rn(v0 * v0 + a1v1 * a1v1 + a2v2 * a2v2);

    constexpr float step = 1.f / (float)kStepDenom;
    const int nSlicesOrig = nSlices / kStepDenom;
    const float a1_step = a1 * step;
    const float a2_step = a2 * step;

    float f0 = startSlice * step + 0.5f * step;

    float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
        + b1 + (nDim1 - 1) * 0.5f + 0.5f;
    float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
        + b2 + (nDim2 - 1) * 0.5f + 0.5f;
    float fVal = 0.f;
    const int endSlice = min(startSlice + kBlockSlices, nSlices);
    for (int s = startSlice; s < endSlice; ++s) {
        fVal += DIR::sample(volTex, f0, f1, f2);
        f0 += step;
        f1 += a1_step;
        f2 += a2_step;
    }

    fVal *= fPhysLen * step;

    const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
    if (accumulate)
        d_sino[idx] += fVal;
    else
        d_sino[idx] = fVal;
}
