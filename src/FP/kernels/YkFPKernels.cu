#include "YkFPHelpers.cuh"
#include "YkFPLaunch.cuh"
#include "global/YkMacro.hpp"

namespace YK {
    namespace Fp {

        namespace detail {

            struct DirX {
                __host__ __device__ static float c0(float x, float y, float z) { return x; }
                __host__ __device__ static float c1(float x, float y, float z) { return y; }
                __host__ __device__ static float c2(float x, float y, float z) { return z; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static float vox0(float vx, float vy, float vz) { return vx; }
                __host__ __device__ static float vox1(float vx, float vy, float vz) { return vy; }
                __host__ __device__ static float vox2(float vx, float vy, float vz) { return vz; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_x; }
                __device__ static float sample(cudaTextureObject_t tex, float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f0, f1, f2);
                }
            };

            struct DirY {
                __host__ __device__ static float c0(float x, float y, float z) { return y; }
                __host__ __device__ static float c1(float x, float y, float z) { return x; }
                __host__ __device__ static float c2(float x, float y, float z) { return z; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static float vox0(float vx, float vy, float vz) { return vy; }
                __host__ __device__ static float vox1(float vx, float vy, float vz) { return vx; }
                __host__ __device__ static float vox2(float vx, float vy, float vz) { return vz; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_y; }
                __device__ static float sample(cudaTextureObject_t tex, float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f1, f0, f2);
                }
            };

            struct DirZ {
                __host__ __device__ static float c0(float x, float y, float z) { return z; }
                __host__ __device__ static float c1(float x, float y, float z) { return x; }
                __host__ __device__ static float c2(float x, float y, float z) { return y; }
                __host__ __device__ static int nSlices(int Nx, int Ny, int Nz) { return Nz; }
                __host__ __device__ static int nDim1(int Nx, int Ny, int Nz) { return Nx; }
                __host__ __device__ static int nDim2(int Nx, int Ny, int Nz) { return Ny; }
                __host__ __device__ static float vox0(float vx, float vy, float vz) { return vz; }
                __host__ __device__ static float vox1(float vx, float vy, float vz) { return vx; }
                __host__ __device__ static float vox2(float vx, float vy, float vz) { return vy; }
                __host__ __device__ static float voxSize(const SVolGeom& g) { return g.vox_z; }
                __device__ static float sample(cudaTextureObject_t tex, float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f1, f2, f0);
                }
            };
        }; // detail


        namespace detail {

            // ============================================================
            // fp_joseph_kernel
            //
            // 线程分工：
            //   threadIdx.x — detector U 方向
            //   threadIdx.y — chunk 内相对角度（kAnglesPerBlock 个）
            //   blockIdx.x  — (detV block, detU block) 打包
            //   blockIdx.y  — 角度 block
            //
            // 每次 launch 只处理一段连续的主轴切片 [startSlice, startSlice+kBlockSlices)
            // 外层在 launchGroup 里循环所有切片
            //
            // sinogram 布局：[Na_total][Nv][Nu]，U 最快
            // 写入位置：(angleOffset + startAngle + localAngle, detV, detU)
            // ============================================================
            // ============================================================
            // fp_joseph_kernel
            //
            // 线程分工：
            //   threadIdx.x — detector U 方向
            //   threadIdx.y — chunk 内相对角度（kAnglesPerBlock 个）
            //   blockIdx.x  — (detV block, detU block) 打包
            //   blockIdx.y  — 角度 block
            //
            // 每次 launch 只处理一段连续的主轴切片 [startSlice, startSlice+kBlockSlices)
            // 外层在 launchGroup 里循环所有切片
            //
            // sinogram 布局：[Na_total][Nv][Nu]，U 最快
            // 写入位置：(angleOffset + startAngle + localAngle, detV, detU)
            // ============================================================
            //template<typename DIR, int kStepDenom = 1>
            //__global__ void fp_joseph_kernel(
            //    cudaTextureObject_t      volTex,
            //    const SConeProjGeomVec* d_views_vox,   // 已归一化，体素坐标系
            //    float* d_sino,
            //    int nSlices, int nDim1, int nDim2,
            //    int Nu, int Nv,
            //    int startSlice,
            //    int startAngle, int endAngle,
            //    float fMainAxisVox,  // 主轴方向体素物理尺寸 [mm]，将体素路径长度转换为 mm
            //    bool accumulate)
            //{
            //    const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
            //    const int angle = startAngle + localAngle;
            //    if (angle >= endAngle) return;

            //    const SConeProjGeomVec& v = d_views_vox[angle];

            //    const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
            //    const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
            //    const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
            //    const float fDetSX = v.detS.x + 0.5f * fDetUX + 0.5f * fDetVX;
            //    const float fDetSY = v.detS.y + 0.5f * fDetUY + 0.5f * fDetVY;
            //    const float fDetSZ = v.detS.z + 0.5f * fDetUZ + 0.5f * fDetVZ;

            //    const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
            //    const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
            //    if (detU >= Nu) return;

            //    const int startV = (blockIdx.x / nUBlocks) * kDetBlockV;
            //    const int endV = min(startV + kDetBlockV, Nv);
            //    const int endSlice = min(startSlice + kBlockSlices, nSlices);

            //    for (int detV = startV; detV < endV; ++detV)
            //    {
            //        const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
            //        const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
            //        const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;

            //        // 完全照搬 ASTRA，无任何换算
            //        const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ))
            //            / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
            //        const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ))
            //            / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
            //        const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
            //        const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

            //        const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f);

            //        // 步长 = 1.0 / kStepDenom
            //        constexpr float step = 1.f / (float)kStepDenom;

            //        // 原始 nSlices（体素数）= nSlices / kStepDenom
            //        const int nSlicesOrig = nSlices / kStepDenom;

            //        // ASTRA 原版 f0/f1/f2 推导
            //        float f0 = startSlice * step + 0.5f * step;
            //        float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
            //            + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
            //        float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
            //            + b2 + 0.5f * nDim2 - 0.5f + 0.5f;

            //        float fVal = 0.f;
            //        const int endSlice = min(startSlice + kBlockSlices, nSlices);
            //        for (int s = startSlice; s < endSlice; ++s)
            //        {
            //            fVal += DIR::sample(volTex, f0, f1, f2);
            //            f0 += step;
            //            f1 += a1 * step;
            //            f2 += a2 * step;
            //        }
            //        fVal *= fDistCorr * step;  // 路径长度补偿乘以步长
            //        fVal *= fMainAxisVox;  // 主轴方向体素物理尺寸 [mm]，将体素路径长度转换为 mm


            //        const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
            //        if (accumulate) d_sino[idx] += fVal;
            //        else            d_sino[idx] = fVal;
            //    }
            //}


            //template<typename DIR, int kStepDenom = 1, int kRaysPerDim = 1>
            //__global__ void fp_joseph_kernel_stepss_detss(
            //    cudaTextureObject_t      volTex,
            //    const SConeProjGeomVec* d_views_vox,
            //    float* d_sino,
            //    int nSlices, int nDim1, int nDim2,
            //    int Nu, int Nv,
            //    int startSlice,
            //    int startAngle, int endAngle,
            //    float fMainAxisVox,  // 主轴方向体素物理尺寸 [mm]，将体素路径长度转换为 mm
            //    bool accumulate)
            //{
            //    const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
            //    const int angle = startAngle + localAngle;
            //    if (angle >= endAngle) return;

            //    const SConeProjGeomVec& v = d_views_vox[angle];
            //    const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
            //    const float fDetSX = v.detS.x + 0.5f * v.detU.x + 0.5f * v.detV.x;
            //    const float fDetSY = v.detS.y + 0.5f * v.detU.y + 0.5f * v.detV.y;
            //    const float fDetSZ = v.detS.z + 0.5f * v.detU.z + 0.5f * v.detV.z;

            //    const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
            //    const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
            //    if (detU >= Nu) return;

            //    const int startV = (blockIdx.x / nUBlocks) * kDetBlockV;
            //    const int endV = min(startV + kDetBlockV, Nv);
            //    const int endSlice = min(startSlice + kBlockSlices, nSlices);

            //    constexpr float kStep = 1.f / (float)kStepDenom;
            //    constexpr float kSubStep = 1.f / (float)kRaysPerDim;
            //    const int nSlicesOrig = nSlices / kStepDenom;

            //    for (int detV = startV; detV < endV; ++detV)
            //    {
            //        float fV = 0.f;

            //        // 子射线循环：U 方向 kRaysPerDim 个，V 方向 kRaysPerDim 个
            //        float fdU = detU - 0.5f + 0.5f * kSubStep;
            //        for (int iSubU = 0; iSubU < kRaysPerDim; ++iSubU, fdU += kSubStep)
            //        {
            //            float fdV = detV - 0.5f + 0.5f * kSubStep;
            //            for (int iSubV = 0; iSubV < kRaysPerDim; ++iSubV, fdV += kSubStep)
            //            {
            //                const float fDetX = fDetSX + fdU * v.detU.x + fdV * v.detV.x;
            //                const float fDetY = fDetSY + fdU * v.detU.y + fdV * v.detV.y;
            //                const float fDetZ = fDetSZ + fdU * v.detU.z + fdV * v.detV.z;

            //                const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ))
            //                    / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
            //                const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ))
            //                    / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
            //                const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
            //                const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

            //                const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f);

            //                float f0 = startSlice * kStep + 0.5f * kStep;
            //                float f1 = a1 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
            //                    + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
            //                float f2 = a2 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
            //                    + b2 + 0.5f * nDim2 - 0.5f + 0.5f;

            //                float fVal = 0.f;
            //                for (int s = startSlice; s < endSlice; ++s)
            //                {
            //                    fVal += DIR::sample(volTex, f0, f1, f2);
            //                    f0 += kStep;
            //                    f1 += a1 * kStep;
            //                    f2 += a2 * kStep;
            //                }
            //                fVal *= fDistCorr * kStep;
            //                fV += fVal;
            //            }
            //        }

            //        // 子射线平均
            //        fV /= (float)(kRaysPerDim * kRaysPerDim);
            //        fV *= fMainAxisVox;  // 主轴方向体素物理尺寸 [mm]，将体素路径长度转换为 mm

            //        const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
            //        if (accumulate) d_sino[idx] += fV;
            //        else            d_sino[idx] = fV;
            //    }
            //}




            //// launchGroupImpl 加模板参数
            //template<typename DIR, int kStepDenom = 1>
            //// kStepDenom=1 → 步长1.0（原始）
            //// kStepDenom=2 → 步长0.5
            //// kStepDenom=4 → 步长0.25
            //static void fp_joseph_launch_group_impl(
            //    cudaTextureObject_t      volTex,
            //    const SConeProjGeomVec* d_views_vox,
            //    float* d_sino,
            //    const SVolGeom& g,
            //    int Nu, int Nv,
            //    int startAngle, int endAngle,
            //    bool accumulate,
            //    cudaStream_t stream)
            //{
            //    const int K = endAngle - startAngle;
            //    if (K <= 0) return;

            //    const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
            //    const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
            //    const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);

            //    const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
            //    const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
            //    const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

            //    dim3 block(kDetBlockU, kAnglesPerBlock);
            //    dim3 grid(nUBlocks * nVBlocks, nABlocks);

            //    const float fVoxScale = DIR::voxSize(g);  // 按主轴选体素尺寸

            //    for (int s = 0; s < nSlices; s += kBlockSlices)
            //    {
            //        const bool acc = (s == 0) ? accumulate : true;
            //        fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
            //            volTex, d_views_vox, d_sino,
            //            nSlices, nDim1, nDim2,
            //            Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
            //    }
            //}

            //template<typename DIR, int kStepDenom, int kRaysPerDim>
            //static void fp_joseph_launch_group_ss_impl(cudaTextureObject_t      volTex,
            //    const SConeProjGeomVec* d_views_vox,
            //    float* d_sino,
            //    const SVolGeom& g,
            //    int Nu, int Nv,
            //    int startAngle, int endAngle,
            //    bool accumulate,
            //    cudaStream_t stream)
            //{
            //    const int K = endAngle - startAngle;
            //    if (K <= 0) return;

            //    const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
            //    const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
            //    const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);

            //    const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
            //    const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
            //    const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

            //    dim3 block(kDetBlockU, kAnglesPerBlock);
            //    dim3 grid(nUBlocks * nVBlocks, nABlocks);

            //    const float fVoxScale = DIR::voxSize(g);  // 按主轴选体素尺寸

            //    for (int s = 0; s < nSlices; s += kBlockSlices)
            //    {
            //        const bool acc = (s == 0) ? accumulate : true;

            //        if constexpr (kRaysPerDim == 1)
            //        {
            //            fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
            //                volTex, d_views_vox, d_sino,
            //                nSlices, nDim1, nDim2,
            //                Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
            //        }
            //        else
            //        {
            //            fp_joseph_kernel_stepss_detss<DIR, kStepDenom, kRaysPerDim> << <grid, block, 0, stream >> > (
            //                volTex, d_views_vox, d_sino,
            //                nSlices, nDim1, nDim2,
            //                Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
            //        }
            //    }
            //}

            template<typename DIR, int kStepDenom = 1>
            __global__ void fp_joseph_kernel(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,   // 已归一化，体素坐标系
                float* d_sino,
                int nSlices, int nDim1, int nDim2,
                int Nu, int Nv,
                int startSlice,
                int startAngle, int endAngle,
                float fMainAxisVox,  // 主轴方向体素物理尺寸 [mm]，将体素路径长度转换为 mm
                bool accumulate)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV_par + threadIdx.z;
                if (detV >= Nv) return;

                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
                const float fDetSX = v.detS.x + 0.5f * v.detU.x + 0.5f * v.detV.x;
                const float fDetSY = v.detS.y + 0.5f * v.detU.y + 0.5f * v.detV.y;
                const float fDetSZ = v.detS.z + 0.5f * v.detU.z + 0.5f * v.detV.z;

                const float fDetX = fDetSX + detU * v.detU.x + detV * v.detV.x;
                const float fDetY = fDetSY + detU * v.detU.y + detV * v.detV.y;
                const float fDetZ = fDetSZ + detU * v.detU.z + detV * v.detV.z;

                const float rcp_dc0 = __frcp_rn(
                    DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_dc0;
                const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_dc0;
                const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                const float fDistCorr = __fsqrt_rn(a1 * a1 + a2 * a2 + 1.f);

                constexpr float step = 1.f / (float)kStepDenom;
                const int nSlicesOrig = nSlices / kStepDenom;
                const float a1_step = a1 * step;
                const float a2_step = a2 * step;

                float f0 = startSlice * step + 0.5f * step;
                float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b1 + 0.5f * nDim1 + 0.5f - 0.5f;
                float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b2 + 0.5f * nDim2 + 0.5f - 0.5f;

                float fVal = 0.f;
                const int endSlice = min(startSlice + kBlockSlices, nSlices);
                for (int s = startSlice; s < endSlice; ++s)
                {
                    fVal += DIR::sample(volTex, f0, f1, f2);
                    f0 += step;
                    f1 += a1_step;
                    f2 += a2_step;
                }

                fVal *= fDistCorr * step * fMainAxisVox;

                const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
                if (accumulate) d_sino[idx] += fVal;
                else            d_sino[idx] = fVal;
            }


            template<typename DIR, int kStepDenom = 1, int kRaysPerDim = 1>
            __global__ void fp_joseph_kernel_stepss_detss(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_sino,
                int nSlices, int nDim1, int nDim2,
                int Nu, int Nv,
                int startSlice,
                int startAngle, int endAngle,
                float fMainAxisVox,
                bool accumulate)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV_par + threadIdx.z;
                if (detV >= Nv) return;

                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
                const float fDetSX = v.detS.x + 0.5f * v.detU.x + 0.5f * v.detV.x;
                const float fDetSY = v.detS.y + 0.5f * v.detU.y + 0.5f * v.detV.y;
                const float fDetSZ = v.detS.z + 0.5f * v.detU.z + 0.5f * v.detV.z;

                constexpr float kStep = 1.f / (float)kStepDenom;
                constexpr float kSubStep = 1.f / (float)kRaysPerDim;
                constexpr float rcp_nRays = 1.f / (float)(kRaysPerDim * kRaysPerDim);
                const int nSlicesOrig = nSlices / kStepDenom;
                const int endSlice = min(startSlice + kBlockSlices, nSlices);

                float fV = 0.f;

                float fdU = detU - 0.5f + 0.5f * kSubStep;
                for (int iSubU = 0; iSubU < kRaysPerDim; ++iSubU, fdU += kSubStep)
                {
                    float fdV = detV - 0.5f + 0.5f * kSubStep;
                    for (int iSubV = 0; iSubV < kRaysPerDim; ++iSubV, fdV += kSubStep)
                    {
                        const float fDetX = fDetSX + fdU * v.detU.x + fdV * v.detV.x;
                        const float fDetY = fDetSY + fdU * v.detU.y + fdV * v.detV.y;
                        const float fDetZ = fDetSZ + fdU * v.detU.z + fdV * v.detV.z;

                        const float rcp_dc0 = __frcp_rn(
                            DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                        const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_dc0;
                        const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_dc0;
                        const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                        const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                        const float fDistCorr = __fsqrt_rn(a1 * a1 + a2 * a2 + 1.f);

                        const float a1_step = a1 * kStep;
                        const float a2_step = a2 * kStep;

                        float f0 = startSlice * kStep + 0.5f * kStep;
                        float f1 = a1 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
                            + b1 + 0.5f * nDim1 + 0.5f;
                        float f2 = a2 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
                            + b2 + 0.5f * nDim2 + 0.5f;

                        float fVal = 0.f;
                        for (int s = startSlice; s < endSlice; ++s)
                        {
                            fVal += DIR::sample(volTex, f0, f1, f2);
                            f0 += kStep;
                            f1 += a1_step;
                            f2 += a2_step;
                        }
                        fV += fVal * fDistCorr * kStep;
                    }
                }

                fV *= rcp_nRays * fMainAxisVox;

                const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
                if (accumulate) d_sino[idx] += fV;
                else            d_sino[idx] = fV;
            }


            template<typename DIR, int kStepDenom = 1>
            static void fp_joseph_launch_group_impl(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_sino,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream)
            {
                const int K = endAngle - startAngle;
                if (K <= 0) return;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int nVBlocks = (Nv + kDetBlockV_par - 1) / kDetBlockV_par;  // ← par版
                const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV_par);  // 32×4×8=1024
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                const float fVoxScale = DIR::voxSize(g);

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    const bool acc = (s == 0) ? accumulate : true;
                    fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
                        volTex, d_views_vox, d_sino,
                        nSlices, nDim1, nDim2,
                        Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
                }
            }


            template<typename DIR, int kStepDenom, int kRaysPerDim>
            static void fp_joseph_launch_group_ss_impl(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_sino,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream)
            {
                const int K = endAngle - startAngle;
                if (K <= 0) return;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz) * kStepDenom;
                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int nVBlocks = (Nv + kDetBlockV_par - 1) / kDetBlockV_par;
                const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV_par);  // 32×4×8=1024
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                const float fVoxScale = DIR::voxSize(g);

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    const bool acc = (s == 0) ? accumulate : true;

                    if constexpr (kRaysPerDim == 1)
                    {
                        fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
                            volTex, d_views_vox, d_sino,
                            nSlices, nDim1, nDim2,
                            Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
                    }
                    else
                    {
                        fp_joseph_kernel_stepss_detss<DIR, kStepDenom, kRaysPerDim> << <grid, block, 0, stream >> > (
                            volTex, d_views_vox, d_sino,
                            nSlices, nDim1, nDim2,
                            Nu, Nv, s, startAngle, endAngle, fVoxScale, acc);
                    }
                }
            }




            template<int kStepDenom, int kRaysPerDim = 1>
            static void launchAngle_ss(
                MainAxis ax,
                cudaTextureObject_t volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_s,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream)
            {
                switch (ax) {
                case MainAxis::X:
                    fp_joseph_launch_group_ss_impl<DirX, kStepDenom, kRaysPerDim>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                case MainAxis::Y:
                    fp_joseph_launch_group_ss_impl<DirY, kStepDenom, kRaysPerDim>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                case MainAxis::Z:
                    fp_joseph_launch_group_ss_impl<DirZ, kStepDenom, kRaysPerDim>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                }
            }

            // 辅助函数，避免 switch 嵌套
            template<int kStepDenom>
            static void launchAngle(
                MainAxis ax,
                cudaTextureObject_t volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_s,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,   // ← 改这里
                bool accumulate,
                cudaStream_t stream)
            {
                switch (ax) {
                case MainAxis::X:
                    fp_joseph_launch_group_impl<DirX, kStepDenom>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                case MainAxis::Y:
                    fp_joseph_launch_group_impl<DirY, kStepDenom>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                case MainAxis::Z:
                    fp_joseph_launch_group_impl<DirZ, kStepDenom>(
                        volTex, d_views_vox, d_s, g, Nu, Nv, startAngle, endAngle, accumulate, stream);
                    break;
                }
            }


            // 统一分发，避免两个重载重复 if - else
            template<typename GetAxis>
            static void fp_ss_dispatch(
                GetAxis                  getAxis,   // lambda: int -> MainAxis
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_sino,
                const SVolGeom& g,
                int Nu, int Nv,
                int start, int end,
                bool accumulate,
                cudaStream_t stream,
                FpStepSuperSample ss,
                FpDetSuperSample   det)
            {
                const MainAxis ax = getAxis(start);
                float* d_s = d_sino + (size_t)start * Nv * Nu;

                if (ss == FpStepSuperSample::x1 && det == FpDetSuperSample::x1)
                    launchAngle_ss<1, 1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x2 && det == FpDetSuperSample::x1)
                    launchAngle_ss<2, 1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x4 && det == FpDetSuperSample::x1)
                    launchAngle_ss<4, 1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x1 && det == FpDetSuperSample::x2)
                    launchAngle_ss<1, 2>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x2 && det == FpDetSuperSample::x2)
                    launchAngle_ss<2, 2>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x1 && det == FpDetSuperSample::x4)
                    launchAngle_ss<1, 4>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x2 && det == FpDetSuperSample::x4)
                    launchAngle_ss<2, 4>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else if (ss == FpStepSuperSample::x4 && det == FpDetSuperSample::x2)
                    launchAngle_ss<4, 2>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                else {
                    YK_LOGW("Unsupported super-sampling combination, fallback to no super-sampling");
                    launchAngle_ss<1, 1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, start, end, accumulate, stream);
                }

            }

        }; // detail 




        void fp_joseph_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float4>& h_src_dirs,
            const SConeProjGeomVec* d_views_vox,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample step)
        {
            for (int a = 0; a < Na; ++a)
            {
                const MainAxis ax = getMainAxis(h_src_dirs[a]);
                float* d_s = d_sino + (size_t)a * Nv * Nu;

                switch (step) {
                case FpStepSuperSample::x1:
                    detail::launchAngle<1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, a, a + 1, accumulate, stream);
                    break;
                case FpStepSuperSample::x2:
                    detail::launchAngle<2>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, a, a + 1, accumulate, stream);
                    break;
                case FpStepSuperSample::x4:
                    detail::launchAngle<4>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, a, a + 1, accumulate, stream);
                    break;
                }
            }
        }

        void fp_joseph_launch(
            cudaTextureObject_t                  volTex,
            const std::vector<SConeProjGeomVec>& h_views,   // 用于主轴判断
            const SConeProjGeomVec* d_views_vox,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss)
        {
            int i = 0;
            while (i < Na)
            {
                const MainAxis ax = getMainAxis(h_views[i].src);
                int j = i + 1;
                while (j < Na && getMainAxis(h_views[j].src) == ax) ++j;

                float* d_s = d_sino + (size_t)i * Nv * Nu;

                switch (ss) {
                case FpStepSuperSample::x1:
                    detail::launchAngle<1>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case FpStepSuperSample::x2:
                    detail::launchAngle<2>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case FpStepSuperSample::x4:
                    detail::launchAngle<4>(ax, volTex, d_views_vox, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                }
                i = j;
            }
        }


        // ============================================================
         // 重载1：h_src_dirs，per-angle
         // ============================================================
        void fp_joseph_ss_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float4>& h_src_dirs,
            const SConeProjGeomVec* d_views_vox,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss,
            FpDetSuperSample   det)
        {
            for (int a = 0; a < Na; ++a)
                detail::fp_ss_dispatch(
                    [&](int i) { return getMainAxis(h_src_dirs[i]); },
                    volTex, d_views_vox, d_sino, g, Nu, Nv,
                    a, a + 1, accumulate, stream, ss, det);
        }

        // ============================================================
        // 重载2：h_views，连续 run 分组
        // ============================================================
        void fp_joseph_ss_launch(
            cudaTextureObject_t                  volTex,
            const std::vector<SConeProjGeomVec>& h_views,
            const SConeProjGeomVec* d_views_vox,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream,
            FpStepSuperSample ss,
            FpDetSuperSample   det)
        {
            int i = 0;
            while (i < Na)
            {
                const MainAxis ax = getMainAxis(h_views[i].src);
                int j = i + 1;
                while (j < Na && getMainAxis(h_views[j].src) == ax) ++j;

                detail::fp_ss_dispatch(
                    [&](int k) { return getMainAxis(h_views[k].src); },
                    volTex, d_views_vox, d_sino, g, Nu, Nv,
                    i, j, accumulate, stream, ss, det);
                i = j;
            }
        }

    } // namespace Fp
} // namespace YK