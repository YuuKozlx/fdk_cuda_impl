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

            template<typename DIR, int kStepDenom = 1>
            __global__ void fp_joseph_kernel(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,
                float* d_sino,
                int nSlices, int nDim1, int nDim2,
                int Nu, int Nv,
                int startSlice,
                int startAngle, int endAngle,
                SVolGeom vg,        // ← 原来是 float fMainAxisVox，改为 SVolGeom vg
                bool accumulate)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV + threadIdx.z;
                if (detV >= Nv) return;

                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;

                // detS 是像素(0,0)中心，直接用，不加 0.5f 偏移
                const float fDetX = v.detS.x + detU * v.detU.x + detV * v.detV.x;
                const float fDetY = v.detS.y + detU * v.detU.y + detV * v.detV.y;
                const float fDetZ = v.detS.z + detU * v.detU.z + detV * v.detV.z;

                // ================================================================
                // fp_joseph_kernel 射线方程推导注释版
                // ================================================================

                // ----------------------------------------------------------------
                // 坐标系约定
                // ----------------------------------------------------------------
                // 【体积中心坐标系】：原点在体积中心，单位=体素索引
                //   src/det 经过 normalizeToVoxel 处理：(world - center) / vox
                //   所以 src_c0/c1/c2 范围约 [-N/2, N/2]
                //
                // 【0-based 体素索引】：原点在体积角落，范围 [0, N)
                //   0-based = 体积中心坐标 + N/2
                //
                // 【CUDA 纹理坐标】：tex3D 约定第 k 个体素中心 = k + 0.5
                //   纹理坐标 = 0-based 体素索引 + 0.5
                //
                // 三者关系：
                //   纹理坐标 = 体积中心坐标 + N/2 + 0.5
                // ----------------------------------------------------------------

                // ----------------------------------------------------------------
                // 射线参数化
                // ----------------------------------------------------------------
                // 射线过源点 src，方向指向探测器像素中心 det。
                // 用主轴坐标 c0 作为参数（体积中心坐标系），消掉参数 t：
                //
                //   t = (c0 - src_c0) / (det_c0 - src_c0)
                //     = (c0 - src_c0) / -(src_c0 - det_c0)
                //
                // 代入副轴：
                //   c1 = src_c1 + t * (det_c1 - src_c1)
                //      = src_c1 + [(c0 - src_c0) * (src_c1 - det_c1)] / (src_c0 - det_c0)
                //
                // 令斜率：
                //   a1 = (src_c1 - det_c1) / (src_c0 - det_c0)
                //
                // 则：
                //   c1 = src_c1 + a1 * (c0 - src_c0)
                //      = a1 * c0 + (src_c1 - a1 * src_c0)
                //      = a1 * c0 + b1
                //
                // 截距：
                //   b1 = src_c1 - a1 * src_c0
                //
                // 完整射线方程（体积中心坐标系）：
                //   c1 = a1 * c0 + b1
                //   c2 = a2 * c0 + b2
                // ----------------------------------------------------------------

                // ----------------------------------------------------------------
                // 步进起点 f0/f1/f2 的推导
                // ----------------------------------------------------------------
                // 目标：算出第 startSlice 步的纹理坐标初始值。
                // 之后每步只需 f0+=step, f1+=a1*step, f2+=a2*step（递推，不重算）。
                //
                // ── f0（主轴纹理坐标）──────────────────────────────────────────
                //
                // 第 startSlice 步的主轴步中心（0-based 超采样步索引）：
                //   pos_step = (startSlice + 0.5) * step
                //
                // 换成纹理坐标（主轴方向 0-based 步索引直接对应纹理坐标，无原点平移）：
                //   f0 = (startSlice + 0.5) * step
                //      = startSlice * step + 0.5 * step
                //
                // ── f1（副轴纹理坐标）──────────────────────────────────────────
                //
                // 步骤1：当前步的主轴坐标（体积中心坐标系）
                //   c0 = (startSlice + 0.5) * step   [0-based 步中心]
                //      - nSlicesOrig / 2              [0-based → 体积中心原点]
                //   即：
                //   c0 = startSlice * step - 0.5 * nSlicesOrig + 0.5 * step
                //
                // 步骤2：代入射线方程得副轴坐标（体积中心坐标系）
                //   c1 = a1 * c0 + b1
                //      = a1 * (startSlice*step - 0.5*nSlicesOrig + 0.5*step) + b1
                //
                // 步骤3：体积中心坐标系 → 0-based 体素索引
                //   体积中心原点在第 nDim1/2 个体素处，所以：
                //   c1_idx = c1 + nDim1 / 2
                //          = c1 + 0.5 * nDim1
                //
                // 步骤4：0-based 体素索引 → CUDA 纹理坐标
                //   tex3D 第 k 个体素中心 = k + 0.5，所以：
                //   f1 = c1_idx + 0.5
                //
                // 合并步骤1~4：
                //   f1 = a1 * (startSlice*step - 0.5*nSlicesOrig + 0.5*step)
                //      + b1          ← 射线截距（体积中心坐标系）
                //      + 0.5*nDim1   ← 步骤3：体积中心 → 0-based
                //      + 0.5         ← 步骤4：0-based → CUDA 纹理中心
                //
                // f2 完全对称（a1→a2, b1→b2, nDim1→nDim2）。
                // ----------------------------------------------------------------

                // ----------------------------------------------------------------
                // 为什么 f0 不需要 +0.5*nDim0 和 +0.5？
                // ----------------------------------------------------------------
                // f1/f2 通过射线方程得到，截距 b1/b2 来自源点坐标（体积中心坐标系），
                // 所以 c1/c2 在体积中心坐标系下，必须做步骤3+步骤4的两次平移。
                //
                // f0 从 0-based 步索引直接算，没有经过体积中心坐标系，
                // 0.5*step 已经是步中心偏移（相当于步骤4），不需要步骤3。
                // ----------------------------------------------------------------

                // ── 代码实现 ────────────────────────────────────────────────────

                // 斜率：(src_c1 - det_c1) / (src_c0 - det_c0)
                const float rcp_dc0 = __frcp_rn(
                    DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_dc0;
                const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_dc0;

                // 截距：b1 = src_c1 - a1 * src_c0（体积中心坐标系）
                const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                // ── 物理弦长（各向异性精确，在步进循环外算一次）────────────────
// a1/a2 已知，每条射线只算一次
                const float v0 = DIR::vox0(vg.vox_x, vg.vox_y, vg.vox_z);
                const float v1 = DIR::vox1(vg.vox_x, vg.vox_y, vg.vox_z);
                const float v2 = DIR::vox2(vg.vox_x, vg.vox_y, vg.vox_z);
                const float a1v1 = a1 * v1;
                const float a2v2 = a2 * v2;
                const float fPhysLen = __fsqrt_rn(v0 * v0 + a1v1 * a1v1 + a2v2 * a2v2);


                constexpr float step = 1.f / (float)kStepDenom;
                const int nSlicesOrig = nSlices / kStepDenom;   // 原始体素数（主轴方向）
                const float a1_step = a1 * step;
                const float a2_step = a2 * step;

                // 步进起点纹理坐标（见上方推导）
                //
                // f0 = (startSlice + 0.5) * step
                //    主轴：0-based 步中心，直接是纹理坐标
                float f0 = startSlice * step + 0.5f * step;

                // f1 = a1*(c0_体积中心) + b1 + 0.5*nDim1 + 0.5
                //    c0_体积中心 = (startSlice+0.5)*step - nSlicesOrig/2
                //    + 0.5*nDim1：体积中心 → 0-based 体素索引
                //    + 0.5      ：0-based  → CUDA 纹理中心
                float f1 = a1 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b1
                    + 0.5f * nDim1   // 体积中心坐标 → 0-based 体素索引
                    + 0.5f;           // 0-based 体素索引 → CUDA 纹理中心

                float f2 = a2 * (startSlice * step - 0.5f * nSlicesOrig + 0.5f * step)
                    + b2
                    + 0.5f * nDim2   // 同 f1
                    + 0.5f;
                float fVal = 0.f;
                const int endSlice = min(startSlice + kBlockSlices, nSlices);
                for (int s = startSlice; s < endSlice; ++s)
                {
                    fVal += DIR::sample(volTex, f0, f1, f2);
                    f0 += step;
                    f1 += a1_step;
                    f2 += a2_step;
                }

                fVal *= fPhysLen * step;   // 原: fDistCorr * step * fMainAxisVox

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
                SVolGeom vg,        // ← 原 float fMainAxisVox，改为 SVolGeom vg
                bool accumulate)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int detV = (blockIdx.x / nUBlocks) * kDetBlockV + threadIdx.z;
                if (detV >= Nv) return;

                const SConeProjGeomVec& v = d_views_vox[angle];
                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;

                const float fDetSX = v.detS.x;
                const float fDetSY = v.detS.y;
                const float fDetSZ = v.detS.z;

                constexpr float kStep = 1.f / (float)kStepDenom;
                constexpr float kSubStep = 1.f / (float)kRaysPerDim;
                constexpr float rcp_nRays = 1.f / (float)(kRaysPerDim * kRaysPerDim);
                const int nSlicesOrig = nSlices / kStepDenom;
                const int endSlice = min(startSlice + kBlockSlices, nSlices);

                // ── 各向异性物理体素尺寸（子射线循环外，只算一次）────────
                // DIR::vox0/1/2 按主轴重排：
                //   DirX: vox0=vox_x(主轴), vox1=vox_y, vox2=vox_z
                //   DirY: vox0=vox_y(主轴), vox1=vox_x, vox2=vox_z
                //   DirZ: vox0=vox_z(主轴), vox1=vox_x, vox2=vox_y
                const float v0 = DIR::vox0(vg.vox_x, vg.vox_y, vg.vox_z);  // 主轴
                const float v1 = DIR::vox1(vg.vox_x, vg.vox_y, vg.vox_z);  // 副轴1
                const float v2 = DIR::vox2(vg.vox_x, vg.vox_y, vg.vox_z);  // 副轴2

                float fV = 0.f;

                // kRaysPerDim=1 时 fdU=detU, fdV=detV（像素中心，无超采样）
                float fdU = detU - 0.5f + 0.5f * kSubStep;
                for (int iSubU = 0; iSubU < kRaysPerDim; ++iSubU, fdU += kSubStep)
                {
                    float fdV = detV - 0.5f + 0.5f * kSubStep;
                    for (int iSubV = 0; iSubV < kRaysPerDim; ++iSubV, fdV += kSubStep)
                    {
                        // 当前子射线的探测器点
                        const float fDetX = fDetSX + fdU * v.detU.x + fdV * v.detV.x;
                        const float fDetY = fDetSY + fdU * v.detU.y + fdV * v.detV.y;
                        const float fDetZ = fDetSZ + fdU * v.detU.z + fdV * v.detV.z;

                        // 射线斜率（体积中心坐标系）
                        // a1 = (src_c1-det_c1)/(src_c0-det_c0)
                        const float rcp_dc0 = __frcp_rn(
                            DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                        const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ)) * rcp_dc0;
                        const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ)) * rcp_dc0;

                        // 射线截距：b1 = src_c1 - a1*src_c0（体积中心坐标系）
                        const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                        const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                        // ── 物理弦长（各向异性精确）──────────────────────
                        // fPhysLen = sqrt(v0² + (a1·v1)² + (a2·v2)²)
                        // 各向同性(v0=v1=v2=vox):
                        //   = vox·sqrt(a1²+a2²+1) = fDistCorr·fMainAxisVox ✓
                        const float a1v1 = a1 * v1;
                        const float a2v2 = a2 * v2;
                        const float fPhysLen = __fsqrt_rn(v0 * v0 + a1v1 * a1v1 + a2v2 * a2v2);


                        const float a1_step = a1 * kStep;
                        const float a2_step = a2 * kStep;

                        // ── 步进起点纹理坐标 ──────────────────────────────
                        // f0：主轴，0-based 步中心
                        //   f0 = (startSlice + 0.5) * kStep
                        //
                        // f1：副轴，体积中心坐标 → 纹理坐标
                        //   c0(体积中心) = (startSlice+0.5)*kStep - nSlicesOrig/2
                        //   c1 = a1*c0 + b1
                        //   纹理坐标 = c1 + (nDim1-1)*0.5 + 0.5
                        float f0 = startSlice * kStep + 0.5f * kStep;
                        float f1 = a1 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
                            + b1 + (nDim1 - 1) * 0.5f + 0.5f;
                        float f2 = a2 * (startSlice * kStep - 0.5f * nSlicesOrig + 0.5f * kStep)
                            + b2 + (nDim2 - 1) * 0.5f + 0.5f;

                        float fVal = 0.f;
                        for (int s = startSlice; s < endSlice; ++s)
                        {
                            fVal += DIR::sample(volTex, f0, f1, f2);
                            f0 += kStep;
                            f1 += a1_step;
                            f2 += a2_step;
                        }

                        fV += fVal * fPhysLen * kStep;  // 原: fVal * fDistCorr * kStep
                        // fPhysLen 已含主轴物理尺寸


                    }
                }

                fV *= rcp_nRays;   // 原: rcp_nRays * fMainAxisVox
                // fMainAxisVox 已合进 fPhysLen，不再单独乘

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
                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
                const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV);
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                // ← 原来算 fVoxScale = DIR::voxSize(g)，现在直接传 g
                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    const bool acc = (s == 0) ? accumulate : true;
                    fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
                        volTex, d_views_vox, d_sino,
                        nSlices, nDim1, nDim2,
                        Nu, Nv, s,
                        startAngle, endAngle,
                        g,      // ← 原来是 fVoxScale，现在传 SVolGeom g
                        acc);
                }
            }



            // ============================================================
            // launch_impl：fVoxScale → g
            // ============================================================
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
                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
                const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV);
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                // fVoxScale 行删除，改为直接传 g（SVolGeom 按值传，kernel 按值接收）

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    const bool acc = (s == 0) ? accumulate : true;

                    if constexpr (kRaysPerDim == 1)
                    {
                        fp_joseph_kernel<DIR, kStepDenom> << <grid, block, 0, stream >> > (
                            volTex, d_views_vox, d_sino,
                            nSlices, nDim1, nDim2,
                            Nu, Nv, s, startAngle, endAngle,
                            g,      // ← 原 fVoxScale，改为 g
                            acc);
                    }
                    else
                    {
                        fp_joseph_kernel_stepss_detss<DIR, kStepDenom, kRaysPerDim>
                            << <grid, block, 0, stream >> > (
                                volTex, d_views_vox, d_sino,
                                nSlices, nDim1, nDim2,
                                Nu, Nv, s, startAngle, endAngle,
                                g,      // ← 原 fVoxScale，改为 g
                                acc);
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