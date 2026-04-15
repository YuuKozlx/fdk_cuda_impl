#include "YkFPHelpers.cuh"
#include "YkFPLaunch.cuh"

namespace YK {
    namespace Fp {
        namespace detail {

            // ============================================================
            // cone_fp_kernel
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
            // cone_fp_kernel
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
            template<typename DIR>
            __global__ void cone_fp_kernel(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views,   // 已归一化，体素坐标系
                float* d_sino,
                int nSlices, int nDim1, int nDim2,
                int Nu, int Nv,
                int startSlice,
                int startAngle, int endAngle,
                bool accumulate)
            {
                const int localAngle = blockIdx.y * kAnglesPerBlock + threadIdx.y;
                const int angle = startAngle + localAngle;
                if (angle >= endAngle) return;

                const SConeProjGeomVec& v = d_views[angle];

                const float fSrcX = v.src.x, fSrcY = v.src.y, fSrcZ = v.src.z;
                const float fDetUX = v.detU.x, fDetUY = v.detU.y, fDetUZ = v.detU.z;
                const float fDetVX = v.detV.x, fDetVY = v.detV.y, fDetVZ = v.detV.z;
                const float fDetSX = v.detS.x + 0.5f * fDetUX + 0.5f * fDetVX;
                const float fDetSY = v.detS.y + 0.5f * fDetUY + 0.5f * fDetVY;
                const float fDetSZ = v.detS.z + 0.5f * fDetUZ + 0.5f * fDetVZ;

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int detU = (blockIdx.x % nUBlocks) * kDetBlockU + threadIdx.x;
                if (detU >= Nu) return;

                const int startV = (blockIdx.x / nUBlocks) * kDetBlockV;
                const int endV = min(startV + kDetBlockV, Nv);
                const int endSlice = min(startSlice + kBlockSlices, nSlices);

                for (int detV = startV; detV < endV; ++detV)
                {
                    const float fDetX = fDetSX + detU * fDetUX + detV * fDetVX;
                    const float fDetY = fDetSY + detU * fDetUY + detV * fDetVY;
                    const float fDetZ = fDetSZ + detU * fDetUZ + detV * fDetVZ;

                    // 完全照搬 ASTRA，无任何换算
                    const float a1 = (DIR::c1(fSrcX, fSrcY, fSrcZ) - DIR::c1(fDetX, fDetY, fDetZ))
                        / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                    const float a2 = (DIR::c2(fSrcX, fSrcY, fSrcZ) - DIR::c2(fDetX, fDetY, fDetZ))
                        / (DIR::c0(fSrcX, fSrcY, fSrcZ) - DIR::c0(fDetX, fDetY, fDetZ));
                    const float b1 = DIR::c1(fSrcX, fSrcY, fSrcZ) - a1 * DIR::c0(fSrcX, fSrcY, fSrcZ);
                    const float b2 = DIR::c2(fSrcX, fSrcY, fSrcZ) - a2 * DIR::c0(fSrcX, fSrcY, fSrcZ);

                    const float fDistCorr = sqrtf(a1 * a1 + a2 * a2 + 1.f);

                    // ASTRA 原版 f0/f1/f2 推导
                    float f0 = startSlice + 0.5f;
                    float f1 = a1 * (startSlice - 0.5f * nSlices + 0.5f)
                        + b1 + 0.5f * nDim1 - 0.5f + 0.5f;
                    float f2 = a2 * (startSlice - 0.5f * nSlices + 0.5f)
                        + b2 + 0.5f * nDim2 - 0.5f + 0.5f;

                    float fVal = 0.f;
                    for (int s = startSlice; s < endSlice; ++s)
                    {
                        fVal += DIR::sample(volTex, f0, f1, f2);
                        f0 += 1.f;
                        f1 += a1;
                        f2 += a2;
                    }
                    fVal *= fDistCorr;

                    const size_t idx = ((size_t)localAngle * Nv + detV) * Nu + detU;
                    if (accumulate) d_sino[idx] += fVal;
                    else            d_sino[idx] = fVal;
                }
            }



            template<typename DIR>
            static void launchGroupImpl(
                cudaTextureObject_t      volTex,
                const SConeProjGeomVec* d_views_vox,  // 已归一化
                float* d_sino,
                const SVolGeom& g,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream)
            {
                const int K = endAngle - startAngle;
                if (K <= 0) return;

                const int nSlices = DIR::nSlices(g.Nx, g.Ny, g.Nz);
                const int nDim1 = DIR::nDim1(g.Nx, g.Ny, g.Nz);
                const int nDim2 = DIR::nDim2(g.Nx, g.Ny, g.Nz);

                const int nUBlocks = (Nu + kDetBlockU - 1) / kDetBlockU;
                const int nVBlocks = (Nv + kDetBlockV - 1) / kDetBlockV;
                const int nABlocks = (K + kAnglesPerBlock - 1) / kAnglesPerBlock;

                dim3 block(kDetBlockU, kAnglesPerBlock);
                dim3 grid(nUBlocks * nVBlocks, nABlocks);

                for (int s = 0; s < nSlices; s += kBlockSlices)
                {
                    const bool acc = (s == 0) ? accumulate : true;  // slice bug 修复

                    detail::cone_fp_kernel<DIR> << <grid, block, 0, stream >> > (
                        volTex, d_views_vox, d_sino,
                        nSlices, nDim1, nDim2,
                        Nu, Nv,
                        s,
                        startAngle, endAngle,
                        acc);
                }
            }

        }; // namespace detail



        // ============================================================
        // 对外接口实现（与 YkFPLaunch.cuh 中的声明对应）
        // ============================================================
        void fp_launchGroupX(
            cudaTextureObject_t volTex, const SConeProjGeomVec* d_views,
            float* d_sino, const SVolGeom& vol_geom,
            int Nu, int Nv, int startAngle, int endAngle,
            bool accumulate, cudaStream_t stream)
        {
            detail::launchGroupImpl<detail::DirX>(volTex, d_views, d_sino, vol_geom,
                Nu, Nv, startAngle, endAngle, accumulate, stream);
        }

        void fp_launchGroupY(
            cudaTextureObject_t volTex, const SConeProjGeomVec* d_views,
            float* d_sino, const SVolGeom& vol_geom,
            int Nu, int Nv, int startAngle, int endAngle,
            bool accumulate, cudaStream_t stream)
        {
            detail::launchGroupImpl<detail::DirY>(volTex, d_views, d_sino, vol_geom,
                Nu, Nv, startAngle, endAngle, accumulate, stream);
        }

        void fp_launchGroupZ(
            cudaTextureObject_t volTex, const SConeProjGeomVec* d_views,
            float* d_sino, const SVolGeom& vol_geom,
            int Nu, int Nv, int startAngle, int endAngle,
            bool accumulate, cudaStream_t stream)
        {
            detail::launchGroupImpl<detail::DirZ>(volTex, d_views, d_sino, vol_geom,
                Nu, Nv, startAngle, endAngle, accumulate, stream);
        }


        // ----------------------------------------------------------------
           // 核心 dispatch：按主轴分组调用 launchGroupX/Y/Z
           // 直接操作 launchGroup 接口，不经过任何封装层
           // ----------------------------------------------------------------
        void fp_launch(
            cudaTextureObject_t              volTex,
            const std::vector<SConeProjGeomVec>& h_views,   // host，用于主轴判断
            const SConeProjGeomVec* d_views,        // device，全部角度
            float* d_sino,         // device，[Na][Nv][Nu]
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream)
        {
            // 按主轴分组，找连续同主轴的 run
            int i = 0;
            while (i < Na)
            {
                const MainAxis ax = Fp::getMainAxis(h_views[i].srcCR);
                int j = i + 1;
                while (j < Na && Fp::getMainAxis(h_views[i].srcCR) == ax) ++j;

                printf("  group [%d,%d) axis=%c\n", i, j,
                    ax == Fp::MainAxis::X ? 'X' :
                    ax == Fp::MainAxis::Y ? 'Y' : 'Z');

                // [i, j) 是同一主轴的连续 run
                // d_sino 的 angleOffset = i（写入全局 sinogram 的起始角度）
                float* d_s = d_sino + (size_t)i * Nv * Nu;
                switch (ax) {
                case MainAxis::X:
                    Fp::fp_launchGroupX(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case Fp::MainAxis::Y:
                    Fp::fp_launchGroupY(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case Fp::MainAxis::Z:
                    Fp::fp_launchGroupZ(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                }
                i = j;
            }
        }


        void fp_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float3>& h_src_dirs,  // 每个角度的源点，仅用于主轴判断
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream)
        {
            int i = 0;
            while (i < Na)
            {
                const MainAxis ax = getMainAxis(h_src_dirs[i]);
                int j = i + 1;
                while (j < Na && getMainAxis(h_src_dirs[j]) == ax) ++j;

                float* d_s = d_sino + (size_t)i * Nv * Nu;
                switch (ax) {
                case MainAxis::X:
                    fp_launchGroupX(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case MainAxis::Y:
                    fp_launchGroupY(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                case MainAxis::Z:
                    fp_launchGroupZ(volTex, d_views, d_s, g, Nu, Nv, i, j, accumulate, stream);
                    break;
                }
                i = j;
            }
        }

    } // namespace Fp
} // namespace YK