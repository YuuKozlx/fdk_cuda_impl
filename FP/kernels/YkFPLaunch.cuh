#pragma once
#include <cuda_runtime.h>
#include "../../global/YkGlobals.h"  // SConeProjGeomVec, SVolGeom

namespace YK {
    namespace Fp {



        namespace detail {


            // ============================================================
// 三个具体 launch 函数，分别对应 DirX / DirY / DirZ
// 在 YkFPKernels.cu 中实现
//
// 参数说明：
//   volTex      — 体积 cudaTextureObject_t
//   d_views     — device 端几何数组，长度 = (endAngle - startAngle)
//                 索引从 0 开始，对应本 chunk 内的局部角度
//   d_sino      — 全局 sinogram 输出 [Na_total][Nv][Nu]，U 最快
//   vol_geom    — 体积几何
//   Nu, Nv      — detector 像素数
//   startAngle  — 本次 launch 在 chunk 内的起始角度（局部索引）
//   endAngle    — 本次 launch 在 chunk 内的结束角度（不含）
//   angleOffset — chunk 在全局 sinogram 中的起始角度偏移
//   accumulate  — true: += / false: =
//   stream      — CUDA stream
// ============================================================
            void fp_launchGroupX(
                cudaTextureObject_t         volTex,
                const SConeProjGeomVec* d_views,
                float* d_sino,
                const SVolGeom& vol_geom,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream);

            void fp_launchGroupY(
                cudaTextureObject_t         volTex,
                const SConeProjGeomVec* d_views,
                float* d_sino,
                const SVolGeom& vol_geom,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream);

            void fp_launchGroupZ(
                cudaTextureObject_t         volTex,
                const SConeProjGeomVec* d_views,
                float* d_sino,
                const SVolGeom& vol_geom,
                int Nu, int Nv,
                int startAngle, int endAngle,
                bool accumulate,
                cudaStream_t stream);
        };// namespace detail


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
            cudaStream_t stream);

        void fp_launch(
            cudaTextureObject_t              volTex,
            const std::vector<float3>& h_src_dirs,  // 每个角度的源点，仅用于主轴判断
            const SConeProjGeomVec* d_views,
            float* d_sino,
            const SVolGeom& g,
            int Na, int Nu, int Nv,
            bool accumulate,
            cudaStream_t stream);

    } // namespace Fp
} // namespace YK