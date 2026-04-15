#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "../../global/YkGlobals.h"



namespace YK {
    namespace Fp {
        // ============================================================
// 主轴判断（host side）
// 使用 srcCR（中心射线方向单位向量）判断最大分量
// ============================================================
        enum class MainAxis { X, Y, Z };

        inline MainAxis getMainAxis(const float3& CenterRayDir)
        {
            float ax = fabsf(CenterRayDir.x);
            float ay = fabsf(CenterRayDir.y);
            float az = fabsf(CenterRayDir.z);
            if (ax >= ay && ax >= az) return MainAxis::X;
            if (ay >= ax && ay >= az) return MainAxis::Y;
            return MainAxis::Z;
        }



        inline SConeProjGeomVec normalizeToVoxel(
            const SConeProjGeomVec& v, const SVolGeom& g)
        {
            const float cx = g.center.x, cy = g.center.y, cz = g.center.z;
            const float ivx = 1.f / g.vox_x;
            const float ivy = 1.f / g.vox_y;
            const float ivz = 1.f / g.vox_z;

            SConeProjGeomVec r;
            r.src.x = (v.src.x - cx) * ivx;
            r.src.y = (v.src.y - cy) * ivy;
            r.src.z = (v.src.z - cz) * ivz;
            r.detS.x = (v.detS.x - cx) * ivx;
            r.detS.y = (v.detS.y - cy) * ivy;
            r.detS.z = (v.detS.z - cz) * ivz;
            r.detU.x = v.detU.x * ivx;
            r.detU.y = v.detU.y * ivy;
            r.detU.z = v.detU.z * ivz;
            r.detV.x = v.detV.x * ivx;
            r.detV.y = v.detV.y * ivy;
            r.detV.z = v.detV.z * ivz;
            r.srcCR = v.srcCR;
            r.angle = v.angle;
            return r;
        }

        // 批量转换
        inline std::vector<SConeProjGeomVec> normalizeToVoxelBatch(
            const std::vector<SConeProjGeomVec>& views, const SVolGeom& g)
        {
            std::vector<SConeProjGeomVec> out(views.size());
            for (int i = 0; i < (int)views.size(); ++i)
                out[i] = normalizeToVoxel(views[i], g);
            return out;
        }


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
                __device__ static float sample(cudaTextureObject_t tex, float f0, float f1, float f2)
                {
                    return tex3D<float>(tex, f1, f2, f0);
                }
            };
            // ============================================================
            // launch 配置常量
            // ============================================================
            constexpr int kAnglesPerBlock = 4;
            constexpr int kBlockSlices = 4;
            constexpr int kDetBlockU = 32;
            constexpr int kDetBlockV = 32;



        } // namespace detail



    };// namespace Fp
}; // namespace YK