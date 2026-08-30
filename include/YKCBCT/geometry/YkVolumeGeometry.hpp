#pragma once

#include <cuda_runtime.h>

namespace YK {

// 公共体积网格几何。center 是体积中心的世界坐标；体素 (0,0,0)
// 的中心由 origin() 统一派生，不要求体积中心位于世界原点。
struct SVolGeom {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    float vox_x = 1.f;
    float vox_y = 1.f;
    float vox_z = 1.f;
    float tmp_rcp_vox_x = 1.f;
    float tmp_rcp_vox_y = 1.f;
    float tmp_rcp_vox_z = 1.f;
    float3 center = make_float3(0.f, 0.f, 0.f);

    static SVolGeom make_centered(int nx, int ny, int nz, float voxel)
    {
        return make_centered(nx, ny, nz, voxel, voxel, voxel);
    }

    static SVolGeom make_centered(int nx, int ny, int nz, float voxel_xy,
        float voxel_z)
    {
        return make_centered(nx, ny, nz, voxel_xy, voxel_xy, voxel_z);
    }

    static SVolGeom make_centered(int nx, int ny, int nz, float voxel_x,
        float voxel_y, float voxel_z)
    {
        SVolGeom geometry;
        geometry.Nx = nx;
        geometry.Ny = ny;
        geometry.Nz = nz;
        geometry.vox_x = voxel_x;
        geometry.vox_y = voxel_y;
        geometry.vox_z = voxel_z;
        geometry.tmp_rcp_vox_x = 1.f / voxel_x;
        geometry.tmp_rcp_vox_y = 1.f / voxel_y;
        geometry.tmp_rcp_vox_z = 1.f / voxel_z;
        return geometry;
    }

    __host__ __device__ float3 origin() const
    {
        return make_float3(
            center.x - (Nx - 1) * 0.5f * vox_x,
            center.y - (Ny - 1) * 0.5f * vox_y,
            center.z - (Nz - 1) * 0.5f * vox_z);
    }
};

} // namespace YK
