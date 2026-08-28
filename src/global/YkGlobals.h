#pragma once

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"

namespace YK {

struct SDimensions3D {
    unsigned int iVX = 0;
    unsigned int iVY = 0;
    unsigned int iVZ = 0;
    unsigned int iPAng = 0;
    unsigned int iPU = 0;
    unsigned int iPV = 0;
};

struct SProjDims {
    int iPU = 0;
    int iPV = 0;
    int iPAng = 0;

    static SProjDims from(const SDimensions3D& d)
    {
        return {static_cast<int>(d.iPU), static_cast<int>(d.iPV),
            static_cast<int>(d.iPAng)};
    }
};

struct SVolDims {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;

    static SVolDims from(const SDimensions3D& d)
    {
        return {static_cast<int>(d.iVX), static_cast<int>(d.iVY),
            static_cast<int>(d.iVZ)};
    }
};

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

    // 体素 (0,0,0) 中心的世界坐标。
    __host__ __device__ float3 origin() const
    {
        return make_float3(
            center.x - (Nx - 1) * 0.5f * vox_x,
            center.y - (Ny - 1) * 0.5f * vox_y,
            center.z - (Nz - 1) * 0.5f * vox_z);
    }
};

} // namespace YK
