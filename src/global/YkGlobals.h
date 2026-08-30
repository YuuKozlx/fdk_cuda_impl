#pragma once

#include <cuda_runtime.h>

#include "YKCBCT/geometry/YkProjectionGeometry.hpp"
#include "YKCBCT/geometry/YkVolumeGeometry.hpp"

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

} // namespace YK
