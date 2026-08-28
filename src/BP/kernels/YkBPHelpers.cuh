#pragma once

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"

namespace YK::Bp {

using MainAxis = CudaOp::EMainAxis;
using DirX = CudaOp::AxisX;
using DirY = CudaOp::AxisY;
using DirZ = CudaOp::AxisZ;
using CudaOp::forEachAxisRun;
using CudaOp::normalizeToVoxel;
using CudaOp::normalizeToVoxelBatch;

inline MainAxis getMainAxis(const float4& direction)
{
    return CudaOp::mainAxis(direction);
}

// Ray-driven BP 与 FP 的线程映射不同，因此保留独立调优值。
constexpr int kAnglesPerBlock = 4;
constexpr int kBlockSlices = 256;
constexpr int kDetBlockU = 256;
constexpr int kDetBlockV = 1;

} // namespace YK::Bp
