#pragma once

#include "common/cuda/operators/YkOperatorKernelTypes.cuh"

namespace YK::Fp {

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

// 这些值描述 Flat Joseph FP 的算法映射，不属于设备安全启动上限。
constexpr int kAnglesPerBlock = 4;
constexpr int kBlockSlices = 256;
constexpr int kDetBlockU = 32;
constexpr int kDetBlockV = 8;

} // namespace YK::Fp
