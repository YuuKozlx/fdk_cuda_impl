#pragma once

// 兼容聚合头；具体 kernel 文件应包含对应算法的语义化 launch 头。
#include "CylFpBp/kernels/YkCylFdkLaunch.cuh"
#include "CylFpBp/kernels/YkCylJosephLaunch.cuh"
#include "CylFpBp/kernels/YkCylLegacyLaunch.cuh"
#include "CylFpBp/kernels/YkCylSiddonLaunch.cuh"
#include "CylFpBp/kernels/YkCylVoxelDrivenLaunch.cuh"
