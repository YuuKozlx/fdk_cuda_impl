#pragma once

// 兼容聚合头；具体 kernel 文件应包含对应算法的语义化 launch 头。
#include "CylFpBp/kernels/fdk/YkCylFdkLaunch.cuh"
#include "CylFpBp/kernels/joseph/YkCylJosephLaunch.cuh"
#include "CylFpBp/kernels/legacy/YkCylLegacyLaunch.cuh"
#include "CylFpBp/kernels/siddon/YkCylSiddonLaunch.cuh"
#include "CylFpBp/kernels/joseph/YkCylJosephBackprojectV3Launch.cuh"
