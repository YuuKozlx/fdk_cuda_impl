#pragma once

#include <cuda_runtime.h>

namespace YK::Test {

// 按整数像素中心逐点读回 Point 纹理，仅用于区分纹理上传误差和 BP 的
// atomicAdd 累加误差。该辅助接口不属于正式投影算子。
void pointTextureReadback(cudaTextureObject_t texture, float* output,
    int channels, int rows, int views, cudaStream_t stream);

} // namespace YK::Test
