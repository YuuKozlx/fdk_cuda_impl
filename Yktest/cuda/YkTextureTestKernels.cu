#include "cuda/YkTextureTestKernels.cuh"

#include <cstddef>

#include "global/YkMacro.hpp"

namespace YK::Test {
namespace {

__global__ void pointTextureReadbackKernel(cudaTextureObject_t texture,
    float* output, size_t count, int channels, int rows)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
            threadIdx.x; index < count; index += stride) {
        const int channel = static_cast<int>(index % channels);
        const size_t view_row = index / channels;
        const int row = static_cast<int>(view_row % rows);
        const int view = static_cast<int>(view_row / rows);
        output[index] = tex3D<float>(texture, channel + 0.5f, row + 0.5f,
            view + 0.5f);
    }
}

} // namespace

void pointTextureReadback(cudaTextureObject_t texture, float* output,
    int channels, int rows, int views, cudaStream_t stream)
{
    if (!texture || !output || channels <= 0 || rows <= 0 || views <= 0)
        return;
    const size_t count = static_cast<size_t>(channels) * rows * views;
    constexpr int block = 256;
    const int grid = static_cast<int>((count + block - 1) / block);
    pointTextureReadbackKernel<<<grid, block, 0, stream>>>(texture, output,
        count, channels, rows);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::Test
