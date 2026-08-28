#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace YK::CudaOp {

// 投影数组统一布局为 [view][row][channel]。view_offset 使同一个 Reader
// 同时适用于完整纹理和由 pipeline 建立的连续视图分包纹理。
struct RawProjectionPointReader {
    const float* data = nullptr;
    int view_offset = 0;

    __device__ float read(int view, int row, int channel,
        int channels, int rows) const
    {
        return data[((static_cast<size_t>(view + view_offset) * rows + row) *
            channels) + channel];
    }
};

struct TextureProjectionPointReader {
    cudaTextureObject_t texture = 0;
    int view_offset = 0;

    __device__ float read(int view, int row, int channel, int, int) const
    {
        return tex3D<float>(texture, channel + 0.5f, row + 0.5f,
            view + view_offset + 0.5f);
    }
};

} // namespace YK::CudaOp
