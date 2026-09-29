#include "DLTFpBp/kernels/YkDltAlgebraicLaunch.cuh"

#include "global/YkMacro.hpp"

namespace YK::DltFpBp {
namespace detail {

constexpr int kBlockSize = 256;

__global__ void fillOnesKernel(float* data, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) data[index] = 1.f;
}

__global__ void residualKernel(const float* measured, const float* forward,
    float* residual, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) residual[index] = measured[index] - forward[index];
}

__global__ void divideKernel(float* values, const float* denominator,
    float epsilon, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] /= denominator[index] + epsilon;
}

__global__ void updateKernel(float* volume, const float* backprojection,
    const float* column_weight, float relaxation, float epsilon, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        volume[index] += relaxation * backprojection[index] /
            (column_weight[index] + epsilon);
}

__global__ void clampMinKernel(float* values, float minimum, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && values[index] < minimum) values[index] = minimum;
}

__global__ void meanZKernel(const float* volume, float* image,
    int nx, int ny, int nz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t slice = static_cast<size_t>(nx) * ny;
    const size_t xy = static_cast<size_t>(y) * nx + x;
    float sum = 0.f;
    for (int z = 0; z < nz; ++z)
        sum += volume[static_cast<size_t>(z) * slice + xy];
    image[xy] = sum / static_cast<float>(nz);
}

__global__ void thresholdInfKernel(float* values, float threshold, size_t count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && values[index] <= threshold) values[index] = 1e30f;
}

__global__ void update2dKernel(float* volume, const float* backprojection,
    const float* column_weight_2d, float relaxation, float epsilon,
    int nx, int ny, int nz)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    if (index >= count) return;
    const size_t xy = index % (static_cast<size_t>(nx) * ny);
    volume[index] += relaxation * backprojection[index] /
        (column_weight_2d[xy] + epsilon);
}

inline dim3 linearGrid(size_t count)
{
    return dim3(static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize));
}

} // namespace detail

void dltFillOnesLaunch(float* data, size_t count, cudaStream_t stream)
{
    detail::fillOnesKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        data, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltResidualLaunch(const float* measured, const float* forward,
    float* residual, size_t count, cudaStream_t stream)
{
    detail::residualKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        measured, forward, residual, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltDivideLaunch(float* values, const float* denominator, float epsilon,
    size_t count, cudaStream_t stream)
{
    detail::divideKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        values, denominator, epsilon, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltUpdateLaunch(float* volume, const float* backprojection,
    const float* column_weight, float relaxation, float epsilon,
    size_t count, cudaStream_t stream)
{
    detail::updateKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        volume, backprojection, column_weight, relaxation, epsilon, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltClampMinLaunch(float* values, float minimum, size_t count,
    cudaStream_t stream)
{
    detail::clampMinKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        values, minimum, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltMeanZLaunch(const float* volume, float* image,
    int nx, int ny, int nz, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((nx + block.x - 1) / block.x,
        (ny + block.y - 1) / block.y);
    detail::meanZKernel<<<grid, block, 0, stream>>>(volume, image, nx, ny, nz);
    YK_CUDA_KERNEL_CHECK();
}

void dltThresholdInfLaunch(float* values, float threshold, size_t count,
    cudaStream_t stream)
{
    detail::thresholdInfKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        values, threshold, count);
    YK_CUDA_KERNEL_CHECK();
}

void dltUpdate2dLaunch(float* volume, const float* backprojection,
    const float* column_weight_2d, float relaxation, float epsilon,
    int nx, int ny, int nz, cudaStream_t stream)
{
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    detail::update2dKernel<<<detail::linearGrid(count), detail::kBlockSize, 0, stream>>>(
        volume, backprojection, column_weight_2d, relaxation, epsilon,
        nx, ny, nz);
    YK_CUDA_KERNEL_CHECK();
}

} // namespace YK::DltFpBp
