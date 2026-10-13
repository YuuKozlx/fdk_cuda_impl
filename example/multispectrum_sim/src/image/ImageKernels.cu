#include "ImageBackend.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace yk::spectral::image::detail {
template <class F>
__global__ void evaluate(float* out, size_t count, F f)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
        out[i] = f(i);
}
template <class F>
void launch(float* out, size_t n, F f)
{
    unsigned blocks = unsigned((n + 255) / 256);
    if (blocks > 65535)
        blocks = 65535;
    evaluate<<<blocks, 256>>>(out, n, f);
    auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void cudaAxis(float* o, size_t n, AxisFilter f) { launch(o, n, f); }
void cudaResize(float* o, size_t n, ResizeAxis f) { launch(o, n, f); }
void cudaBilateral(float* o, size_t n, Bilateral f) { launch(o, n, f); }
void cudaPointwise(float* o, size_t n, Pointwise f) { launch(o, n, f); }
} // namespace
