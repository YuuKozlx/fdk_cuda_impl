#include "NonStandardFpBp/SourceOrbitTomography/kernels/YkSourceOrbitResampleKernels.cuh"

namespace YK::NonStandardFpBp {
namespace {

__device__ float read_or_zero(const float* raw, int raw_u, int raw_v,
    int view, int u, int v)
{
    if (u < 0 || u >= raw_u || v < 0 || v >= raw_v) return 0.f;
    return raw[(static_cast<size_t>(view) * raw_v + v) * raw_u + u];
}

__global__ void resample_kernel(const float* raw, float* window,
    int raw_u, int raw_v, int window_u, int window_v, int first_view,
    const SourceOrbitResampleMap* maps)
{
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = blockIdx.y * blockDim.y + threadIdx.y;
    const int view = blockIdx.z;
    if (u >= window_u || v >= window_v) return;
    const int source_view = first_view + view;
    const SourceOrbitResampleMap map = maps[source_view];
    const float denominator = map.denominator.x +
        u * map.denominator.y + v * map.denominator.z;
    if (fabsf(denominator) < 1e-12f) {
        window[(static_cast<size_t>(view) * window_v + v) * window_u + u] = 0.f;
        return;
    }
    const float raw_x = (map.numerator_u.x + u * map.numerator_u.y +
        v * map.numerator_u.z) / denominator;
    const float raw_y = (map.numerator_v.x + u * map.numerator_v.y +
        v * map.numerator_v.z) / denominator;
    const int x0 = static_cast<int>(floorf(raw_x));
    const int y0 = static_cast<int>(floorf(raw_y));
    const float tx = raw_x - x0;
    const float ty = raw_y - y0;
    const float p00 = read_or_zero(raw, raw_u, raw_v, source_view, x0, y0);
    const float p10 = read_or_zero(raw, raw_u, raw_v, source_view, x0 + 1, y0);
    const float p01 = read_or_zero(raw, raw_u, raw_v, source_view, x0, y0 + 1);
    const float p11 = read_or_zero(raw, raw_u, raw_v, source_view, x0 + 1, y0 + 1);
    const size_t dst = (static_cast<size_t>(view) * window_v + v) * window_u + u;
    window[dst] = (1.f - ty) * ((1.f - tx) * p00 + tx * p10) +
        ty * ((1.f - tx) * p01 + tx * p11);
}

} // namespace

bool launchSourceOrbitResample(const float* d_raw, float* d_window,
    int raw_u, int raw_v, int window_u, int window_v, int first_view, int views,
    const SourceOrbitResampleMap* d_maps, cudaStream_t stream)
{
    if (!d_raw || !d_window || !d_maps || raw_u <= 0 || raw_v <= 0 ||
        window_u <= 0 || window_v <= 0 || first_view < 0 || views <= 0 ||
        window_u > raw_u || window_v > raw_v) return false;
    const dim3 block(16, 16, 1);
    const dim3 grid((window_u + block.x - 1) / block.x,
        (window_v + block.y - 1) / block.y, views);
    resample_kernel<<<grid, block, 0, stream>>>(d_raw, d_window, raw_u, raw_v,
        window_u, window_v, first_view, d_maps);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace YK::NonStandardFpBp
