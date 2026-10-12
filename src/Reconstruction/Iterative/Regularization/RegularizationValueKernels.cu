#include "Reconstruction/Iterative/Common/YkReductionWorkspace.hpp"
#include "RegularizationValueLaunch.cuh"
#include <cmath>
namespace YK::Iter {
namespace {
    __global__ void regularization_value(const float* u, int nx, int ny, int nz, float hx, float hy, float hz, float shape, int dimensions, int kind, size_t n, double* sums)
    {
        __shared__ double s[256];
        double sum = 0;
        for (size_t id = size_t(blockIdx.x) * 256 + threadIdx.x; id < n; id += size_t(gridDim.x) * 256) {
            const int i = int(id % nx), j = int(id / nx % ny), k = int(id / (size_t(nx) * ny));
            double v = u[id];
            if (kind == 2) {
                sum += .5 * v * v;
                continue;
            }
            const double dx = i + 1 < nx ? (u[id + 1] - v) / hx : 0, dy = j + 1 < ny ? (u[id + nx] - v) / hy : 0,
                         dz = dimensions == 3 && k + 1 < nz ? (u[id + size_t(nx) * ny] - v) / hz : 0;
            const double norm = sqrt(dx * dx + dy * dy + dz * dz);
            sum += kind == 0 ? (norm <= shape ? norm * norm / (2 * shape) : norm - shape / 2) : kind == 1 ? sqrt(norm * norm + double(shape) * shape) - shape
                                                                                                          : norm;
        }
        s[threadIdx.x] = sum;
        __syncthreads();
        for (int offset = 128; offset; offset /= 2) {
            if (threadIdx.x < offset)
                s[threadIdx.x] += s[threadIdx.x + offset];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            sums[blockIdx.x] = s[0];
    }
}
void regularization_value_launch(const float* u, int nx, int ny, int nz, float hx, float hy, float hz, float shape, int dimensions, int kind, double* sums, cudaStream_t s) { regularization_value<<<scalar_reduction_blocks, 256, 0, s>>>(u, nx, ny, nz, hx, hy, hz, shape, dimensions, kind, size_t(nx) * ny * nz, sums); }
}
