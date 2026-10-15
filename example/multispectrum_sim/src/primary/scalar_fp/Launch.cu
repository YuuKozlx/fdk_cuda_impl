#include "primary/ScalarProjector.hpp"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <stdexcept>
namespace yk::spectral::scalar_fp {
void validateGrid(const Grid&, int, int, int, Method);
namespace {
#include "Joseph.cuh"
#include "Siddon.cuh"
    void checked(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    template <class Axis, int Steps>
    void joseph(cudaTextureObject_t texture, const DeviceView* views, float* out, const KernelGrid& g, int nu, int nv, int count, bool accumulate, cudaStream_t stream)
    {
        const int slices = Axis::nSlices(g.Nx, g.Ny, g.Nz) * Steps;
        const dim3 block(kDetBlockU, kAnglesPerBlock, kDetBlockV);
        const dim3 blocks(((nu + kDetBlockU - 1) / kDetBlockU) * ((nv + kDetBlockV - 1) / kDetBlockV), (count + kAnglesPerBlock - 1) / kAnglesPerBlock);
        for (int s = 0; s < slices; s += kBlockSlices) {
            fp_joseph_kernel<Axis, Steps><<<blocks, block, 0, stream>>>(texture, views, out, slices, Axis::nDim1(g.Nx, g.Ny, g.Nz), Axis::nDim2(g.Nx, g.Ny, g.Nz), nu, nv, s, 0, count, g, accumulate || s != 0);
            checked(cudaGetLastError());
        }
    }
    template <int Steps>
    void josephAll(cudaTextureObject_t texture, const DeviceView* views, float* out, const KernelGrid& g, int nu, int nv, int count, bool accumulate, cudaStream_t stream)
    {
        joseph<AxisX, Steps>(texture, views, out, g, nu, nv, count, accumulate, stream);
        joseph<AxisY, Steps>(texture, views, out, g, nu, nv, count, accumulate, stream);
        joseph<AxisZ, Steps>(texture, views, out, g, nu, nv, count, accumulate, stream);
    }
}
void launch(Method method, const float* volume, cudaTextureObject_t texture, const DeviceView* views, float* out, const Grid& g, int nu, int nv, int count, bool accumulate, cudaStream_t stream, int steps)
{
    validateGrid(g, nu, nv, count, method);
    if (!views || !out || (method == Method::Siddon && !volume) || (method == Method::Joseph && !texture) || (steps != 1 && steps != 2 && steps != 4) || (method == Method::Siddon && steps != 1))
        throw std::invalid_argument("invalid scalar FP buffers/sampling");
    if (g.nx > INT_MAX / steps || g.ny > INT_MAX / steps || g.nz > INT_MAX / steps)
        throw std::invalid_argument("FP step index overflow");
    const size_t pixels = size_t(nu) * nv;
    for (int first = 0; first < count;) {
        const int n = std::min(16384, count - first);
        auto* output = out + size_t(first) * pixels;
        if (method == Method::Siddon) {
            if (!accumulate)
                checked(cudaMemsetAsync(output, 0, size_t(n) * pixels * sizeof(float), stream));
            float3 origin = { g.center.x - .5f * (g.nx - 1) * g.spacing.x, g.center.y - .5f * (g.ny - 1) * g.spacing.y, g.center.z - .5f * (g.nz - 1) * g.spacing.z };
            siddon_fp_kernel<<<dim3((nu + 15) / 16, (nv + 15) / 16, n), dim3(16, 16), 0, stream>>>(volume, output, views + first, origin, g.spacing.x, g.spacing.y, g.spacing.z, g.nx, g.ny, g.nz, nu, nv, n, accumulate);
            checked(cudaGetLastError());
        } else {
            KernelGrid kg { g.nx, g.ny, g.nz, g.spacing.x, g.spacing.y, g.spacing.z };
            if (steps == 1)
                josephAll<1>(texture, views + first, output, kg, nu, nv, n, accumulate, stream);
            else if (steps == 2)
                josephAll<2>(texture, views + first, output, kg, nu, nv, n, accumulate, stream);
            else
                josephAll<4>(texture, views + first, output, kg, nu, nv, n, accumulate, stream);
        }
        first += n;
    }
}
}
