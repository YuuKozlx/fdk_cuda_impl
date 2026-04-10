#include <cuda_runtime_api.h>
#include "YkConv.hpp"

namespace YK {
    namespace Filter {
        namespace detail {
            __global__ void _kernel_pointwise_mul(cufftComplex* data, const float* weights, int n_complex, int batch) {
                int u = blockIdx.x * blockDim.x + threadIdx.x;
                int b = blockIdx.y;
                if (u < n_complex && b < batch) {
                    int idx = b * n_complex + u;
                    float w = weights[u];
                    data[idx].x *= w;
                    data[idx].y *= w;
                }
            }

        };


        void launch_pointwise_mul(cufftComplex* data, const float* weights, int n_complex, int batch, cudaStream_t stream) {
            dim3 block(256, 1);
            dim3 grid((n_complex + 255) / 256, batch);
            detail::_kernel_pointwise_mul << <grid, block, 0, stream >> > (data, weights, n_complex, batch);

            YK_CUDA_CHECK(cudaGetLastError());
        }
    }
}