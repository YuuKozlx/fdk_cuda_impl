#include "StatisticalLaunch.cuh"
#include <cmath>
namespace YK::Iter::Statistical {
namespace {
    // Fixed allocation size and double accumulation avoid per-evaluation mallocs.
    template <int Kind>
    __global__ void scalar(const float* a, const float* b, const float* w, float* residual, size_t n, double* sums)
    {
        __shared__ double s[256];
        double sum = 0;
        bool invalid = false;
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
            const double x = a[i];
            double v = 0;
            if constexpr (Kind == 0) {
                const double weight = w ? w[i] : 1, measured = b ? b[i] : 0, r = x - measured;
                invalid = invalid || !isfinite(x) || !isfinite(measured) || !isfinite(weight) || weight < 0;
                v = .5 * weight * r * r;
                residual[i] = float(weight * r);
                invalid = invalid || !isfinite(residual[i]);
            } else if constexpr (Kind == 1) {
                double d = x - b[i];
                v = d * d;
            } else
                v = fabs(x);
            invalid = invalid || !isfinite(v);
            if constexpr (Kind == 2)
                sum = fmax(sum, v);
            else
                sum += v;
        }
        s[threadIdx.x] = invalid ? nan("") : sum;
        __syncthreads();
        for (int offset = 128; offset; offset /= 2) {
            if (threadIdx.x < offset) {
                if constexpr (Kind == 2) {
                    double v = s[threadIdx.x + offset];
                    s[threadIdx.x] = isfinite(v) && isfinite(s[threadIdx.x]) ? fmax(s[threadIdx.x], v) : nan("");
                } else
                    s[threadIdx.x] += s[threadIdx.x + offset];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
            sums[blockIdx.x] = s[0];
    }
    __global__ void update(float* out, const float* x, const float* g, size_t n, float step, float lo, float hi)
    {
        for (size_t i = size_t(blockIdx.x) * 256 + threadIdx.x; i < n; i += size_t(gridDim.x) * 256) {
            float v = x[i] - step * g[i];
            out[i] = isfinite(v) ? fminf(hi, fmaxf(lo, v)) : v;
        }
    }
}
void weighted_residual_launch(float* prediction, const float* b, const float* w, size_t n, double* sums, cudaStream_t s) { scalar<0><<<reduction_blocks, 256, 0, s>>>(prediction, b, w, prediction, n, sums); }
void squared_change_launch(const float* a, const float* b, size_t n, double* sums, cudaStream_t s) { scalar<1><<<reduction_blocks, 256, 0, s>>>(a, b, nullptr, nullptr, n, sums); }
void max_absolute_launch(const float* a, size_t n, double* sums, cudaStream_t s) { scalar<2><<<reduction_blocks, 256, 0, s>>>(a, nullptr, nullptr, nullptr, n, sums); }
void gradient_update_launch(float* out, const float* x, const float* g, size_t n, float step, float lo, float hi, cudaStream_t s) { update<<<reduction_blocks, 256, 0, s>>>(out, x, g, n, step, lo, hi); }
}
