#include "IntensityOps.hpp"
#include "cuda/DeviceMemoryPool.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
namespace yk::spectral::postprocess::intensity_detail {
namespace {
    void check(cudaError_t e)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
    }
    class Lease {
        void* pointer_ = nullptr;

    public:
        explicit Lease(size_t bytes) { check(cuda_resources::poolAllocate(&pointer_, std::max(size_t(1), bytes))); }
        ~Lease() { cuda_resources::poolRelease(pointer_); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        template <class T>
        T* as() { return static_cast<T*>(pointer_); }
    };
    unsigned blocks(size_t n) { return unsigned(std::min(size_t(1024), (n + 255) / 256)); }
    __global__ void checkValues(const float* data, size_t n, double scale, double offset, int* error)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x) {
            double v = data[i], out = scale * v + offset;
            if (!isfinite(v) || !isfinite(out) || fabs(out) > 3.4028234663852886e38)
                atomicExch(error, 1);
        }
    }
    void validateValues(const float* data, size_t n, HuLinearCoefficients c)
    {
        Lease flag(sizeof(int));
        check(cudaMemsetAsync(flag.as<int>(), 0, sizeof(int), 0));
        checkValues<<<blocks(n), 256>>>(data, n, c.scale, c.offset, flag.as<int>());
        check(cudaGetLastError());
        int error = 0;
        check(cudaMemcpy(&error, flag.as<int>(), sizeof(int), cudaMemcpyDeviceToHost));
        if (error)
            throw std::runtime_error("nonfinite or overflowing intensity conversion input/output");
    }
    __global__ void linear(float* data, size_t n, double scale, double offset)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x)
            data[i] = float(scale * double(data[i]) + offset);
    }
    __global__ void thresholds(float* data, size_t n, const double* limits, int count, const int* labels)
    {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(blockDim.x) * gridDim.x) {
            int lo = 0, hi = count;
            double v = data[i];
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (v < limits[mid])
                    hi = mid;
                else
                    lo = mid + 1;
            }
            data[i] = float(labels[lo]);
        }
    }
    struct Box {
        int x, y, z, nx, ny, nz, vx, vy;
        size_t count;
    };
    __global__ void roiSums(const float* data, Box box, double* partial, int* error)
    {
        __shared__ double sums[256];
        double sum = 0;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < box.count; i += size_t(blockDim.x) * gridDim.x) {
            int x = box.x + int(i % box.nx), y = box.y + int(i / box.nx % box.ny), z = box.z + int(i / (size_t(box.nx) * box.ny));
            double v = data[(size_t(z) * box.vy + y) * box.vx + x];
            if (!isfinite(v))
                atomicExch(error, 1);
            else
                sum += v;
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (unsigned step = blockDim.x / 2; step; step /= 2) {
            if (threadIdx.x < step)
                sums[threadIdx.x] += sums[threadIdx.x + step];
            __syncthreads();
        }
        if (!threadIdx.x)
            partial[blockIdx.x] = sums[0];
    }
}
void cudaLinear(float* values, size_t n, HuLinearCoefficients c)
{
    validateValues(values, n, c);
    linear<<<blocks(n), 256>>>(values, n, c.scale, c.offset);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(0));
}
void cudaClassify(float* values, size_t n, const HuThresholdSegmentation& s)
{
    validateValues(values, n, {});
    Lease limits(s.thresholds_hu.size() * sizeof(double)), labels(s.classes.size() * sizeof(int));
    std::vector<int> ids;
    ids.reserve(s.classes.size());
    for (const auto& c : s.classes)
        ids.push_back(c.value);
    if (!s.thresholds_hu.empty())
        check(cudaMemcpy(limits.as<double>(), s.thresholds_hu.data(), s.thresholds_hu.size() * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(labels.as<int>(), ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    thresholds<<<blocks(n), 256>>>(values, n, limits.as<double>(), int(s.thresholds_hu.size()), labels.as<int>());
    check(cudaGetLastError());
    check(cudaStreamSynchronize(0));
}
double cudaMeanRoi(const float* data, const Grid& grid, const CalibrationRoi& r)
{
    Box box { r.begin[0], r.begin[1], r.begin[2], r.end[0] - r.begin[0], r.end[1] - r.begin[1], r.end[2] - r.begin[2], grid.dimensions[0], grid.dimensions[1], roiCount(r) };
    const unsigned count = blocks(box.count);
    Lease partial(count * sizeof(double)), flag(sizeof(int));
    check(cudaMemsetAsync(flag.as<int>(), 0, sizeof(int), 0));
    roiSums<<<count, 256>>>(data, box, partial.as<double>(), flag.as<int>());
    check(cudaGetLastError());
    int error = 0;
    check(cudaMemcpy(&error, flag.as<int>(), sizeof(int), cudaMemcpyDeviceToHost));
    if (error)
        throw std::runtime_error("nonfinite calibration ROI value");
    std::vector<double> sums(count);
    check(cudaMemcpy(sums.data(), partial.as<double>(), count * sizeof(double), cudaMemcpyDeviceToHost));
    double total = 0;
    for (double x : sums)
        total += x;
    return total / double(box.count);
}
}
