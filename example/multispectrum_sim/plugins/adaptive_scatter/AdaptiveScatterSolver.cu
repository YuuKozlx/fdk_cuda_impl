#include "recon/preprocess/DeviceMemoryPool.hpp"
#include "AdaptiveScatterSolver.hpp"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace yk::spectral::experimental {
namespace {
struct SortAllocator {
    using value_type=char;
    char* allocate(std::ptrdiff_t bytes){char* p=nullptr;auto status=preprocess::poolAllocate(&p,size_t(bytes));if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));return p;}
    void deallocate(char* p,size_t){preprocess::poolRelease(p);}
};
constexpr int threads = 256;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(
        std::string("adaptive scatter CUDA: ") + cudaGetErrorString(error));
}

__global__ void transmission(const float* measured,const float* air,const float* primary,
    const unsigned char* mask,double* output,size_t n,int* invalid) {
    size_t p=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(p>=n)return;
    if(!isfinite(measured[p])||measured[p]<0||!isfinite(air[p])||air[p]<=0||
        !isfinite(primary[p])||primary[p]<air[p]*1e-8)atomicExch(invalid,1);
    output[p]=(!mask||mask[p])?double(measured[p])/air[p]:INFINITY;
}
__global__ void update(const float* current, float* candidate,
                        const float* measured, const float* air,
                        const float* scatter, size_t count, double scale, double damping) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count) return;
    const double residual = static_cast<double>(measured[p]) - current[p]
        - scale * fmax(0.0, static_cast<double>(scatter[p]));
    candidate[p] = static_cast<float>(current[p] + damping * residual);
}

// Partial sums keep the global reduction deterministic and avoid atomics.
__global__ void metrics(const float* value, const float* previous,
                         const float* measured, const float* scatter, const float* air,
                         size_t count, double scale, double margin, double* sums) {
    __shared__ double partial[5][threads];
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    double residual = 0.0, delta = 0.0, signal = 0.0, old = 0.0;
    if (p < count) {
        residual = static_cast<double>(measured[p]) - value[p] -
            scale * fmax(0.0, static_cast<double>(scatter[p]));
        if (!isfinite(value[p]) || value[p] < static_cast<double>(air[p]) * 1e-8)
            residual = NAN;
        delta = static_cast<double>(value[p]) - previous[p];
        signal = measured[p];
        old = previous[p];
    }
    partial[0][threadIdx.x] = residual * residual;
    partial[1][threadIdx.x] = delta * delta;
    partial[2][threadIdx.x] = signal * signal;
    partial[3][threadIdx.x] = old * old;
    // Round the floor upward so conversion of the candidate back to float
    // cannot cross the mathematical positivity bound at tiny step sizes.
    const float positive_floor = p < count
        ? nextafterf(static_cast<float>(static_cast<double>(air[p]) * 1e-8), INFINITY) : 0.0f;
    partial[4][threadIdx.x] = p < count && residual < 0.0
        ? margin * fmax(0.0, static_cast<double>(value[p]) - positive_floor) / -residual
        : 1.0;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride)
            for (int k = 0; k < 4; ++k)
                partial[k][threadIdx.x] += partial[k][threadIdx.x + stride];
        if (threadIdx.x < stride)
            partial[4][threadIdx.x] = fmin(partial[4][threadIdx.x], partial[4][threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int k = 0; k < 5; ++k) sums[static_cast<size_t>(blockIdx.x) * 5 + k] = partial[k][0];
}
}

AdaptiveScatterSolver::AdaptiveScatterSolver(size_t pixels) : pixels_(pixels) {
    if (!pixels || pixels > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("invalid adaptive scatter image size");
    try {
        check(yk::spectral::preprocess::poolAllocate(&transmissions_,pixels*sizeof(double)));
        check(yk::spectral::preprocess::poolAllocate(&mask_,pixels));
        check(yk::spectral::preprocess::poolAllocate(&current_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&candidate_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&measured_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&air_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&scatter_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&accepted_scatter_, pixels * sizeof(float)));
        check(yk::spectral::preprocess::poolAllocate(&statistics_, ((pixels + threads - 1) / threads) * 5 * sizeof(double)));
    } catch (...) {
        yk::spectral::preprocess::poolRelease(transmissions_);yk::spectral::preprocess::poolRelease(mask_);
        yk::spectral::preprocess::poolRelease(current_); yk::spectral::preprocess::poolRelease(candidate_); yk::spectral::preprocess::poolRelease(measured_);
        yk::spectral::preprocess::poolRelease(air_); yk::spectral::preprocess::poolRelease(scatter_); yk::spectral::preprocess::poolRelease(statistics_);
        yk::spectral::preprocess::poolRelease(accepted_scatter_);
        throw;
    }
}

AdaptiveScatterSolver::~AdaptiveScatterSolver() {
    yk::spectral::preprocess::poolRelease(transmissions_);yk::spectral::preprocess::poolRelease(mask_);
    yk::spectral::preprocess::poolRelease(current_); yk::spectral::preprocess::poolRelease(candidate_); yk::spectral::preprocess::poolRelease(measured_);
    yk::spectral::preprocess::poolRelease(air_); yk::spectral::preprocess::poolRelease(scatter_); yk::spectral::preprocess::poolRelease(statistics_);
    yk::spectral::preprocess::poolRelease(accepted_scatter_);
}

AdaptiveScatterReport AdaptiveScatterSolver::solve(const std::vector<float>& m,
    const std::vector<float>& a,std::vector<float>& p,
    const ScatterCorrectionConfig::SolverConfig::AdaptiveConfig& o,double scale,
    const ScatterPredictor& predict,const std::vector<unsigned char>* mask,
    const DeviceScatterPredictor& device_predict) {
    return solveImpl(m,a,p,o,scale,predict,mask,device_predict);
}
AdaptiveScatterReport AdaptiveScatterSolver::solveDevice(const float* m,const float* a,float* p,
    const ScatterCorrectionConfig::SolverConfig::AdaptiveConfig& o,double scale,
    const DeviceScatterPredictor& predict,const std::vector<unsigned char>* mask) {
    if(!m||!a||!p||!predict)throw std::runtime_error("invalid adaptive scatter device buffers");
    std::vector<float> empty;
    return solveImpl(empty,empty,empty,o,scale,{},mask,predict,m,a,p);
}
AdaptiveScatterReport AdaptiveScatterSolver::solveImpl(const std::vector<float>& measured,
    const std::vector<float>& air, std::vector<float>& primary,
    const ScatterCorrectionConfig::SolverConfig::AdaptiveConfig& options,
    double scale, const ScatterPredictor& predict, const std::vector<unsigned char>* object_mask,
    const DeviceScatterPredictor& device_predict,
    const float* device_measured,const float* device_air,float* device_primary) {
    if (!device_primary && (measured.size() != pixels_ || air.size() != pixels_ || primary.size() != pixels_))
        throw std::runtime_error("adaptive scatter image size mismatch");
    if (object_mask && object_mask->size() != pixels_)
        throw std::runtime_error("adaptive scatter mask size mismatch");
    const auto& guard = options.transmission_guard;
    if (!std::isfinite(guard.low_quantile) || guard.low_quantile <= 0 || guard.low_quantile > 1 ||
        !std::isfinite(guard.risk_threshold) || guard.risk_threshold <= 0 || guard.risk_threshold >= 1 ||
        !std::isfinite(guard.positivity_margin) || guard.positivity_margin <= 0 || guard.positivity_margin >= 1)
        throw std::runtime_error("invalid adaptive scatter transmission guard");
    if (!std::isfinite(options.damping_decay) || options.damping_decay <= 0 || options.damping_decay > 1 ||
        options.iterations <= 0 || !std::isfinite(options.initial_damping) ||
        options.initial_damping <= 0 || options.initial_damping > 1 ||
        !std::isfinite(options.min_damping) || options.min_damping <= 0 ||
        options.min_damping > options.initial_damping || !std::isfinite(scale) || scale < 0)
        throw std::runtime_error("invalid adaptive scatter solver options");
    if (!device_primary) for (size_t p = 0; p < pixels_; ++p)
        if (!std::isfinite(measured[p]) || measured[p] < 0 ||
            !std::isfinite(air[p]) || air[p] <= 0 || !std::isfinite(primary[p]) || primary[p] < air[p] * 1e-8)
            throw std::runtime_error("invalid adaptive scatter input signal");
    const size_t bytes = pixels_ * sizeof(float);
    const int blocks = static_cast<int>((pixels_ + threads - 1) / threads);
    check(cudaMemcpy(current_, device_primary?device_primary:primary.data(), bytes, device_primary?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
    check(cudaMemcpy(measured_, device_measured?device_measured:measured.data(), bytes, device_primary?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
    check(cudaMemcpy(air_, device_air?device_air:air.data(), bytes, device_primary?cudaMemcpyDeviceToDevice:cudaMemcpyHostToDevice));
    std::vector<float> prediction, candidate(device_predict?0:pixels_);
    std::vector<double> sums(static_cast<size_t>(blocks) * 5);
    AdaptiveScatterReport report;
    report.damping = options.initial_damping;
    if(device_primary) {
        if(object_mask)check(cudaMemcpy(mask_,object_mask->data(),pixels_,cudaMemcpyHostToDevice));
        check(cudaMemset(statistics_,0,sizeof(int)));
        transmission<<<blocks,threads>>>(measured_,air_,current_,object_mask?mask_:nullptr,
            transmissions_,pixels_,reinterpret_cast<int*>(statistics_));
        check(cudaGetLastError());int invalid=0;
        check(cudaMemcpy(&invalid,statistics_,sizeof(int),cudaMemcpyDeviceToHost));
        if(invalid)throw std::runtime_error("invalid adaptive scatter input signal");
        size_t count=object_mask?size_t(std::count_if(object_mask->begin(),object_mask->end(),[](auto v){return v!=0;})):pixels_;
        if(count){
            thrust::device_ptr<double> first(transmissions_);SortAllocator allocator;thrust::sort(thrust::cuda::par(allocator),first,first+pixels_);
            size_t index=size_t(guard.low_quantile*(count-1));
            check(cudaMemcpy(&report.low_transmission,transmissions_+index,sizeof(double),cudaMemcpyDeviceToHost));
            report.low_transmission_risk=report.low_transmission<guard.risk_threshold;
        }
    }
    std::vector<double> transmissions;
    transmissions.reserve(pixels_);
    if(!device_primary) for (size_t p = 0; p < pixels_; ++p)
        if (!object_mask || (*object_mask)[p])
            transmissions.push_back(static_cast<double>(measured[p]) / air[p]);
    if (!transmissions.empty()) {
        const size_t index = static_cast<size_t>(guard.low_quantile * (transmissions.size() - 1));
        std::nth_element(transmissions.begin(), transmissions.begin() + index, transmissions.end());
        report.low_transmission = transmissions[index];
        report.low_transmission_risk = report.low_transmission < guard.risk_threshold;
    }
    auto evaluate = [&](const std::vector<float>& value, float* device_value) {
        if (device_predict) {
            device_predict(device_value, air_, scatter_);
        } else {
            predict(value, prediction);
            if (prediction.size() != pixels_) throw std::runtime_error("adaptive scatter prediction size mismatch");
            for (float v : prediction)
                if (!std::isfinite(v)) throw std::runtime_error("adaptive scatter prediction is not finite");
            check(cudaMemcpy(scatter_, prediction.data(), bytes, cudaMemcpyHostToDevice));
        }
        ++report.predictions;
        metrics<<<blocks, threads>>>(device_value, current_, measured_, scatter_, air_, pixels_,
                                     scale, guard.positivity_margin, statistics_);
        check(cudaGetLastError());
        check(cudaMemcpy(sums.data(), statistics_, sums.size() * sizeof(double), cudaMemcpyDeviceToHost));
        double total[4] = {};
        double safe_damping = 1.0;
        for (int b = 0; b < blocks; ++b)
        {
            for (int k = 0; k < 4; ++k) total[k] += sums[static_cast<size_t>(b) * 5 + k];
            safe_damping = std::min(safe_damping, sums[static_cast<size_t>(b) * 5 + 4]);
        }
        const double residual = std::sqrt(total[0] / std::max(total[2], 1e-300));
        const double change = std::sqrt(total[1] / std::max(total[3], 1e-300));
        if (!std::isfinite(residual) || !std::isfinite(change))
            throw std::runtime_error("adaptive scatter metrics are not finite");
        return std::array<double, 3>{residual, change, safe_damping};
    };
    auto current_metrics = evaluate(primary, current_);
    report.relative_residual = current_metrics[0];
    report.history.push_back({0, 0.0, report.relative_residual, 0.0});
    check(cudaMemcpy(accepted_scatter_, scatter_, bytes, cudaMemcpyDeviceToDevice));
    const auto finish = [&]() {
        if (device_predict)
            check(cudaMemcpy(device_primary?device_primary:primary.data(), current_, bytes, device_primary?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost));
        return report;
    };
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
        // Transmission flags risk before prediction; the actual residual gives
        // a positivity bound even when the total signal conceals that risk.
        const double ceiling = options.initial_damping * std::pow(options.damping_decay, iteration);
        const double proposed = iteration == 0 ? ceiling
            : std::min(ceiling, report.damping * 1.25);
        report.damping = std::min(proposed, current_metrics[2]);
        if (report.damping < proposed) ++report.positivity_limited_steps;
        if (!(report.damping > 0)) {
            report.status = "positivity_stalled";
            return finish();
        }
        const double minimum_trial = std::min(options.min_damping, report.damping);
        for (;;) {
            update<<<blocks, threads>>>(current_, candidate_, measured_, air_, accepted_scatter_,
                                       pixels_, scale, report.damping);
            check(cudaGetLastError());
            if (!device_predict) {
                check(cudaMemcpy(candidate.data(), candidate_, bytes, cudaMemcpyDeviceToHost));
                for (size_t p = 0; p < pixels_; ++p)
                    if (!std::isfinite(candidate[p]) || candidate[p] < air[p] * 1e-8)
                        throw std::runtime_error("adaptive scatter positivity bound failed");
            }
            const auto candidate_metrics = evaluate(candidate, candidate_);
            if (candidate_metrics[0] <= report.relative_residual) {
                if (!device_predict) primary.swap(candidate);
                std::swap(current_, candidate_);
                std::swap(accepted_scatter_, scatter_);
                ++report.iterations;
                report.relative_residual = candidate_metrics[0];
                report.relative_change = candidate_metrics[1];
                current_metrics = candidate_metrics;
                report.history.push_back({report.iterations, report.damping,
                                          report.relative_residual, report.relative_change});
                break;
            }
            ++report.rejected_steps;
            if (report.damping <= minimum_trial) {
                report.status = "stalled";
                return finish();
            }
            report.damping = std::max(minimum_trial, report.damping * 0.5);
        }
    }
    return finish();
}
}
