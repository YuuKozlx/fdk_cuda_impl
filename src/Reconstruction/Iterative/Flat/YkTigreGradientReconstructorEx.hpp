#pragma once

#include "Reconstruction/Analytic/Circular/Flat/FDK/YkFdkPipeline.hpp"
#include "Reconstruction/Iterative/Flat/YkAlgebraicBackends.hpp"
#include "Reconstruction/Iterative/Common/kernels/YkIterLaunch.cuh"
#include "Reconstruction/Iterative/Common/kernels/YkTvRegularizationLaunch.cuh"
#include "common/YkDeviceWorkspace.hpp"
#include "global/YkLog.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace YK::Iter {

enum class ETigreGradientAlgorithm : int {
    Sart = 0,
    OsSart,
    Sirt,
    AsdPocs,
    OsAsdPocs,
    BAsdPocsBeta,
    Pcsd,
    OsPcsd,
    AwPcsd,
    OsAwPcsd,
    AwAsdPocs,
    OsAwAsdPocs
};

enum class ETigreInitialization : int {
    Zero = 0,
    Fdk,
    DeviceVolume
};

enum class ETigreRelaxationMode : int {
    Scalar = 0,
    Nesterov
};

struct TigreGradientConfig {
    ETigreGradientAlgorithm algorithm = ETigreGradientAlgorithm::OsSart;
    int iterations = 10;
    // TIGRE 的 blocksize 是每个块包含的视角数，而不是块数量。
    int block_size = 20;
    float lambda = 1.f;
    float lambda_reduction = 1.f;
    ETigreRelaxationMode relaxation_mode = ETigreRelaxationMode::Scalar;
    ETigreInitialization initialization = ETigreInitialization::Zero;
    const float* d_initial_volume = nullptr;
    bool non_negative = true;

    int tv_iterations = 20;
    float alpha = 0.002f;
    float alpha_reduction = 0.95f;
    float maximum_update_ratio = 0.95f;
    // 负值表示按 TIGRE 规则自动估计：0.2 * ||A(FDK(b))-b||_2。
    // 兼容性注意：ASD-POCS 将它作为 L2 阈值 epsilon 与 dd 比较；PCSD
    // 的 TIGRE MATLAB/Python 实现则保留历史判断 d_p^2 > epsilon。
    // 因此本字段不能简单解释成所有算法统一的“L2 范数阈值”。
    float max_l2_error = -1.f;
    float minimum_beta = 0.005f;
    float tv_epsilon = 1e-8f;
    float adaptive_delta = -0.005f;

    float bregman_beta = 1.f;
    float bregman_beta_reduction = 0.75f;
    int bregman_interval = 5;

    ETask fp_task = ETask::FP_Joseph;
    ETask bp_task = ETask::BP_FDK_matched;
};

struct TigreGradientStatistics {
    int completed_iterations = 0;
    unsigned int subset_updates = 0;
    float projection_l2 = std::numeric_limits<float>::quiet_NaN();
    float data_update_l2 = std::numeric_limits<float>::quiet_NaN();
    float regularization_update_l2 = std::numeric_limits<float>::quiet_NaN();
    float direction_cosine = std::numeric_limits<float>::quiet_NaN();
    float beta = 1.f;
    float tv_step = 0.f;
    bool stopped_by_projection_condition = false;
    bool stopped_by_beta = false;
    bool stopped_by_nesterov_residual = false;
};

// TIGRE MATLAB/Python 梯度类算法的统一显式 geometry 求解器。
// 数据步复用 TIGRE 权重后端；POCS/PCSD 的参数更新与停止条件由本类管理，
// 不把它们降格成一次 regularizer.apply() 后处理。
class TigreGradientReconstructorEx {
public:
    using Config = TigreGradientConfig;
    ~TigreGradientReconstructorEx() { release(); }
    TigreGradientReconstructorEx() = default;
    TigreGradientReconstructorEx(const TigreGradientReconstructorEx&) = delete;
    TigreGradientReconstructorEx& operator=(const TigreGradientReconstructorEx&) = delete;

    bool prepare(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const Config& config, cudaStream_t stream, int device_id = 0)
    {
        release();
        if (!validate_(params, geometry, config, stream)) return false;
        params_ = params;
        geometry_ = geometry;
        config_ = config;
        stream_ = stream;
        device_id_ = device_id;
        volume_count_ = static_cast<size_t>(params.volume.Nx) * params.volume.Ny * params.volume.Nz;
        projection_count_ = static_cast<size_t>(params.scan.NAng) * params.scan.Nu * params.scan.Nv;

        const int block_size = resolvedBlockSize_(config, params.scan.NAng);
        requested_subset_count_ = (params.scan.NAng + block_size - 1) / block_size;
        AlgebraicTigreBackend::Config data_config{};
        data_config.n_iter = config.iterations;
        data_config.n_subset = requested_subset_count_;
        data_config.lambda = config.relaxation_mode == ETigreRelaxationMode::Nesterov
            ? 1.f : config.lambda;
        data_config.lambda_red = 1.f;
        data_config.eps = 0.f;
        data_config.use_min = config.non_negative;
        data_config.min_constraint = 0.f;
        data_config.fp_task = config.fp_task;
        data_config.bp_task = config.bp_task;
        if (!data_backend_.init(params_, data_config, geometry_, stream_, device_id_))
            return false;
        actual_subset_count_ = data_backend_.actualSubsetCount();

        if (!fp_.init(params_, geometry_, config.fp_task, device_id_, stream_)) {
            release();
            return false;
        }
        d_projection_model_.allocate(projection_count_, device_id_);
        d_working_projection_.allocate(projection_count_, device_id_);
        d_outer_start_.allocate(volume_count_, device_id_);
        d_regularization_start_.allocate(volume_count_, device_id_);
        d_data_update_.allocate(volume_count_, device_id_);
        d_regularization_update_.allocate(volume_count_, device_id_);
        d_tv_gradient_.allocate(volume_count_, device_id_);
        if (config.relaxation_mode == ETigreRelaxationMode::Nesterov) {
            d_nesterov_previous_.allocate(volume_count_, device_id_);
            d_nesterov_candidate_.allocate(volume_count_, device_id_);
        }
        prepared_ = true;
        resetState_();
        return true;
    }

    bool reconstruct(const float* d_measured_projection, float* d_volume)
    {
        if (!prepared_ || !d_measured_projection || !d_volume) return false;
        resetState_();
        YK_CUDA_CHECK(cudaMemcpyAsync(d_working_projection_, d_measured_projection,
            projection_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
        if (!initializeVolume_(d_measured_projection, d_volume)) return false;
        if (isPocs_(config_.algorithm) && config_.max_l2_error < 0.f)
            maximum_l2_error_ = estimateMaximumL2Error_(d_measured_projection);
        else
            maximum_l2_error_ = config_.max_l2_error;

        if (isBasicArt_(config_.algorithm))
            return runBasicArt_(d_working_projection_, d_volume);
        return runPocs_(d_measured_projection, d_volume);
    }

    const TigreGradientStatistics& statistics() const { return statistics_; }
    int actualSubsetCount() const { return actual_subset_count_; }

    void reset() { data_backend_.reset(); resetState_(); }

    void release()
    {
        if (stream_)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        data_backend_.release();
        fp_.release();
        d_projection_model_.reset();
        d_working_projection_.reset();
        d_outer_start_.reset();
        d_regularization_start_.reset();
        d_data_update_.reset();
        d_regularization_update_.reset();
        d_tv_gradient_.reset();
        d_nesterov_previous_.reset();
        d_nesterov_candidate_.reset();
        params_ = {};
        geometry_.clear();
        config_ = {};
        stream_ = nullptr;
        volume_count_ = projection_count_ = 0;
        requested_subset_count_ = actual_subset_count_ = 0;
        prepared_ = false;
        resetState_();
    }

private:
    static bool isBasicArt_(ETigreGradientAlgorithm algorithm)
    {
        return algorithm == ETigreGradientAlgorithm::Sart ||
            algorithm == ETigreGradientAlgorithm::OsSart ||
            algorithm == ETigreGradientAlgorithm::Sirt;
    }

    static bool isPocs_(ETigreGradientAlgorithm algorithm)
    { return !isBasicArt_(algorithm); }

    static bool isPcsd_(ETigreGradientAlgorithm algorithm)
    {
        return algorithm == ETigreGradientAlgorithm::Pcsd ||
            algorithm == ETigreGradientAlgorithm::OsPcsd ||
            algorithm == ETigreGradientAlgorithm::AwPcsd ||
            algorithm == ETigreGradientAlgorithm::OsAwPcsd;
    }

    static bool isAdaptive_(ETigreGradientAlgorithm algorithm)
    {
        return algorithm == ETigreGradientAlgorithm::AwPcsd ||
            algorithm == ETigreGradientAlgorithm::OsAwPcsd ||
            algorithm == ETigreGradientAlgorithm::AwAsdPocs ||
            algorithm == ETigreGradientAlgorithm::OsAwAsdPocs;
    }

    static bool isBregman_(ETigreGradientAlgorithm algorithm)
    { return algorithm == ETigreGradientAlgorithm::BAsdPocsBeta; }

    static int resolvedBlockSize_(const Config& config, int angle_count)
    {
        switch (config.algorithm) {
        case ETigreGradientAlgorithm::Sirt:
            return angle_count;
        case ETigreGradientAlgorithm::Sart:
        case ETigreGradientAlgorithm::AsdPocs:
        case ETigreGradientAlgorithm::BAsdPocsBeta:
        case ETigreGradientAlgorithm::Pcsd:
        case ETigreGradientAlgorithm::AwPcsd:
        case ETigreGradientAlgorithm::AwAsdPocs:
            return 1;
        default:
            return config.block_size;
        }
    }

    bool initializeVolume_(const float* d_projection, float* d_volume)
    {
        if (config_.initialization == ETigreInitialization::Zero) {
            YK_CUDA_CHECK(cudaMemsetAsync(d_volume, 0,
                volume_count_ * sizeof(float), stream_));
            return true;
        }
        if (config_.initialization == ETigreInitialization::DeviceVolume) {
            YK_CUDA_CHECK(cudaMemcpyAsync(d_volume, config_.d_initial_volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            return true;
        }
        return reconstructFdk_(d_projection, d_volume);
    }

    bool reconstructFdk_(const float* d_projection, float* d_volume)
    {
        std::vector<float> host_projection(projection_count_);
        YK_CUDA_CHECK(cudaMemcpyAsync(host_projection.data(), d_projection,
            projection_count_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        FdkPipeline fdk;
        if (!fdk.prepareWithGeometry(params_, geometry_,
                std::min(params_.scan.NAng, kMaxChunkAng), stream_, device_id_))
            return false;
        FdkProjectionBatch batch{};
        batch.projection = host_projection.data();
        batch.geometry = &geometry_;
        batch.count = params_.scan.NAng;
        const bool ok = fdk.processBatch(batch, d_volume, true);
        // FdkPipeline 的 stage 和临时缓冲区归它所有；局部对象释放前必须
        // 等待本 stream 上的异步 kernel 完成，否则初始化路径会过早释放资源。
        if (ok)
            YK_CUDA_CHECK(cudaStreamSynchronize(stream_));
        fdk.release();
        if (ok && config_.non_negative)
            clamp_min_launch(d_volume, volume_count_, 0.f, stream_);
        return ok;
    }

    float estimateMaximumL2Error_(const float* d_projection)
    {
        DeviceWorkspaceF32 d_fdk;
        d_fdk.allocate(volume_count_, device_id_);
        float estimate = 0.f;
        if (reconstructFdk_(d_projection, d_fdk)) {
            estimate = 0.2f * projectionResidualNorm_(d_projection, d_fdk);
        }
        else {
            float norm2 = 0.f;
            dot_launch(d_projection, d_projection, projection_count_, &norm2, stream_);
            estimate = 0.2f * std::sqrt(norm2);
            YK_LOGW("[TigreGradient] 当前 geometry 无法使用 FDK 自动估计 max_l2_error，"
                "退回 0.2*||b||_2={:.6e}", estimate);
        }
        d_fdk.reset();
        return estimate;
    }

    float projectionResidualNorm_(const float* d_projection, const float* d_volume)
    {
        YK_CUDA_CHECK(cudaMemsetAsync(d_projection_model_, 0,
            projection_count_ * sizeof(float), stream_));
        fp_.run(d_volume, params_, d_projection_model_, stream_);
        residual_launch(d_projection, d_projection_model_, d_projection_model_,
            projection_count_, stream_);
        float norm2 = 0.f;
        dot_launch(d_projection_model_, d_projection_model_, projection_count_,
            &norm2, stream_);
        return std::sqrt(norm2);
    }

    float vectorNorm_(const float* vector)
    {
        float norm2 = 0.f;
        dot_launch(vector, vector, volume_count_, &norm2, stream_);
        return std::sqrt(norm2);
    }

    float directionCosine_(const float* a, const float* b,
        float norm_a, float norm_b)
    {
        float dot = 0.f;
        dot_launch(a, b, volume_count_, &dot, stream_);
        return dot / std::max(norm_a * norm_b, 1e-6f);
    }

    bool runDataOuter_(const float* d_projection, float* d_volume, float beta)
    {
        data_backend_.setRelaxation(beta);
        if (config_.relaxation_mode != ETigreRelaxationMode::Nesterov)
            return data_backend_.iterate(d_projection, d_volume, params_, stream_,
                static_cast<unsigned int>(actual_subset_count_));

        for (int subset = 0; subset < actual_subset_count_; ++subset) {
            if (!data_backend_.iterate(d_projection, d_volume, params_, stream_, 1))
                return false;
            YK_CUDA_CHECK(cudaMemcpyAsync(d_nesterov_candidate_, d_volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            scale_launch(d_volume, 1.f - nesterov_gamma_, volume_count_, stream_);
            axpy_launch(d_volume, d_nesterov_previous_, nesterov_gamma_,
                volume_count_, stream_);
            YK_CUDA_CHECK(cudaMemcpyAsync(d_nesterov_previous_, d_nesterov_candidate_,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            if (config_.non_negative)
                clamp_min_launch(d_volume, volume_count_, 0.f, stream_);
        }
        return true;
    }

    bool runBasicArt_(const float* d_projection, float* d_volume)
    {
        float beta = config_.lambda;
        float previous_residual = std::numeric_limits<float>::infinity();
        for (int outer = 0; outer < config_.iterations; ++outer) {
            if (!runDataOuter_(d_projection, d_volume,
                    config_.relaxation_mode == ETigreRelaxationMode::Nesterov ? 1.f : beta))
                return false;
            statistics_.completed_iterations = outer + 1;
            statistics_.subset_updates = data_backend_.totalIterations();
            if (config_.relaxation_mode == ETigreRelaxationMode::Nesterov) {
                const float residual = projectionResidualNorm_(d_projection, d_volume);
                statistics_.projection_l2 = residual;
                if (outer > 0 && residual > previous_residual) {
                    statistics_.stopped_by_nesterov_residual = true;
                    break;
                }
                previous_residual = residual;
                const float next = 0.5f * (1.f +
                    std::sqrt(1.f + 4.f * nesterov_t_ * nesterov_t_));
                nesterov_gamma_ = (1.f - nesterov_t_) / next;
                nesterov_t_ = next;
            }
            else {
                beta *= config_.lambda_reduction;
            }
        }
        statistics_.beta = beta;
        return true;
    }

    bool runPocs_(const float* d_original_projection, float* d_volume)
    {
        float beta = config_.lambda;
        float dtvg = 0.f;
        float first_projection_distance = 1.f;
        float bregman = config_.bregman_beta;

        for (int outer = 0; outer < config_.iterations; ++outer) {
            YK_CUDA_CHECK(cudaMemcpyAsync(d_outer_start_, d_volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));

            const float projection_distance_before = isPcsd_(config_.algorithm)
                ? projectionResidualNorm_(d_working_projection_, d_volume) : 0.f;
            // TIGRE PCSD/AwPCSD 的公开实现使用 d_p^2 > maxl2err。
            // 虽然参数名容易让人理解为直接比较 L2 范数，这里仍保留其
            // 历史数值语义，以便同一组参数可与 TIGRE 对照。
            const bool execute_data = !isPcsd_(config_.algorithm) ||
                projection_distance_before * projection_distance_before > maximum_l2_error_;
            if (execute_data && !runDataOuter_(d_working_projection_, d_volume, beta))
                return false;

            statistics_.projection_l2 = projectionResidualNorm_(
                d_working_projection_, d_volume);
            subtract_launch(d_data_update_, d_volume, d_outer_start_,
                volume_count_, stream_);
            const float dp = vectorNorm_(d_data_update_);

            float step = 0.f;
            if (isPcsd_(config_.algorithm)) {
                step = outer == 0 ? 1.f :
                    projection_distance_before / std::max(first_projection_distance, 1e-6f);
                if (outer == 0)
                    first_projection_distance = projectionResidualNorm_(
                        d_working_projection_, d_volume);
            }
            else {
                if (outer == 0) dtvg = config_.alpha * dp;
                step = dtvg;
            }

            YK_CUDA_CHECK(cudaMemcpyAsync(d_regularization_start_, d_volume,
                volume_count_ * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
            tigre_tv_descent_launch(d_volume, d_tv_gradient_,
                params_.volume.Nx, params_.volume.Ny, params_.volume.Nz,
                step, config_.tv_iterations, config_.tv_epsilon,
                isAdaptive_(config_.algorithm), config_.adaptive_delta, stream_);
            if (config_.non_negative)
                clamp_min_launch(d_volume, volume_count_, 0.f, stream_);
            subtract_launch(d_regularization_update_, d_volume,
                d_regularization_start_, volume_count_, stream_);
            const float dg = vectorNorm_(d_regularization_update_);
            const float cosine = directionCosine_(d_regularization_update_,
                d_data_update_, dg, dp);

            if (!isPcsd_(config_.algorithm) &&
                dg > config_.maximum_update_ratio * dp &&
                statistics_.projection_l2 > maximum_l2_error_)
                dtvg *= config_.alpha_reduction;
            beta *= config_.lambda_reduction;

            statistics_.completed_iterations = outer + 1;
            statistics_.subset_updates = data_backend_.totalIterations();
            statistics_.data_update_l2 = dp;
            statistics_.regularization_update_l2 = dg;
            statistics_.direction_cosine = cosine;
            statistics_.beta = beta;
            statistics_.tv_step = step;

            if (isBregman_(config_.algorithm) &&
                (outer + 1) % config_.bregman_interval == 0) {
                // b <- b + mu * (b - A x)，与 TIGRE MATLAB 保持相同的
                // 递推投影语义；原始调用者投影保持只读。
                YK_CUDA_CHECK(cudaMemsetAsync(d_projection_model_, 0,
                    projection_count_ * sizeof(float), stream_));
                fp_.run(d_volume, params_, d_projection_model_, stream_);
                residual_launch(d_working_projection_, d_projection_model_,
                    d_projection_model_, projection_count_, stream_);
                axpy_launch(d_working_projection_, d_projection_model_, bregman,
                    projection_count_, stream_);
                bregman *= config_.bregman_beta_reduction;
            }

            if (cosine < -0.99f && statistics_.projection_l2 <= maximum_l2_error_) {
                statistics_.stopped_by_projection_condition = true;
                break;
            }
            if (beta < config_.minimum_beta) {
                statistics_.stopped_by_beta = true;
                break;
            }
        }
        (void)d_original_projection;
        return true;
    }

    void resetState_()
    {
        statistics_ = {};
        statistics_.beta = config_.lambda;
        maximum_l2_error_ = config_.max_l2_error;
        nesterov_t_ = 0.5f * (1.f + std::sqrt(5.f));
        nesterov_gamma_ = 0.f;
        if (d_nesterov_previous_)
            cudaMemsetAsync(d_nesterov_previous_, 0,
                volume_count_ * sizeof(float), stream_);
    }

    static bool validate_(const SReconstructionParams& params,
        const std::vector<SConeProjGeomVec>& geometry,
        const Config& config, cudaStream_t stream)
    {
        if (!stream || params.scan.NAng <= 0 || params.scan.Nu <= 0 || params.scan.Nv <= 0 ||
            params.volume.Nx <= 0 || params.volume.Ny <= 0 || params.volume.Nz <= 0 ||
            static_cast<int>(geometry.size()) != params.scan.NAng ||
            config.iterations <= 0 || config.block_size <= 0 ||
            config.lambda <= 0.f || config.lambda_reduction <= 0.f ||
            config.tv_iterations <= 0 || config.alpha <= 0.f ||
            config.alpha_reduction <= 0.f || config.maximum_update_ratio <= 0.f ||
            config.minimum_beta <= 0.f || config.tv_epsilon <= 0.f ||
            config.bregman_interval <= 0 || config.bregman_beta < 0.f ||
            config.bregman_beta_reduction <= 0.f ||
            (config.relaxation_mode == ETigreRelaxationMode::Nesterov &&
                !isBasicArt_(config.algorithm)) ||
            (config.initialization == ETigreInitialization::DeviceVolume &&
                !config.d_initial_volume)) {
            YK_LOGE("[TigreGradient] 无效配置");
            return false;
        }
        return true;
    }

    SReconstructionParams params_{};
    std::vector<SConeProjGeomVec> geometry_{};
    Config config_{};
    TigreGradientStatistics statistics_{};
    cudaStream_t stream_ = nullptr;
    int device_id_ = 0;
    size_t volume_count_ = 0;
    size_t projection_count_ = 0;
    int requested_subset_count_ = 0;
    int actual_subset_count_ = 0;
    float maximum_l2_error_ = -1.f;
    float nesterov_t_ = 0.f;
    float nesterov_gamma_ = 0.f;
    bool prepared_ = false;

    AlgebraicTigreBackend data_backend_{};
    ForwardOperatorAdapter fp_{};
    DeviceWorkspaceF32 d_projection_model_{};
    DeviceWorkspaceF32 d_working_projection_{};
    DeviceWorkspaceF32 d_outer_start_{};
    DeviceWorkspaceF32 d_regularization_start_{};
    DeviceWorkspaceF32 d_data_update_{};
    DeviceWorkspaceF32 d_regularization_update_{};
    DeviceWorkspaceF32 d_tv_gradient_{};
    DeviceWorkspaceF32 d_nesterov_previous_{};
    DeviceWorkspaceF32 d_nesterov_candidate_{};
};

} // namespace YK::Iter
