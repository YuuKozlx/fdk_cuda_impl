#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "global/YkCBCTParams.h"

namespace YK::TestConfig {

enum class Pipeline {
    Fdk, Cfdk, Xfdk, Sirt, Sart, Ossart, Cgls, TigreGradientLocal, Pwls,
    FdkOssart, FdkCgls
};
enum class InputMode { ProjectionRaw, Phantom };

struct InputConfig {
    InputMode mode = InputMode::Phantom;
    std::string phantom = "catphan";
    // 解析模体参数。size_fraction 表示相对可用 FOV 的直径或边长，
    // height_fraction 只用于沿 Z 延伸的圆柱和方柱。
    float phantom_value = 0.02f;
    float phantom_size_fraction = 0.65f;
    float phantom_height_fraction = 0.80f;
    std::filesystem::path projection{};
    // 可选空气场。提供后按 -log(clamp(I/I0, minimum_ratio, 1)) 转为线积分。
    std::filesystem::path air{};
    float minimum_ratio = 1e-6f;
};

struct AlgorithmConfig {
    int iterations = 1;
    int subsets = 2;
    float relaxation = 0.2f;
    float relaxation_reduction = 1.f;
    float epsilon = 1e-6f;
    bool use_min = true;
    float minimum = 0.f;
    bool use_max = false;
    float maximum = 1e30f;
    std::string forward_projector = "joseph";
    std::string back_projector = "joseph-v3";
    std::string weight_model = "detailed";
    std::string subset_order = "sequential";
    std::string cgls_strategy = "robust-restart";
    bool restart_on_divergence = true;

    std::string regularizer = "none";
    float regularization_strength = 0.f;
    int regularization_iterations = 5;
    std::string tv_dimensionality = "3d";
    float regularization_epsilon = 1e-6f;
    float regularization_reduction = 1.f;

    float relative_residual_tolerance = 0.f;
    float relative_update_tolerance = 0.f;
    float relative_improvement_tolerance = 0.f;
    int minimum_iterations = 1;
    int check_interval = 1;
    int patience = 1;
};

struct TigreConfig {
    std::string method = "os-sart";
    int block_size = 20;
    float lambda = 1.f;
    float lambda_reduction = 1.f;
    std::string relaxation_mode = "scalar";
    std::string initialization = "zero";
    bool non_negative = true;
    int tv_iterations = 20;
    float alpha = 0.002f;
    float alpha_reduction = 0.95f;
    float maximum_update_ratio = 0.95f;
    float max_l2_error = -1.f;
    float minimum_beta = 0.005f;
    float tv_epsilon = 1e-8f;
    float adaptive_delta = -0.005f;
    float bregman_beta = 1.f;
    float bregman_beta_reduction = 0.75f;
    int bregman_interval = 5;
};

struct PwlsCaseConfig {
    int subsets = 1;
    float relaxation = 0.8f;
    std::string regularizer = "quadratic";
    float regularization = 1e-3f;
    float huber_delta = 3e-3f;
    float epsilon = 1e-6f;
    float lower_bound = 0.f;
    float upper_bound = 1e30f;
};

struct ReconstructionCase {
    std::string name{};
    Pipeline pipeline = Pipeline::Fdk;
    int device = 0;
    int chunk_views = 32;
    int batch_views = 32;
    SReconstructionParams params{};
    InputConfig input{};
    AlgorithmConfig algorithm{};
    TigreConfig tigre{};
    PwlsCaseConfig pwls{};
    std::filesystem::path output{};
    std::filesystem::path preview{};
};

// 相对输入和输出路径始终相对于 TOML 文件所在目录解析，避免 IDE 与命令行
// 使用不同工作目录时读取到不同数据。
std::vector<ReconstructionCase> loadCases(const std::filesystem::path& file);

// case_name 为空时按配置顺序执行全部任务；非空时只运行同名任务。
int runConfiguredReconstruction(const std::filesystem::path& file,
    const std::string& case_name);

} // namespace YK::TestConfig
