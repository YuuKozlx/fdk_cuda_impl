#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
namespace yk::spectral {
enum class GeometryKind { FlatCbct, FlatHelical, CylCbct, CylHelical };
enum class WorkflowMode { Project, Reconstruct, ProjectAndReconstruct };
struct WorkflowConfig { WorkflowMode mode = WorkflowMode::Project; };
// 材料可以用 formula 表示单一化合物，也可以用 preset 引用内置的
// 元素质量分数表。preset 适合人体组织等不能用单一化学式准确表达的材料。
struct MaterialSpec { std::uint8_t label=0; std::string name; std::string formula; std::string preset; double density_g_cm3=1.0; };
struct SpectrumPoint { double energy_keV=0.0; double relative_photons=0.0; };
struct DetectorEffectsConfig {
    bool efficiency_enabled=false; double efficiency=1.0; bool scatter_enabled=false;
    bool optical_crosstalk_enabled=false; bool afterglow_enabled=false;
    bool electronic_noise_enabled=false; double electronic_noise_sigma=0.0;
};

struct GeometryConfig {
    int views = 360;
    int views_per_turn = 360;
    int rotation_direction = 1;
    int detector_u = 256;
    int detector_v = 128;
    double pixel_u_mm = 1.0;
    double pixel_v_mm = 1.0;
    double sid_mm = 500.0;
    double sdd_mm = 1000.0;
    // 探测器局部坐标偏移：U（通道切向）、N（表面法向）、V（行方向）。
    double offset_u_mm = 0.0;
    double offset_n_mm = 0.0;
    double offset_v_mm = 0.0;
    // Detector pose perturbation in the local U/V/N frame, in radians.
    double tilt_u_rad = 0.0;
    double tilt_v_rad = 0.0;
    double tilt_n_rad = 0.0;
    // 焦点相对标称旋转轨迹的 XYZ 偏移，单位为 mm。
    double source_offset_x_mm = 0.0;
    double source_offset_y_mm = 0.0;
    double source_offset_z_mm = 0.0;
    double start_angle_rad = 0.0;
    double pitch_mm_per_turn = 0.0;
    double start_z_mm = 0.0;
    double voxel_x_mm = 1.0;
    double voxel_y_mm = 1.0;
    double voxel_z_mm = 1.0;
    int volume_x = 0;
    int volume_y = 0;
    int volume_z = 0;
    int reconstruction_volume_x = 0;
    int reconstruction_volume_y = 0;
    int reconstruction_volume_z = 0;
    double reconstruction_voxel_x_mm = 1.0;
    double reconstruction_voxel_y_mm = 1.0;
    double reconstruction_voxel_z_mm = 1.0;
    // 标签模体和重建网格分别拥有独立世界坐标中心，单位为 mm。
    double phantom_offset_x_mm = 0.0;
    double phantom_offset_y_mm = 0.0;
    double phantom_offset_z_mm = 0.0;
    double phantom_rotation_x_rad = 0.0;
    double phantom_rotation_y_rad = 0.0;
    double phantom_rotation_z_rad = 0.0;
    double reconstruction_offset_x_mm = 0.0;
    double reconstruction_offset_y_mm = 0.0;
    double reconstruction_offset_z_mm = 0.0;
};
struct FocalSpotConfig {
    bool enabled = false;
    double size_u_mm = 0.0;
    double size_v_mm = 0.0;
};

struct SamplingConfig {
    std::string mode = "pixel_local_random";
    std::string photon_count_mode = "poisson";
    int samples_per_pixel = 1;
    std::uint64_t total_samples = 0;
    double photons_per_pixel = 100000.0;
    std::uint32_t seed = 12345;
    std::uint32_t photon_seed = 67890;
};

struct GeometrySpec {
    GeometryKind kind = GeometryKind::FlatCbct;
    GeometryConfig parameters;
};

struct ProjectionConfig {
    std::filesystem::path label_volume;
    std::filesystem::path spectrum_file;
    std::filesystem::path xcom_data_directory;
    std::filesystem::path output_file;
    std::filesystem::path energy_output_file;
    bool apply_geometry_flux = true;
    std::string engine = "pixel_local_random";
    std::vector<MaterialSpec> materials;
    DetectorEffectsConfig detector_effects;
    FocalSpotConfig focal_spot;
    SamplingConfig sampling;
};

// 仿真结束后的可选重建设置。投影仍按文件流式读取；FDK 使用 chunk_views
// 分包，wFBP/迭代管线要求单次提交完整投影，不改变投影文件布局。
struct AnalyticReconstructionConfig {
    std::string pipeline = "fdk";
    std::string filter = "shepp-logan";
    double wfbp_cutoff = 1.0;
    double wfbp_apodization = 1.0;
    int chunk_views = 32;
};

struct IterativeReconstructionConfig {
    std::string algorithm = "os_sart_tv";
    int iterations = 10;
    double relaxation = 1.0;
    int subsets = 1;
    std::string projection_model = "joseph";
    int tv_iterations = 20;
    double tv_alpha = 0.002;
    double tv_alpha_reduction = 0.95;
    double maximum_update_ratio = 0.95;
    bool non_negative = true;
};

struct ReconstructionConfig {
    std::filesystem::path input_projection_file;
    std::string type = "analytic";
    AnalyticReconstructionConfig analytic;
    IterativeReconstructionConfig iterative;
    std::filesystem::path output_volume_file;
    std::filesystem::path slice_prefix;
    bool save_slices = false;
};
}
