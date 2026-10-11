#pragma once
#include "config/SpectrumSourceConfig.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
namespace yk::spectral {
enum class GeometryKind {
    FlatCbct, FlatHelical, CylCbct, CylHelical,
    PlanarSourceOrbit, NikonTilted,
};
enum class WorkflowMode { Project, Reconstruct, ProjectAndReconstruct };
struct WorkflowConfig { WorkflowMode mode = WorkflowMode::Project; };
struct VisualizationConfig {
    bool enabled = false;
    bool interactive = false;
    int texture_max_size = 512;
    double max_fps = 15;
    std::string projection = "signal"; // signal, primary, log
    bool hold_open = true;
    bool offscreen = false;
    std::filesystem::path snapshot_directory;
};
// 材料可以用 formula 表示单一化合物，也可以用 preset 引用内置的
// 元素质量分数表。preset 适合人体组织等不能用单一化学式准确表达的材料。
struct MaterialSpec { std::uint8_t label=0; std::string name; std::string formula; std::string preset; double density_g_cm3=1.0; };
struct SpectrumPoint { double energy_keV=0.0; double fluence_per_keV_cm2_mAs_at_1m=0.0; double bin_width_keV=0.0; };
struct DetectorLayerConfig {
    bool enabled = true;
    std::string name;
    std::string preset;
    double density_g_cm3 = 1.0;
    double thickness_mm = 0.0;
    std::filesystem::path thickness_map_file;
};
struct DetectorResponseConfig {
    bool enabled = false;
    bool oblique_path_correction = true;
    DetectorLayerConfig protective_layer;
    DetectorLayerConfig impurity_layer;
    DetectorLayerConfig scintillator_layer;
};
struct CrosstalkConfig {
    bool enabled = false;
    std::vector<double> kernel_u{1.0};
    std::vector<double> kernel_v{1.0};
};
struct DetectorPostprocessConfig {
    CrosstalkConfig optical_crosstalk;
    struct AfterglowConfig {
        bool enabled = false;
        double p = 0.0;
        std::string initialization = "zero";
        double frame_time_ms = 1.0;
        double exposure_time_ms = 1.0;
        std::vector<double> weights;
        std::vector<double> time_constants_ms;
    } afterglow;
    CrosstalkConfig electronic_crosstalk;
    struct DasConfig {
        bool enabled = false;
        double gain_electrons_per_keV = 1.0;
        double offset_electrons = 0.0;
        double electronic_noise_std_electrons = 0.0;
        double saturation_electrons = 0.0;
        double adc_lsb_electrons = 0.0;
        std::uint32_t noise_seed = 24680;
    } das;
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
    // Parameters for non-standard flat-detector trajectories.
    double scan_range_rad = 6.28318530717958647692;
    double planar_detector_y_mm = 0.0;
    double source_orbit_radius_mm = 0.0;
    double source_lateral_offset_mm = 0.0;
    int reconstruction_detector_u = 0;
    int reconstruction_detector_v = 0;
    std::string planar_resample_mode = "perspective_orthogonal";
    double nikon_tilt_rad = 0.0;
    double nikon_tilt_axis_azimuth_rad = 0.0;
    double rotation_axis_pivot_x_mm = 0.0;
    double rotation_axis_pivot_y_mm = 0.0;
    double rotation_axis_pivot_z_mm = 0.0;
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
    double normal_size_u_mm = 0.0;
    double normal_size_v_mm = 0.0;
    double defocus_size_u_mm = 0.0;
    double defocus_size_v_mm = 0.0;
    // Defocus exposure fraction; normal exposure fraction is 1 - epsilon.
    double epsilon = 0.0;
};

struct GeometryFluxConfig {
    bool enabled = true;
};

struct QuantumNoiseConfig {
    bool enabled = true;
    std::uint32_t seed = 67890;
};

struct SamplingConfig {
    int samples_per_pixel = 1;
    std::uint32_t seed = 12345;
};

struct ScatterConfig {
    bool enabled = false;
    std::filesystem::path model_file;
    struct KernelConfig {
        std::string mode = "all";
        int segments = 3;
    } kernel;
};

struct GeometrySpec {
    GeometryKind kind = GeometryKind::FlatCbct;
    GeometryConfig parameters;
};

struct ProjectionConfig {
    SpectrumSourceConfig spectrum;
    std::filesystem::path label_volume;
    std::filesystem::path spectrum_file;
    std::filesystem::path output_file;
    std::filesystem::path energy_output_file;
    std::filesystem::path air_output_file;
    bool save_metadata = true;
    double mAs_per_view = 1.0;
    GeometryFluxConfig geometry_flux;
    QuantumNoiseConfig quantum_noise;
    std::vector<MaterialSpec> materials;
    DetectorResponseConfig detector_response;
    DetectorPostprocessConfig detector_postprocess;
    FocalSpotConfig focal_spot;
    SamplingConfig sampling;
    ScatterConfig scatter;
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

struct BeamHardeningConfig {
    bool enabled = false;
    bool fit_from_water_plate = false;
    std::filesystem::path spectrum_file;
    int order = 1;
    std::vector<double> coefficients{0.0, 1.0};
    double clamp_min = 0.0;
    double clamp_max = 20.0;
    double water_plate_max_mm = 400.0;
    int water_plate_samples = 81;
    double reference_energy_keV = 0.0;
};

struct ScatterCorrectionConfig {
    bool enabled = false;
    std::filesystem::path air_projection_file;
    std::filesystem::path model_file;
    struct CorrectionConfig {
        int iterations = 3;
        double damping = 0.7;
        double scatter_scale = 1.0;
        double object_mask_threshold = 1.0e-6;
    } correction;
    struct GuideConfig {
        bool enabled = false;
        double sigma_u_pixels = 2.0;
        double sigma_v_pixels = 2.0;
    } guide;
    struct SolverConfig {
        std::string type = "fixed";
        struct AdaptiveConfig {
            int iterations = 10;
            double initial_damping = 0.7;
            double min_damping = 0.1;
            double damping_decay = 1.0;
            struct TransmissionGuardConfig {
                double low_quantile = 0.01;
                double risk_threshold = 0.02;
                double positivity_margin = 0.9;
            } transmission_guard;
        } adaptive;
    } solver;
    struct KernelConfig {
        std::string mode = "all";
        int segments = 3;
    } kernel;
};

struct ReconstructionPreprocessConfig {
    bool enabled = false;
    BeamHardeningConfig beam_hardening;
    ScatterCorrectionConfig scatter;
};

struct ReconstructionConfig {
    std::filesystem::path input_projection_file;
    std::string type = "analytic";
    AnalyticReconstructionConfig analytic;
    IterativeReconstructionConfig iterative;
    std::filesystem::path output_volume_file;
    std::filesystem::path slice_prefix;
    bool save_slices = false;
    ReconstructionPreprocessConfig preprocessing;
};
}
