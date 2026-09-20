#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
namespace yk::spectral {
enum class GeometryKind { FlatCbct, FlatHelical, CylCbct, CylHelical };
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
    // 标签模体和重建网格分别拥有独立世界坐标中心，单位为 mm。
    double phantom_offset_x_mm = 0.0;
    double phantom_offset_y_mm = 0.0;
    double phantom_offset_z_mm = 0.0;
    double reconstruction_offset_x_mm = 0.0;
    double reconstruction_offset_y_mm = 0.0;
    double reconstruction_offset_z_mm = 0.0;
};

// 仿真结束后的可选重建设置。投影仍按文件流式读取，chunk_views
// 只控制单次 DLL execute() 的视图数，不改变投影文件布局。
struct ReconstructionConfig {
    bool enabled = false;
    // 保留配置层算法选择。当前示例真正执行的 DLL 重建管线仍由
    // LibraryReconstructor 校验；未知或尚未接入的值不得静默降级。
    std::string pipeline = "fdk";
    std::string filter = "shepp-logan";
    // FreeCT 空间域核参数，与 Flat FDK 的滤波枚举独立。
    double wfbp_cutoff = 1.0;
    double wfbp_apodization = 1.0;
    std::filesystem::path output_volume_file;
    std::filesystem::path slice_prefix;
    int chunk_views = 32;
};
}
