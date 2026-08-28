#pragma once

#include <vector>

#include "YkFilterTypes.hpp"

// 标称扫描参数只服务于几何构造、采集顺序和投影数组布局。逐视图 vector
// geometry 一旦生成，后续算子不得再用这些标称量覆盖实际空间几何。
struct SScanGeometryConfig {
    std::vector<float> angles{};
    int Nu = 0;
    int Nv = 0;
    int NAng = 0;
    int totalViews = 0;
    float du_mm = 1.f;
    float dv_mm = 1.f;
    float offsetU_mm = 0.f;
    float offsetV_mm = 0.f;
    float sourceOffsetX_mm = 0.f;
    float sourceOffsetY_mm = 0.f;
    float sourceOffsetZ_mm = 0.f;
    float tiltN_rad = 0.f;
    float tiltU_rad = 0.f;
    float tiltV_rad = 0.f;
    float range_rad = 6.28318530717958647692f;
    float start_angle_rad = 0.f;
    bool short_scan = false;
    int direction = 1;
    float sid_mm = 0.f;
    float sdd_mm = 0.f;
};

// 重建体积独立于扫描仪标称参数，中心为 object/world 坐标。
struct SVolumeGeometryConfig {
    int Nx = 0;
    int Ny = 0;
    int Nz = 0;
    float voxelX_mm = 1.f;
    float voxelY_mm = 1.f;
    float voxelZ_mm = 1.f;
    float centerX_mm = 0.f;
    float centerY_mm = 0.f;
    float centerZ_mm = 0.f;
};

struct SReconstructionConfig {
    YK::SFilterKernelDesc filter = YK::SFilterKernelDesc::RamLak(
        YK::EWeightsBuildSource::AnalyticFreq);
};

// 管线入口可以整体传递该对象，但模块内部必须按职责读取其中一个子配置。
struct SReconstructionParams {
    SScanGeometryConfig scan{};
    SVolumeGeometryConfig volume{};
    SReconstructionConfig reconstruction{};
};
