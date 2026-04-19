#pragma once
#include "global/YkGlobals.h"
#include "YKCBCT/interface/YkTaskTypes.hpp"

using namespace YK;

struct SHeliCTParam {
    // ---- 探测器 ----
    int   iPU = 1024;
    int   iPV = 32;
    float du_mm = 0.25f;
    float dv_mm = 0.25f;
    float offsetU_mm = 0.f;
    float offsetV_mm = 0.f;   // 新增

    // ---- 几何 ----
    float SID = 500.f;
    float SDD = 1000.f;

    // ---- 体积 ----
    int   iVX = 512;
    int   iVY = 512;
    int   iVZ = 400;
    float vox_x_mm = 0.1f;
    float vox_y_mm = 0.1f;
    float vox_z_mm = 0.1f;
    float vol_offset_x_mm = 0.f;
    float vol_offset_y_mm = 0.f;
    float vol_offset_z_mm = 0.f;

    // ---- 探测器倾斜 ----
    float tiltu_angle_rad = 0.f;
    float tiltn_angle_rad = 0.f;
    float tiltv_angle_rad = 0.f;

    // ---- 螺旋扫描 ----
    float              pitch_mm = 3.f;
    float              start_z_mm = 0.f;
    std::vector<float> angle_list;  // 新增：外部传入，仿真或真实数据均用此

    // ---- 重建 ----
    bool  bShortScan = false;
    float z_block_mm = 3.f;
    float z_step_mm = 1.0f;
    int   Kchunk = 32;
    ETask fp_task = ETask::FP_Joseph;
    YK::SFilterKernelDesc desc =
        YK::SFilterKernelDesc(YK::EFilterKernel::RamLak);
};